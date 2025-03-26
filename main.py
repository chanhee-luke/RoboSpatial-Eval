# main.py for RoboSpatial Evaluation

import os
import sys
import json
import base64
import random
import numpy as np
from tqdm import tqdm
import yaml
import argparse
import re
import ast

# Import evaluation functions - these are always needed
from evaluation import (
    eval_robospatial_home,
    eval_pregenerated_results
)

# ------------------------------------------------------------------------
# 2) General Utility Functions
# ------------------------------------------------------------------------
def encode_image(image_path):
    """Base64-encode an image file (for GPT-based or generic usage)."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def import_model_modules(model_name):
    """
    Dynamically import only the required modules for the specified model.
    Returns the appropriate model loading and running functions.
    """
    if model_name == "llava_next":
        from models import load_llava_next_model, run_llava_next
        return load_llava_next_model, run_llava_next
    elif model_name.startswith("spatialvlm"):
        from models import load_spatialvlm_model, run_spatialvlm
        return load_spatialvlm_model, run_spatialvlm
    elif model_name.startswith("robopoint"):
        from models import load_robopoint_model, run_robopoint
        return load_robopoint_model, run_robopoint
    elif model_name.startswith("qwen2vl"):
        from models import load_qwen2vl_model, run_qwen2vl
        return load_qwen2vl_model, run_qwen2vl
    elif model_name.startswith("molmo"):
        from models import load_molmo_model, run_molmo
        return load_molmo_model, run_molmo
    elif model_name.startswith("gpt"):
        from models import load_gpt_model, send_question_to_openai
        return load_gpt_model, send_question_to_openai
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def load_model(model_name, model_path=None):
    """
    Load the chosen model once, returning any needed kwargs (tokenizer, model object, etc.).
    If model_path is provided, it overrides the default model checkpoint.
    """
    # Import torch only when we actually need to load a model
    import torch
    global device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    # Dynamically import only what we need
    load_func, _ = import_model_modules(model_name)
    return load_func(model_path)

def run_model(question, image_path, model_name, model_kwargs):
    """
    Generic helper: runs the specified model on a single (question, image).
    Returns the string answer from the model.
    """
    if not os.path.exists(image_path):
        # Just a warning if the path doesn't exist
        print(f"[WARNING] Image path not found: {image_path}")

    # Access the correct run function for this model
    _, run_func = import_model_modules(model_name)
    
    if model_name.startswith("gpt"):
        # For GPT-based models, typically pass base64
        image_base64 = encode_image(image_path)
        answer = run_func(question, image_base64)
    else:
        answer = run_func(question, image_path, model_kwargs)
        
    return answer

def load_config(config_path):
    """
    Load configuration from YAML file.
    Raises an error if the file doesn't exist or is improperly formatted.
    """
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file '{config_path}' not found. A valid config file is required.")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Verify required configuration elements
    if "datasets" not in config:
        raise ValueError("Config file missing 'datasets' section")
    if "robospatial_home" not in config["datasets"]:
        raise ValueError("Config file missing 'robospatial_home' section in 'datasets'")
    if "data_dir" not in config["datasets"]["robospatial_home"]:
        raise ValueError("Config file missing 'data_dir' for robospatial_home")
    
    # Set defaults for output configuration
    if "output" not in config or config["output"] is None:
        config["output"] = {}
    
    # If output_dir is specified, use it, otherwise default to 'results' in current directory
    if "output_dir" not in config["output"]:
        config["output"]["output_dir"] = os.path.join(os.getcwd(), "results")
    
    return config

# ------------------------------------------------------------------------
# 3) Main Script
# ------------------------------------------------------------------------
def main():
    """
    Usage:
        python main.py <MODEL_NAME> [MODEL_PATH] --config CONFIG_PATH [--dry-run]
        python main.py --results RESULTS_FILE --config CONFIG_PATH [--dry-run]

    Required arguments:
        - --config CONFIG_PATH: Path to YAML config file with dataset paths
        
    Either specify a model to run:
        - MODEL_NAME: Name of the model to use for evaluation
        - [MODEL_PATH]: Optional path to model weights if not using default
        
    Or specify pre-generated results:
        - --results RESULTS_FILE: Path to pre-generated results file in the same format as the benchmark data

    Optional arguments:
        - --dry-run: Only evaluate the first 3 examples from each JSON file

    Examples:
        python main.py molmo /path/to/my/model --config config.yaml --dry-run
        python main.py --results /path/to/results.json --config config.yaml
    """
    parser = argparse.ArgumentParser(description='RoboSpatial Evaluation Script')
    
    # Create a mutually exclusive group for either model or results
    model_or_results = parser.add_mutually_exclusive_group()
    model_or_results.add_argument('model_name', type=str, nargs='?', help='Name of the model to use')
    model_or_results.add_argument('--results', type=str, help='Path to pre-generated results file')
    
    parser.add_argument('model_path', nargs='?', default=None, help='Optional path to model weights')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--dry-run', action='store_true', help='Only evaluate the first 3 examples')
    
    args = parser.parse_args()
    
    dataset_type = "robospatial_home"  # Default to the only supported dataset
    model_name = args.model_name
    model_path = args.model_path
    config_path = args.config
    results_file = args.results
    dry_run = args.dry_run

    # Check if either model_name or results_file is provided
    if not model_name and not results_file:
        print("Error: Either a model name or --results must be provided.")
        parser.print_help()
        sys.exit(1)

    # 2) Load configuration from YAML file
    try:
        config = load_config(config_path)
        data_dir = config["datasets"]["robospatial_home"]["data_dir"]
        output_dir = config["output"]["output_dir"]
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Check if paths exist
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found")
        sys.exit(1)

    # 3) Where to save output
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Prefix for file names if dry run is enabled
    file_prefix = "dry_run_" if dry_run else ""

    # 4) Find all .json files in the data_dir (ground truth)
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        sys.exit(0)

    # 5) If using pre-generated results, verify the results file exists
    model_kwargs = None
    if results_file:
        if not os.path.exists(results_file):
            print(f"Error: Results file '{results_file}' not found")
            sys.exit(1)
        try:
            with open(results_file, "r") as f:
                results_data = json.load(f)
                if not isinstance(results_data, list):
                    results_data = [results_data]
        except json.JSONDecodeError:
            print(f"Error: Results file '{results_file}' is not a valid JSON file")
            sys.exit(1)
    else:
        # Load the model if not using pre-generated results
        # Import torch-related modules only when actually running a model
        import torch
        # Set random seeds for reproducibility 
        random.seed(56)
        np.random.seed(56)
        torch.manual_seed(56)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(56)
            
        print(f"Loading model '{model_name}' for RoboSpatial-Home evaluation...")
        model_kwargs = load_model(model_name, model_path)
        print("Model loaded successfully.")

    all_stats = []
    overall_category_stats = {}  # For aggregating per-category stats across files

    # 6) Evaluate each JSON file
    for jf in tqdm(json_files, desc=f"Processing RoboSpatial-Home JSONs"):
        file_path = os.path.join(data_dir, jf)
        with open(file_path, "r") as f:
            data = json.load(f)

        # If data is a single dict, make it a list
        if not isinstance(data, list):
            data = [data]

        # If dry run is enabled, only evaluate the first 3 examples
        if dry_run:
            print("Dry run enabled: Evaluating only the first 3 examples from this file.")
            data = data[:3]

        # Evaluate either using pre-generated results or by running the model
        if results_file:
            # Filter results data to only include entries for this file
            file_results_data = [r for r in results_data if r.get("source_file", "") == jf]
            
            # If no source_file field is present, use all results
            # The eval_pregenerated_results function will handle matching
            if not file_results_data:
                file_results_data = results_data
                
            stats = eval_pregenerated_results(data, file_results_data, data_dir)
        else:
            stats = eval_robospatial_home(data, model_name, model_kwargs, data_dir, run_model)

        # Save per-file results with the appropriate prefix
        base_name = os.path.splitext(jf)[0]
        result_name = f"{file_prefix}{base_name}_{'custom' if results_file else model_name}_results.json"
        out_file = os.path.join(output_dir, result_name)
        with open(out_file, "w") as outf:
            json.dump(stats["results"], outf, indent=2)

        # Print file-level statistics
        print(f"\n=== Stats for file '{jf}' ===")
        print(f"  Accuracy: {stats['accuracy']:.2f}% ({stats['num_correct']}/{stats['num_total']})")
        
        # Print per-category accuracy for this file
        print("  Per-category accuracy:")
        for cat, cat_stats in stats["category_stats"].items():
            acc = 100.0 * cat_stats["num_correct"] / cat_stats["num_total"] if cat_stats["num_total"] > 0 else 0.0
            print(f"    {cat}: {acc:.2f}% ({cat_stats['num_correct']}/{cat_stats['num_total']})")
            # Aggregate overall category stats
            if cat not in overall_category_stats:
                overall_category_stats[cat] = {"num_correct": 0, "num_total": 0}
            overall_category_stats[cat]["num_correct"] += cat_stats["num_correct"]
            overall_category_stats[cat]["num_total"] += cat_stats["num_total"]
            
        # Print diagnostic info for pre-generated results
        if results_file and "unmatched_entries" in stats:
            print(f"  Unmatched entries: {stats['unmatched_entries']}")
            
        print(f"  Illformed questions: {stats['illformed_questions']}")
        print(f"  Illformed responses: {stats['illformed_responses']}")
        print(f"  Detailed results saved to: {out_file}\n")

        # Summarize for aggregation
        summary_entry = {
            "file": jf,
            "accuracy": stats["accuracy"],
            "num_correct": stats["num_correct"],
            "num_total": stats["num_total"],
            "illformed_questions": stats["illformed_questions"],
            "illformed_responses": stats["illformed_responses"],
            "category_stats": stats["category_stats"]
        }
        if "unmatched_entries" in stats:
            summary_entry["unmatched_entries"] = stats["unmatched_entries"]

        all_stats.append(summary_entry)

    # 7) Aggregate Stats across all files
    overall_correct = sum(e["num_correct"] for e in all_stats)
    overall_total = sum(e["num_total"] for e in all_stats)
    overall_acc = 100.0 * overall_correct / overall_total if overall_total > 0 else 0

    model_identifier = 'custom' if results_file else model_name
    
    aggregate_summary = {
        "dataset_type": dataset_type,
        "model_name": model_identifier,
        "overall_accuracy": overall_acc,
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "file_summaries": all_stats,
    }
    
    # Add overall per-category statistics
    overall_cat_acc = {}
    for cat, stats in overall_category_stats.items():
        acc = 100.0 * stats["num_correct"] / stats["num_total"] if stats["num_total"] > 0 else 0.0
        overall_cat_acc[cat] = {
            "accuracy": acc,
            "num_correct": stats["num_correct"],
            "num_total": stats["num_total"]
        }
    aggregate_summary["category_stats"] = overall_cat_acc

    # Write the aggregate summary with the appropriate prefix
    agg_file = os.path.join(output_dir, f"{file_prefix}aggregate_robospatial_home_{model_identifier}.json")
    with open(agg_file, "w") as af:
        json.dump(aggregate_summary, af, indent=2)

    # Print final summary
    print(f"\n=== ROBOSPATIAL-HOME Results with model '{model_identifier}' ===")
    print(f"Overall accuracy: {overall_acc:.2f}%  ({overall_correct}/{overall_total})")
    print("Overall per-category accuracy:")
    for cat, cat_stats in overall_category_stats.items():
        acc = 100.0 * cat_stats["num_correct"] / cat_stats["num_total"] if cat_stats["num_total"] > 0 else 0.0
        print(f"  {cat}: {acc:.2f}% ({cat_stats['num_correct']}/{cat_stats['num_total']})")
    print(f"Aggregate summary saved to: {agg_file}")


if __name__ == "__main__":
    main()

