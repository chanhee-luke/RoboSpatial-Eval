# main.py for RoboSpatial Evaluation

import os
import sys
import json
import base64
import random
import numpy as np
from tqdm import tqdm
import argparse
import re
import ast
import io
from datasets import load_dataset, get_dataset_split_names

# Import evaluation functions - these are always needed
from evaluation import (
    eval_robospatial_home,
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
    elif model_name.startswith("qwen25vl"):
        from models import load_qwen25vl_model, run_qwen25vl
        return load_qwen25vl_model, run_qwen25vl
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

def run_model(question, image_input, model_name, model_kwargs):
    """
    Generic helper: runs the specified model on a single (question, image_input).
    image_input can be a path (str) or an image object (e.g., PIL.Image).
    Returns the string answer from the model.
    """
    # Access the correct run function for this model
    _, run_func = import_model_modules(model_name)

    if model_name.startswith("gpt"):
        # For GPT-based models, encode the image to base64
        image_base64 = None
        if isinstance(image_input, str): # Check if it's a path
            if not os.path.exists(image_input):
                print(f"[WARNING] Image path not found: {image_input}")
            else:
                try:
                    image_base64 = encode_image(image_input)
                except Exception as e:
                    print(f"[WARNING] Failed to encode image path {image_input}: {e}")
        else: # Assume it's an image object (e.g., PIL)
            try:
                # Assuming image_input is a PIL Image
                from PIL import Image
                import io
                if isinstance(image_input, Image.Image):
                    buffer = io.BytesIO()
                    # Preserve original format if possible, otherwise use PNG/JPEG
                    img_format = image_input.format if image_input.format else 'PNG'
                    if img_format.upper() == 'WEBP': # Handle cases like WEBP which might not be universally supported
                        print("[INFO] Converting WEBP image to PNG for GPT.")
                        img_format = 'PNG'
                    image_input.save(buffer, format=img_format)
                    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                else:
                    print(f"[WARNING] Unsupported image object type for GPT: {type(image_input)}. Skipping encoding.")
            except Exception as e:
                print(f"[WARNING] Failed to encode image object to base64: {e}")

        if image_base64:
            answer = run_func(question, image_base64)
        else:
            answer = "Error processing image for GPT." # Default error message
    else:
        # For other models, pass the image path or object directly
        # Ensure the specific run_func can handle the image object if it's not a path
        if isinstance(image_input, str) and not os.path.exists(image_input):
             print(f"[WARNING] Image path not found: {image_input}")
             # Let the specific run_func handle the missing path or return an error message
        
        # The specific run_func (e.g., run_llava_next) must be able to handle the image_input type
        answer = run_func(question, image_input, model_kwargs)

    return answer

# ------------------------------------------------------------------------
# 3) Main Script
# ------------------------------------------------------------------------
def main():
    """
    Usage:
        python main.py --model-name <MODEL_NAME> [--model-path MODEL_PATH] \\
                       [--dataset-name DATASET_NAME] [--split SPLIT_NAME | 'all'] \\
                       [--question-col QUESTION_COL] [--answer-col ANSWER_COL] \\
                       [--image-col IMAGE_COL] [--output-dir OUTPUT_DIR] [--dry-run]

    Required arguments:
        - --model-name MODEL_NAME: Name of the model to use for evaluation

    Optional arguments:
        - --model-path MODEL_PATH: Optional path to model weights if not using default
        - --dataset-name DATASET_NAME: Hugging Face dataset name (default: chanhee-luke/RoboSpatial-Home)
        - --split SPLIT_NAME | 'all': Dataset split(s) to evaluate (e.g., 'test', 'train', or 'all' for all available splits) (default: test)
        - --question-col QUESTION_COL: Name of the question column (default: question)
        - --answer-col ANSWER_COL: Name of the ground truth answer column (default: answer)
        - --image-col IMAGE_COL: Name of the image column (default: img)
        - --output-dir OUTPUT_DIR: Directory to save results (default: ./results)
        - --dry-run: Only evaluate the first 3 examples per split

    Examples:
        python main.py --model-name molmo --model-path /path/to/my/model --output-dir /path/to/results --dry-run
        python main.py --model-name llava_next --dataset-name another/dataset --question-col prompt --answer-col gt_answer --image-col image_obj --split all
    """
    parser = argparse.ArgumentParser(description='RoboSpatial Evaluation Script')

    # Model and Data Arguments
    parser.add_argument('--model-name', type=str, required=True, help='Name of the model to use')
    parser.add_argument('--model-path', nargs='?', default=None, help='Optional path to model weights')
    parser.add_argument('--dataset-name', type=str, default='chanhee-luke/RoboSpatial-Home', help='Hugging Face dataset name')
    parser.add_argument('--split', type=str, default='test', help='Dataset split(s) to evaluate (e.g., \'test\', \'train\', or \'all\' for all available splits)')
    parser.add_argument('--question-col', type=str, default='question', help='Name of the question column')
    parser.add_argument('--answer-col', type=str, default='answer', help='Name of the ground truth answer column')
    parser.add_argument('--image-col', type=str, default='img', help='Name of the image column')

    # Output and Control Arguments
    parser.add_argument('--output-dir', type=str, default=os.path.join(os.getcwd(), "results"), help='Directory to save results')
    parser.add_argument('--dry-run', action='store_true', help='Only evaluate the first 3 examples per split')

    args = parser.parse_args()

    model_name = args.model_name
    model_path = args.model_path
    output_dir = args.output_dir
    dry_run = args.dry_run
    requested_split = args.split
    dataset_name = args.dataset_name
    question_col = args.question_col
    answer_col = args.answer_col
    image_col = args.image_col

    # 2) Configuration (Output directory)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Prefix for file names if dry run is enabled
    file_prefix = "dry_run_" if dry_run else ""

    # 5) Load Model
    model_kwargs = None

    # Load the model once
    import torch
    random.seed(56)
    np.random.seed(56)
    torch.manual_seed(56)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(56)

    print(f"Loading model '{model_name}'...")
    model_kwargs = load_model(model_name, model_path)
    print("Model loaded successfully.")

    # 6) Load Dataset Splits
    datasets_to_evaluate = {}
    try:
        if requested_split.lower() == 'all':
            print(f"Loading all available splits for dataset '{dataset_name}'...")
            # Load the entire dataset dictionary
            all_splits_dataset = load_dataset(dataset_name)
            split_names = list(all_splits_dataset.keys())
            print(f"Found splits: {', '.join(split_names)}")
            datasets_to_evaluate = all_splits_dataset # Keep the dict
        else:
            print(f"Loading dataset '{dataset_name}' split '{requested_split}'...")
            dataset = load_dataset(dataset_name, split=requested_split)
            datasets_to_evaluate = {requested_split: dataset} # Store single split in a dict for uniform processing
            print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset '{dataset_name}' from Hugging Face: {e}")
        sys.exit(1)

    # 7) Evaluate each loaded dataset split
    for current_split_name, current_dataset in datasets_to_evaluate.items():
        print(f"--- Evaluating split: {current_split_name} ---")
        overall_category_stats = {} # Reset for each split

        # Convert dataset split to list of dicts
        data_list = [entry for entry in current_dataset]

        if not data_list:
            print(f"Split '{current_split_name}' is empty or failed to load. Skipping.")
            continue

        # If dry run is enabled, only evaluate the first N examples
        if dry_run:
            limit = 3
            print(f"Dry run enabled: Evaluating only the first {limit} examples for split '{current_split_name}'.")
            data_list = data_list[:limit]

        # Run evaluation for the current split
        print(f"Starting evaluation for {len(data_list)} examples from split '{current_split_name}'...")
        stats = eval_robospatial_home(
            data_list,
            model_name,
            model_kwargs,
            run_model,
            question_col=question_col,
            answer_col=answer_col,
            image_col=image_col
        )

        # --- Process and Save Results for the current split ---
        if 'results' not in stats:
            print(f"Error: Evaluation failed to produce results for split '{current_split_name}'.")
            continue # Skip to the next split

        # Save the detailed results for this split
        sanitized_dataset_name = dataset_name.replace('/', '_')
        # Include split name in the filename
        result_name = f"{file_prefix}{sanitized_dataset_name}_{current_split_name}_{model_name}_results.json"
        out_file = os.path.join(output_dir, result_name)
        with open(out_file, "w") as outf:
            json.dump(stats["results"], outf, indent=2)

        # Print overall statistics for this split
        print(f"=== Stats for {dataset_name} ({current_split_name} split) ===")
        print(f"  Accuracy: {stats['accuracy']:.2f}% ({stats['num_correct']}/{stats['num_total']})")

        # Print per-category accuracy for this split
        if "category_stats" in stats and stats["category_stats"]:
            print("  Per-category accuracy:")
            for cat, cat_stats in stats["category_stats"].items():
                acc = 100.0 * cat_stats["num_correct"] / cat_stats["num_total"] if cat_stats["num_total"] > 0 else 0.0
                print(f"    {cat}: {acc:.2f}% ({cat_stats['num_correct']}/{cat_stats['num_total']})")
                # Aggregate overall category stats for this split (used in summary below)
                overall_category_stats[cat] = cat_stats
        else:
            print("  No category statistics available for this split.")

        # Print diagnostic info for this split
        print(f"  Illformed questions (in source data): {stats['illformed_questions']}")
        print(f"  Illformed responses (unparsable model outputs): {stats['illformed_responses']}")
        print(f"  Detailed results for split '{current_split_name}' saved to: {out_file}")

        # --- Aggregate and Save Summary for the current split ---
        overall_correct = stats['num_correct']
        overall_total = stats['num_total']
        overall_acc = stats['accuracy']

        aggregate_summary = {
            "dataset_name": dataset_name,
            "dataset_split": current_split_name, # Use current split name
            "model_name": model_name,
            "overall_accuracy": overall_acc,
            "overall_correct": overall_correct,
            "overall_total": overall_total,
            "category_stats": {}, # Populate below
            "illformed_questions": stats["illformed_questions"],
            "illformed_responses": stats["illformed_responses"],
        }

        # Add overall per-category statistics to summary for this split
        for cat, cat_stats_details in overall_category_stats.items():
            acc = 100.0 * cat_stats_details["num_correct"] / cat_stats_details["num_total"] if cat_stats_details["num_total"] > 0 else 0.0
            aggregate_summary["category_stats"][cat] = {
                "accuracy": acc,
                "num_correct": cat_stats_details["num_correct"],
                "num_total": cat_stats_details["num_total"]
            }

        # Write the aggregate summary for this split
        # Include split name in the filename
        agg_file_name = f"{file_prefix}aggregate_{sanitized_dataset_name}_{current_split_name}_{model_name}.json"
        agg_file = os.path.join(output_dir, agg_file_name)
        with open(agg_file, "w") as af:
            json.dump(aggregate_summary, af, indent=2)

        # Print final summary for this split
        print(f"--- AGGREGATE Results ({dataset_name} / {current_split_name} split) with model '{model_name}' ---")
        print(f"Overall accuracy: {overall_acc:.2f}%  ({overall_correct}/{overall_total})")
        if overall_category_stats:
            print("Overall per-category accuracy:")
            for cat, cat_stats_details in overall_category_stats.items():
                acc = 100.0 * cat_stats_details["num_correct"] / cat_stats_details["num_total"] if cat_stats_details["num_total"] > 0 else 0.0
                print(f"  {cat}: {acc:.2f}% ({cat_stats_details['num_correct']}/{cat_stats_details['num_total']})")
        print(f"Aggregate summary for split '{current_split_name}' saved to: {agg_file}")
        print("-" * (30 + len(current_split_name))) # Separator line

    print("=== Evaluation Complete ===")

if __name__ == "__main__":
    main()

