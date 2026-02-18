# RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics

[**🌐 Homepage**](https://chanh.ee/RoboSpatial/) | [**📖 arXiv**](https://arxiv.org/abs/2411.16537) | [**🛠️ Data Gen**](https://github.com/NVlabs/RoboSpatial) | [**📂 Benchmark**](https://huggingface.co/datasets/chanhee-luke/RoboSpatial-Home)

**✨ CVPR 2025 (Oral) ✨**

Authors: [Chan Hee Song](https://chanh.ee)<sup>1</sup>, [Valts Blukis](https://research.nvidia.com/person/valts-blukis)<sup>2</sup>, [Jonathan Tremblay](https://research.nvidia.com/person/jonathan-tremblay)<sup>2</sup>, [Stephen Tyree](https://research.nvidia.com/person/stephen-tyree)<sup>2</sup>, [Yu Su](https://ysu1989.github.io/)<sup>1</sup>, [Stan Birchfield](https://sbirchfield.github.io/)<sup>2</sup>

 <sup>1</sup> The Ohio State University  <sup>2</sup> NVIDIA

### CVPR 2026 ERA Workshop & Challenge

**RoboSpatial-Home is one of two challenges in the [Embodied Reasoning in Action (ERA) Workshop and Challenge](https://embodied-reasoning.github.io) on Embodied Reasoning for Robotics (CVPR 2026): the RoboSpatial challenge (this repo) and the PointArena challenge.**

- **Challenge instructions (participation, submission, rules):** See the **[RoboSpatial Challenge Instruction](https://docs.google.com/document/d/1Y1wivo8B_8OpHgP9x99IUsdpIyOMENNRtLmzM7ByZfU/edit?usp=sharing)** document for the full challenge, including how to submit your model.
- **Adding a model to this codebase (local evaluation):** See **[ADDING_MODELS.md](ADDING_MODELS.md)** for step-by-step instructions on how to add and run your model in this repo.

## Leaderboard

Benchmark your model against others using the evaluation in this repo. **Configuration** and **Compatibility** are VQA (visual question answering) categories; **Context** is pointing. Scores include a **VQA average** (average of Configuration and Compatibility) and an **overall total** (average of all three categories).

### Existing works

Baselines from published work (not part of the ERA challenge).

#### API-based models (Gemini, GPT)

| Model | Configuration (VQA) | Compatibility (VQA) | **VQA avg** | Context (pointing) | **Total** |
|-------|---------------------|----------------------|--------------|---------------------|-----------|
| **GPT-4o** | 74.0 | 55.2 | 64.6 | 6.6 | 45.3 |
| **Gemini 1.5 ER** | — | — | 31.1 | 79.3 | 47.2 |
| **Gemini 2.5 Pro** | — | — | 71.3 | 8.3 | 50.3 |
| **GPT-5-mini** | — | — | 70.7 | 12.5 | 51.3 |
| **Gemini 2.5 Flash** | — | — | 73.4 | 7.9 | 51.6 |
| **GPT-5** | — | — | 69.3 | 19.0 | 52.5 |

#### Open-weight models

| Model | Configuration (VQA) | Compatibility (VQA) | **VQA avg** | Context (pointing) | **Total** |
|-------|---------------------|----------------------|--------------|---------------------|-----------|
| **RoboPoint-13B** | 69.9 | 70.5 | 70.2 | 19.7| 47.3 |
| **Qwen3-VL 2B Instruct** | — | — | — | — | 49.1 |
| **RoboBrain2.0-7B** | — | — | 59.64 | 44.35 | 54.5 |
| **RoboRefer-8B-SFT** | — | — | 58.33 | 61.48 | 59.4 |
| **SpaceTools-3B** | — | — | 79.38 | 52.46 | 70.4 |
| **RoboBrain2.5-8B** | — | — | - | - | 73.0 |
| **Qwen3-VL 235B-A22B Thinking** | — | — | — | — | 73.9 |


### Challenge

ERA workshop participants: submit your results to appear here.

| Rank | Model | Configuration (VQA) | Compatibility (VQA) | **VQA avg** | Context (pointing) | **Total** |
|------|-------|---------------------|----------------------|--------------|---------------------|-----------|
| *—* | *Submit your results!* | — | — | — | — | — |

*For challenge participation and submission, see the [RoboSpatial Challenge Instruction](https://docs.google.com/document/d/1Y1wivo8B_8OpHgP9x99IUsdpIyOMENNRtLmzM7ByZfU/edit?usp=sharing). To add your model to this codebase for local evaluation, see [ADDING_MODELS.md](ADDING_MODELS.md).*

## Introduction

This repository provides evaluation tools for **RoboSpatial-Home**, a spatial reasoning benchmark designed for robotics, augmented reality (AR), and related applications. RoboSpatial-Home is the **RoboSpatial challenge** in the CVPR 2026 ERA Workshop; the other is the **PointArena challenge**. Full details for both are in the [challenge instructions](https://docs.google.com/document/d/1Y1wivo8B_8OpHgP9x99IUsdpIyOMENNRtLmzM7ByZfU/edit?usp=sharing).

If you are looking for the data generation code, please check out our repository at [here](https://github.com/NVlabs/RoboSpatial).



## Evaluation Guidelines

We provide detailed instructions for evaluating your model on RoboSpatial-Home.
You can either run a model through our interface or evaluate pre-generated results. See the Usage section below for both workflows.

### Requirements

```
pip install numpy tqdm pyyaml
```

### Download & Preprocess Dataset
You’ll need to download the dataset before running the evaluation.
We provide a script to make this easy, especially for debugging or if you’re not using the Hugging Face `datasets` library.
```
python download_benchmark.py [OUTPUT_FOLDER_PATH]
```

### Set Dataset Path
Edit config.yaml to point to your local dataset directory and desired output folder:
```
# Dataset paths
datasets:
  robospatial_home:
    data_dir: "/path/to/robospatial-home"  # Root directory containing JSON files and images/ folder

# Output configuration
output:
  output_dir: "./results"  # Full path to where results will be stored
  # If not specified, a 'results' folder will be created in the current directory
```

If `OUTPUT_FOLDER_PATH` is provided, the dataset will be downloaded and saved there. 
If not provided, the dataset will be saved in a folder named `RoboSpatial-Home` in the current directory.



## Usage

There are two ways to use this tool:

### 1. Run and evaluate a model

```
python main.py <MODEL_NAME> [MODEL_PATH] --config CONFIG_PATH [--dry-run]
```

### 2. Evaluate pre-generated results

```
python main.py --results RESULTS_FILE --config CONFIG_PATH [--dry-run]
```

### Required arguments:
- `--config CONFIG_PATH`: Path to YAML config file with dataset paths

### For running a model:
- `MODEL_NAME`: Name of the model to use (See Supported Models for valid options.)
- `MODEL_PATH` (optional): Path to model weights if not using default

### For evaluating pre-generated results:
- `--results RESULTS_FILE`: Path to a JSON file containing pre-generated model responses

### Optional arguments:
- `--dry-run`: Only evaluate the first 3 examples from each JSON file

### Example Commands:

```bash
# Run LLaVA-Next with default model weights
python main.py llava_next --config config.yaml

# Run RoboPoint with a custom model checkpoint
python main.py robopoint /path/to/my/model --config config.yaml

# Run SpatialVLM in dry-run mode (only 3 samples)
python main.py spatialvlm --config config.yaml --dry-run

# Evaluate pre-generated results from JSON
python main.py --results /path/to/results.json --config config.yaml
```

### Pre-generated Results Format

The pre-generated results file should be a JSON file containing a list of QAs with the same structure as the RoboSpatial-Home dataset:

```json
[
  {
    "img": "images/img_context_0.png", 
    "category": "context",
    "question": "In the image, there is a bowl. Pinpoint several points within the vacant space situated to the left of the bowl...",
    "answer": "[(0.383, 0.873), (0.390, 0.990), ...]"
  },
  ...
]
```

🚨 **Important**: The evaluation script matches each entry in your results file to the ground truth using a combination of the `question` and `img` fields. These two fields must exactly match the corresponding example in the dataset.

⚠️ **Note on point formatting**: The evaluation code attempts to handle common variations in point representations, including cases where model responses contain a mix of text and coordinates. If the answer field includes additional text, the code uses regular expressions to extract only the coordinate points. That said, for best results, format your predictions as clean, Python-style tuples—just like in the benchmark annotations.

### Output

Results are saved in the following locations:
- Prediction results: `<output_dir>/<annotation_file_name>_<model_name>_results.json`
- Evaluation summary: `<output_dir>/aggregate_robospatial_home_<model_name>.json`

For pre-generated results evaluation, `<model_name>` is replaced with `custom` in the output filenames.

For dry runs, all output files are prefixed with `dry_run_`.

### Supported Models

To add a new model to this codebase (for local evaluation), see **[ADDING_MODELS.md](ADDING_MODELS.md)** for step-by-step instructions. For ERA challenge participation and submission, see the [RoboSpatial Challenge Instruction](https://docs.google.com/document/d/1Y1wivo8B_8OpHgP9x99IUsdpIyOMENNRtLmzM7ByZfU/edit?usp=sharing).

🚨 **Important**: For all models, please create a separate Python environment using each model's instructions.
During our evaluation, we switch between Python environments when evaluating different models.
The inference code for all models is containerized to allow for isolated evaluation.

#### LLaVA-Next

- `lmms-lab/llama3-llava-next-8b`

#### SpatialVLM
- `remyxai/SpaceMantis`

#### RoboPoint
- `wentao-yuan/robopoint-v1-vicuna-v1.5-13b`

#### Qwen2-VL
- `Qwen/Qwen2-VL-7B-Instruct`

#### Molmo
- `allenai/Molmo-7B-D-0924`

#### GPT-4o
- `export OPENAI_API_KEY=<Your API Key>`

## Contact
- Luke Song: chanhee.luke@gmail.com

## Citation

**BibTex:**
```bibtex
@inproceedings{song2025robospatial,
  author    = {Song, Chan Hee and Blukis, Valts and Tremblay, Jonathan and Tyree, Stephen and Su, Yu and Birchfield, Stan},
  title     = {{RoboSpatial}: Teaching Spatial Understanding to {2D} and {3D} Vision-Language Models for Robotics},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  note      = {Oral Presentation},
}
```
