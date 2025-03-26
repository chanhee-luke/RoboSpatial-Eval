# RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics

[**🌐 Homepage**](https://chanh.ee/RoboSpatial/) | [**📖 arXiv**](https://arxiv.org/abs/2411.16537) | [**📂 RoboSpatial-Home**](https://huggingface.co/datasets/chanhee-luke/RoboSpatial-Home)

Authors: [Chan Hee Song](https://chanh.ee), [Valts Blukis](https://research.nvidia.com/person/valts-blukis), [Jonathan Tremblay](https://research.nvidia.com/person/jonathan-tremblay), [Stephen Tyree](https://research.nvidia.com/person/stephen-tyree), [Yu Su](https://ysu1989.github.io/), [Stan Birchfield](https://research.nvidia.com/person/stan-birchfield)

## Introduction

This repository provides evaluation tools for RoboSpatial-Home, a spatial reasoning benchmark designed for robotics, augmented reality (AR), and related applications.


## Requirements

```
pip install numpy tqdm pyyaml
```

## Evaluation Guidelines

We provide detailed instructions for evaluating your model on RoboSpatial-Home.
You can either run a model through our interface or evaluate pre-generated results. See the examples below for both workflows.

### Download & Preprocess Dataset
You’ll need to download the dataset before running the evaluation.
We provide a script to make this easy, especially for debugging or if you’re not using the Hugging Face `datasets`.
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

### Run and evaluate a model

```
python main.py <MODEL_NAME> [MODEL_PATH] --config CONFIG_PATH [--dry-run]
```

### Evaluate pre-generated results

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

🚨 Important: The evaluation script matches each entry in your results file to the ground truth using a combination of the `question` and `img` fields. These two fields must exactly match the corresponding example in the dataset.

### Output

Results are saved in the following locations:
- Prediction results: `<output_dir>/<annotation_file_name>_<model_name>_results.json`
- Evaluation summary: `<output_dir>/aggregate_robospatial_home_<model_name>.json`

For pre-generated results evaluation, `<model_name>` is replaced with `custom` in the output filenames.

For dry runs, all output files are prefixed with `dry_run_`.

## Supported Models

For all models, please create a separate Python environment using each model's instructions.
During our testing, we switch between  Python environment when evaluating different models.
The inference code for all models is containerized to allow for isolated evaluation.

### LLaVA-Next

- `lmms-lab/llama3-llava-next-8b`

### SpatialVLM
- `remyxai/SpaceMantis`

### RoboPoint
- `wentao-yuan/robopoint-v1-vicuna-v1.5-13b`

### Qwen2-VL
- `Qwen/Qwen2-VL-7B-Instruct`

### Molmo
- `allenai/Molmo-7B-D-0924`

### GPT-4o
- `export OPENAI_API_KEY=<Your API Key>`

## Contact
- Luke Song: song.1855@osu.edu

## Citation

**BibTex:**
```bibtex
@inproceedings{song2025robospatial,
  author    = {Song, Chan Hee and Blukis, Valts and Tremblay, Jonathan and Tyree, Stephen and Su, Yu and Birchfield, Stan},
  title     = {{RoboSpatial}: Teaching Spatial Understanding to {2D} and {3D} Vision-Language Models for Robotics},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
  note      = {To appear},
}
```
