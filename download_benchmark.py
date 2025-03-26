"""
Download the RoboSpatial-Home dataset from Hugging Face

Usage: python download_benchmark.py [OUTPUT_FOLDER_PATH]

If OUTPUT_FOLDER_PATH is provided, the dataset will be downloaded and saved there. 
If not provided, the dataset will be saved in a folder named "RoboSpatial-Home" in the current directory.
"""

import os
import sys
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import time

def save_pil_image(image_data, dest_path):
    """
    Save a PIL image object (or None) to the specified destination path.
    If image_data is None, does nothing and returns None.
    """
    if image_data is None:
        return None
    # 'image_data' should be a PIL image if the dataset is cast to Image type
    image_data.save(dest_path)
    return dest_path

def main():
    # The name of the dataset on Hugging Face
    dataset_name = "chanhee-luke/RoboSpatial-Home"
    
    # If a command line argument is provided, use that as the download path
    # otherwise default to "RoboSpatial-Home" in the current directory.
    if len(sys.argv) > 1:
        local_folder = sys.argv[1]
    else:
        local_folder = "RoboSpatial-Home"
    
    print(f"Starting download of {dataset_name} dataset...")
    start_time = time.time()
    
    # Load the dataset dictionary from Hugging Face with progress feedback
    print(f"Downloading dataset from Hugging Face (this may take a while)...")
    ds_dict = load_dataset(dataset_name)
    print(f"Dataset downloaded successfully in {time.time() - start_time:.1f} seconds")

    # Create local folders
    os.makedirs(local_folder, exist_ok=True)
    images_folder = os.path.join(local_folder, "images")
    os.makedirs(images_folder, exist_ok=True)
    print(f"Created output directories at: {local_folder}")

    # We'll gather all annotations into one JSON list
    all_annotations = []
    total_samples = sum(len(ds) for ds in ds_dict.values())
    processed_samples = 0
    saved_images = 0

    # Go through each split (context, compatibility, configuration)
    for split_name, ds in ds_dict.items():
        print(f"\nProcessing split: {split_name} ({len(ds)} samples)")
        
        # Use tqdm for a progress bar
        for idx, sample in tqdm(enumerate(ds), desc=f"Processing {split_name}", total=len(ds), unit="sample"):
            # sample is a dict with keys, e.g.:
            #   category, question, answer, img, depth_image, mask
            # which are PIL images (if already cast to Image in the original dataset),
            # or dictionaries with {'path': ..., 'bytes': ...} if not cast.

            category = sample.get("category", "")
            question = sample.get("question", "")
            answer = sample.get("answer", "")

            # 1) RGB image
            img_field = sample.get("img", None)
            if img_field is not None:
                # Try to get a file name from the 'path' key if it's a dict.
                if isinstance(img_field, dict) and "path" in img_field:
                    original_filename = os.path.basename(img_field["path"])
                    pil_img = img_field.get("pil", Image.open(img_field["path"]))
                elif isinstance(img_field, Image.Image):
                    original_filename = f"img_{split_name}_{idx}.png"
                    pil_img = img_field
                else:
                    # Fallback for other structures
                    original_filename = f"img_{split_name}_{idx}.png"
                    pil_img = img_field
                local_img_path = os.path.join(images_folder, original_filename)
                save_pil_image(pil_img, local_img_path)
                img_path_for_json = os.path.join("images", original_filename)
                saved_images += 1
            else:
                img_path_for_json = None

            # 2) Depth image
            depth_field = sample.get("depth_image", None)
            if depth_field is not None:
                if isinstance(depth_field, dict) and "path" in depth_field:
                    original_depth_filename = os.path.basename(depth_field["path"])
                    pil_depth = depth_field.get("pil", Image.open(depth_field["path"]))
                elif isinstance(depth_field, Image.Image):
                    original_depth_filename = f"depth_{split_name}_{idx}.png"
                    pil_depth = depth_field
                else:
                    original_depth_filename = f"depth_{split_name}_{idx}.png"
                    pil_depth = depth_field
                local_depth_path = os.path.join(images_folder, original_depth_filename)
                save_pil_image(pil_depth, local_depth_path)
                depth_path_for_json = os.path.join("images", original_depth_filename)
                saved_images += 1
            else:
                depth_path_for_json = None

            # 3) Mask (if present)
            mask_field = sample.get("mask", None)
            if mask_field is not None:
                if isinstance(mask_field, dict) and "path" in mask_field:
                    original_mask_filename = os.path.basename(mask_field["path"])
                    pil_mask = mask_field.get("pil", Image.open(mask_field["path"]))
                elif isinstance(mask_field, Image.Image):
                    original_mask_filename = f"mask_{split_name}_{idx}.png"
                    pil_mask = mask_field
                else:
                    original_mask_filename = f"mask_{split_name}_{idx}.png"
                    pil_mask = mask_field
                local_mask_path = os.path.join(images_folder, original_mask_filename)
                save_pil_image(pil_mask, local_mask_path)
                mask_path_for_json = os.path.join("images", original_mask_filename)
                saved_images += 1
            else:
                mask_path_for_json = None

            # Build the annotation record
            record = {
                "category": category,
                "question": question,
                "answer": answer,
                "img": img_path_for_json,
                "depth_image": depth_path_for_json,
                "mask": mask_path_for_json,
            }
            all_annotations.append(record)
            processed_samples += 1

    # Save all annotations to a single JSON file
    print(f"\nSaving annotations to JSON file...")
    annotations_path = os.path.join(local_folder, "annotations.json")
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump(all_annotations, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nDownload and processing complete!")
    print(f"Saved dataset to folder: {local_folder}")
    print(f"Number of samples in JSON: {len(all_annotations)}")
    print(f"Total images saved: {saved_images}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

if __name__ == "__main__":
    main()
