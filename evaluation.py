"""
Evaluation logic for RoboSpatial-Home benchmark.
Contains functions for evaluating model responses against ground truth data.
"""

import os
import re
import ast
from tqdm import tqdm


def point_in_polygon(x, y, poly):
    """
    Check if the point (x, y) lies within the polygon defined by a list of (x, y) tuples.
    Uses the ray-casting algorithm.
    """
    num = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, num + 1):
        p2x, p2y = poly[i % num]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xinters = p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def evaluate_answer(ground_truth, generated_answer):
    """
    Evaluates if the generated answer is correct based on the ground truth.
    Returns a tuple of (is_correct, is_binary_answer, parsed_answer, is_parsable).
    """
    gen_answer = generated_answer.strip().lower()
    gt_lower = ground_truth.strip().lower()
    
    # Check if this is a binary yes/no question
    if gt_lower in ["yes", "no"]:
        is_binary = True
        is_gt_yes = (gt_lower == "yes")
        # Binary answers are always considered parsable if they contain text
        is_parsable = len(gen_answer) > 0
        if is_gt_yes:
            correct = gen_answer.startswith("yes")
        else:
            correct = gen_answer.startswith("no")
        return correct, is_binary, gen_answer, is_parsable
    else:
        # Numeric evaluation: ground_truth is a list of points defining a polygon
        is_binary = False
        parsed_answer = None
        is_parsable = False  # Default to not parsable until we successfully parse
        
        try:
            gt_polygon = ast.literal_eval(ground_truth)
            if not isinstance(gt_polygon, list) or len(gt_polygon) < 3:
                return False, is_binary, parsed_answer, is_parsable
            
            # Extract the first coordinate pair using regex
            # Look for patterns like (0.1,0.2) or (0.1, 0.2) or [0.1, 0.2] or [0.1,0.2]
            # This approach is more robust than trying to parse the entire list
            
            # Try to match tuple format (x,y) or (x, y)
            tuple_match = re.search(r'\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)', generated_answer)
            if tuple_match:
                try:
                    x = float(tuple_match.group(1))
                    y = float(tuple_match.group(2))
                    parsed_answer = (x, y)
                    is_parsable = True
                    correct = point_in_polygon(x, y, gt_polygon)
                    return correct, is_binary, parsed_answer, is_parsable
                except (ValueError, TypeError):
                    pass  # Continue to other formats if float conversion fails
            
            # Try to match list format [x,y] or [x, y]
            list_match = re.search(r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]', generated_answer)
            if list_match:
                try:
                    x = float(list_match.group(1))
                    y = float(list_match.group(2))
                    parsed_answer = (x, y)
                    is_parsable = True
                    correct = point_in_polygon(x, y, gt_polygon)
                    return correct, is_binary, parsed_answer, is_parsable
                except (ValueError, TypeError):
                    pass  # Continue to other formats if float conversion fails
            
            # Fall back to the original approach but with extra safety
            try:
                # Extract the first list (text between square brackets) from generated_answer
                # Use a regex that can handle multi-line content
                match = re.search(r'\[(.*?)\]', generated_answer, re.DOTALL)
                if match is None:
                    return False, is_binary, parsed_answer, is_parsable
                
                # Add spaces after commas if not present (to help ast.literal_eval)
                list_content = match.group(1)
                list_content = re.sub(r',(\S)', r', \1', list_content)
                
                # Try to fix truncated tuples by adding closing parenthesis and brackets if needed
                list_content = list_content.strip()
                if list_content.endswith(','):
                    list_content = list_content[:-1]
                
                list_str = '[' + list_content + ']'
                
                # Try to parse the list directly
                try:
                    gen_val = ast.literal_eval(list_str)
                except (SyntaxError, ValueError):
                    # If direct parsing fails, try to extract just the first tuple
                    tuple_match = re.search(r'\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)', list_content)
                    if tuple_match:
                        x = float(tuple_match.group(1))
                        y = float(tuple_match.group(2))
                        parsed_answer = (x, y)
                        is_parsable = True
                        correct = point_in_polygon(x, y, gt_polygon)
                        return correct, is_binary, parsed_answer, is_parsable
                    else:
                        return False, is_binary, parsed_answer, is_parsable
                
                # Handle different formats for points
                if isinstance(gen_val, list):
                    if len(gen_val) == 0:
                        return False, is_binary, parsed_answer, is_parsable
                        
                    # Case 1: The list itself is a point coordinates [x, y]
                    if len(gen_val) == 2 and all(isinstance(v, (int, float)) for v in gen_val):
                        gen_point = tuple(gen_val)  # Convert [x, y] to (x, y)
                    # Case 2: The list contains points [(x, y), ...]
                    elif isinstance(gen_val[0], tuple):
                        gen_point = gen_val[0]
                    # Case 3: The list contains coordinate pairs as lists [[x, y], ...]
                    elif isinstance(gen_val[0], list) and len(gen_val[0]) == 2:
                        gen_point = tuple(gen_val[0])  # Convert [x, y] to (x, y)
                    else:
                        return False, is_binary, parsed_answer, is_parsable
                elif isinstance(gen_val, tuple):
                    gen_point = gen_val
                else:
                    return False, is_binary, parsed_answer, is_parsable

                if not (isinstance(gen_point, tuple) and len(gen_point) == 2):
                    return False, is_binary, parsed_answer, is_parsable
                
                x, y = float(gen_point[0]), float(gen_point[1])
                parsed_answer = (x, y)
                is_parsable = True
                correct = point_in_polygon(x, y, gt_polygon)
                return correct, is_binary, parsed_answer, is_parsable
            except Exception:
                # If all parsing attempts fail, return False
                return False, is_binary, parsed_answer, is_parsable
                
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return False, is_binary, parsed_answer, is_parsable

def eval_robospatial_home(
        dataset_list,
        model_name,
        model_kwargs,
        run_model_fn,
        question_col="question",
        answer_col="answer",
        image_col="img"
    ):
    """
    Evaluate a dataset by running the model on each example.
    Assumes data comes from Hugging Face dataset format.

    Args:
        dataset_list: List of data entries (dictionaries) to evaluate.
        model_name: Name of the model being evaluated.
        model_kwargs: Model-specific arguments (tokenizer, model object, etc.).
        run_model_fn: Function to run the model on a single example.
                      Expects (question, image_input, model_name, model_kwargs).
        question_col: Name of the column containing the question/prompt.
        answer_col: Name of the column containing the ground truth answer.
        image_col: Name of the column containing the image object.

    Returns:
        Dictionary containing evaluation statistics and results.
    """
    results = []
    num_correct = 0
    num_total = len(dataset_list)
    illformed_questions = 0
    illformed_responses = 0

    # Dictionary to keep per-category statistics
    # Initialize unconditionally, will be populated if categories exist or are inferred
    category_stats = {}

    # Process one by one (original logic)
    for entry in tqdm(dataset_list, desc=f"Evaluating {model_name} (HF Single)"):
        # Extract question, ground-truth answer, and image using specified column names
        question = entry.get(question_col, "")
        ground_truth = entry.get(answer_col, "")
        image_input = entry.get(image_col)

        if not question or not ground_truth or image_input is None:
            illformed_questions += 1
            # Try to get an ID if available, otherwise use index or hash
            entry_id = entry.get('id') or entry.get('index') or hash(str(entry))
            print(f"[Warning] Skipping entry {entry_id} due to missing required columns ('{question_col}', '{answer_col}', or '{image_col}')")
            continue

        # --- Category Determination ---
        category = entry.get("category") # Check if category is provided
        if category is None: # Infer category if not provided
            gt_lower = ground_truth.strip().lower()
            question_lower = question.strip().lower()
            if gt_lower in ["yes", "no"]:
                if "fit" in question_lower:
                    category = "compatibility"
                else:
                    # Assume other binary questions are configuration based on user description
                    category = "configuration"
            else:
                # Assume non-binary answers correspond to context questions (coordinates)
                category = "context"
        # --- End Category Determination ---


        # Update category stats if category is determined
        if category:
            if category not in category_stats:
                category_stats[category] = {"num_correct": 0, "num_total": 0}
            # Update category total for valid entries only
            category_stats[category]["num_total"] += 1

        # Run the model with the image object (using the single-item function)
        generated_answer = run_model_fn(question, image_input, model_name, model_kwargs)

        # Evaluate the answer
        correct, is_binary, parsed_answer, is_parsable = evaluate_answer(ground_truth, generated_answer)

        # Count illformed responses
        if not is_parsable:
            illformed_responses += 1

        if correct:
            num_correct += 1
            # Update category correct count if category exists and answer is correct
            if category and category in category_stats:
                category_stats[category]["num_correct"] += 1

        result_entry = {
            "question": question,
            "expected_answer": ground_truth,
            "generated_answer": generated_answer,
            "parsed_answer": str(parsed_answer) if parsed_answer is not None else None,
            "correct": correct,
            "is_parsable": is_parsable,
            # Include image filename if available
            "image_filename": getattr(image_input, 'filename', None)
        }
        # Add category to result entry if it was determined
        if category:
             result_entry["category"] = category
        results.append(result_entry)

    # Calculate final accuracy (using original total, adjusted for illformed questions)
    actual_total_evaluated = num_total - illformed_questions
    accuracy = 100.0 * num_correct / actual_total_evaluated if actual_total_evaluated > 0 else 0.0

    return_dict = {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": actual_total_evaluated, # Use the actual evaluated count as num_total
        "illformed_questions": illformed_questions,
        "illformed_responses": illformed_responses,
        "results": results
    }
    # Include category_stats in the return dictionary if it's populated
    if category_stats:
        return_dict["category_stats"] = category_stats

    return return_dict

