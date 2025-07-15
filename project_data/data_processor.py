import json
import os
from typing import Dict, List, Any, Optional


def process_json_to_metadata_jsonl(
    input_json_path: str, 
    output_dir: str, 
    task_type: str = "document_parsing",
    split_name: str = "train"
) -> str:
    """
    Process a JSON file containing ground truth data and convert it to metadata.jsonl format
    for Donut model training.
    
    This function:
    1. Takes a JSON file with ground_truth key
    2. Extracts and validates the ground_truth data
    3. Converts it into proper JSON object format
    4. Creates a final metadata.jsonl file in the specified output directory
    
    Args:
        input_json_path (str): Path to the input JSON file containing ground truth data
        output_dir (str): Directory where the metadata.jsonl file will be saved
        task_type (str): Type of task - "document_parsing", "document_classification", 
                        "document_vqa", or "text_reading"
        split_name (str): Name of the split (train, validation, test)
    
    Returns:
        str: Path to the created metadata.jsonl file
        
    Raises:
        FileNotFoundError: If input JSON file doesn't exist
        KeyError: If ground_truth key is missing from JSON
        ValueError: If ground_truth data is invalid
        OSError: If output directory cannot be created
    """
    
    # Step 1: Load and validate the input JSON file
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(f"Input JSON file not found: {input_json_path}")
    
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {input_json_path}: {str(e)}")
    
    # Step 2: Extract ground_truth key
    if 'ground_truth' not in input_data:
        raise KeyError(f"Missing 'ground_truth' key in {input_json_path}")
    
    ground_truth_data = input_data['ground_truth']
    
    # Step 3: Convert ground_truth into proper JSON object format
    # Handle different input formats
    if isinstance(ground_truth_data, str):
        try:
            # If it's already a JSON string, parse it
            ground_truth_json = json.loads(ground_truth_data)
        except json.JSONDecodeError:
            # If it's not valid JSON, treat it as raw text for text_reading task
            if task_type == "text_reading":
                ground_truth_json = {"text_sequence": ground_truth_data}
            else:
                raise ValueError(f"Invalid JSON string in ground_truth: {ground_truth_data}")
    elif isinstance(ground_truth_data, dict):
        ground_truth_json = ground_truth_data
    else:
        raise ValueError(f"Unsupported ground_truth data type: {type(ground_truth_data)}")
    
    # Step 4: Validate and format the ground_truth based on task type
    formatted_ground_truth = _format_ground_truth_for_task(ground_truth_json, task_type)
    
    # Step 5: Create output directory structure
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    # Step 6: Create metadata.jsonl file
    metadata_filepath = os.path.join(split_dir, "metadata.jsonl")
    
    # Prepare metadata entry
    # Extract file_name from input data or use a default
    file_name = input_data.get('file_name', 'document.jpg')
    
    # Create the metadata entry in Donut format
    metadata_entry = {
        "file_name": file_name,
        "ground_truth": json.dumps(formatted_ground_truth, ensure_ascii=False)
    }
    
    # Write to metadata.jsonl file
    try:
        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata_entry, f, ensure_ascii=False)
            f.write('\n')
    except OSError as e:
        raise OSError(f"Failed to write metadata.jsonl file: {str(e)}")
    
    print(f"✅ Successfully created metadata.jsonl at: {metadata_filepath}")
    print(f"📄 File name: {file_name}")
    print(f"🎯 Task type: {task_type}")
    print(f"📊 Ground truth format: {list(formatted_ground_truth.keys())}")
    
    return metadata_filepath


def create_train_val_metadata_from_folders(
    train_folder_path: str,
    val_folder_path: str,
    output_dir: str,
    ground_truth_content: Dict[str, Any],
    task_type: str = "document_parsing",
    image_extensions: List[str] = None
) -> Dict[str, str]:
    """
    Create train and val metadata.jsonl files from existing folders with the same content for all images.
    
    This function:
    1. Scans train and val folders for image files
    2. Creates metadata.jsonl files with the same ground_truth content for all images
    3. Sets up the correct Donut dataset structure
    4. Copies images to the proper locations
    
    Args:
        train_folder_path (str): Path to the train folder containing images
        val_folder_path (str): Path to the val folder containing images  
        output_dir (str): Directory where the dataset will be created
        ground_truth_content (Dict[str, Any]): The ground truth content to use for all images
        task_type (str): Type of task for the dataset
        image_extensions (List[str]): List of image file extensions to process (default: ['.jpg', '.jpeg', '.png'])
    
    Returns:
        Dict[str, str]: Paths to created metadata.jsonl files for train and val
        
    Raises:
        FileNotFoundError: If input folders don't exist
        OSError: If output directory cannot be created
    """
    
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    
    # Validate input folders
    if not os.path.exists(train_folder_path):
        raise FileNotFoundError(f"Train folder not found: {train_folder_path}")
    if not os.path.exists(val_folder_path):
        raise FileNotFoundError(f"Val folder not found: {val_folder_path}")
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Format ground truth content
    formatted_ground_truth = _format_ground_truth_for_task(ground_truth_content, task_type)
    ground_truth_string = json.dumps(formatted_ground_truth, ensure_ascii=False)
    
    results = {}
    
    # Process train folder
    print(f"📁 Processing train folder: {train_folder_path}")
    train_images = _get_image_files(train_folder_path, image_extensions)
    if not train_images:
        raise ValueError(f"No image files found in train folder: {train_folder_path}")
    
    train_dir = os.path.join(output_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    
    train_metadata_path = os.path.join(train_dir, "metadata.jsonl")
    _create_metadata_jsonl(train_metadata_path, train_images, ground_truth_string)
    results['train'] = train_metadata_path
    
    # Copy train images
    _copy_images_to_folder(train_images, train_dir)
    
    # Process val folder
    print(f"📁 Processing val folder: {val_folder_path}")
    val_images = _get_image_files(val_folder_path, image_extensions)
    if not val_images:
        raise ValueError(f"No image files found in val folder: {val_folder_path}")
    
    val_dir = os.path.join(output_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)
    
    val_metadata_path = os.path.join(val_dir, "metadata.jsonl")
    _create_metadata_jsonl(val_metadata_path, val_images, ground_truth_string)
    results['validation'] = val_metadata_path
    
    # Copy val images
    _copy_images_to_folder(val_images, val_dir)
    
    print(f"✅ Successfully created dataset structure at: {output_dir}")
    print(f"📊 Train images: {len(train_images)}")
    print(f"📊 Val images: {len(val_images)}")
    print(f"🎯 Task type: {task_type}")
    print(f"📋 Ground truth format: {list(formatted_ground_truth.keys())}")
    
    return results


def _get_image_files(folder_path: str, extensions: List[str]) -> List[str]:
    """Get all image files from a folder."""
    image_files = []
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext.lower()) for ext in extensions):
            image_files.append(filename)
    return sorted(image_files)


def _create_metadata_jsonl(metadata_path: str, image_files: List[str], ground_truth_string: str):
    """Create metadata.jsonl file with entries for all image files."""
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for image_file in image_files:
            metadata_entry = {
                "file_name": image_file,
                "ground_truth": ground_truth_string
            }
            json.dump(metadata_entry, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"📄 Created metadata.jsonl with {len(image_files)} entries: {metadata_path}")


def _copy_images_to_folder(image_files: List[str], target_folder: str, source_folder: str = None):
    """Copy image files to target folder."""
    import shutil
    
    if source_folder is None:
        # If no source folder specified, assume images are already in target folder
        return
    
    for image_file in image_files:
        source_path = os.path.join(source_folder, image_file)
        target_path = os.path.join(target_folder, image_file)
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
    
    print(f"📁 Copied {len(image_files)} images to: {target_folder}")


def _format_ground_truth_for_task(ground_truth: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    """
    Format ground truth data according to the specified task type.
    
    Args:
        ground_truth (Dict[str, Any]): Raw ground truth data
        task_type (str): Type of task to format for
        
    Returns:
        Dict[str, Any]: Formatted ground truth data
    """
    
    if task_type == "document_classification":
        # Format: {"gt_parse": {"class": "class_name"}}
        if "class" not in ground_truth:
            raise ValueError("Document classification requires 'class' field in ground_truth")
        return {"gt_parse": {"class": ground_truth["class"]}}
    
    elif task_type == "document_vqa":
        # Format: {"gt_parses": [{"question": "q", "answer": "a"}, ...]}
        if "question" in ground_truth and "answer" in ground_truth:
            # Single Q&A pair
            return {"gt_parses": [{"question": ground_truth["question"], "answer": ground_truth["answer"]}]}
        elif "questions" in ground_truth and "answers" in ground_truth:
            # Multiple Q&A pairs
            questions = ground_truth["questions"]
            answers = ground_truth["answers"]
            if len(questions) != len(answers):
                raise ValueError("Number of questions and answers must match")
            qa_pairs = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
            return {"gt_parses": qa_pairs}
        else:
            raise ValueError("Document VQA requires 'question'/'answer' or 'questions'/'answers' fields")
    
    elif task_type == "text_reading":
        # Format: {"gt_parse": {"text_sequence": "text content"}}
        if "text_sequence" in ground_truth:
            return {"gt_parse": {"text_sequence": ground_truth["text_sequence"]}}
        else:
            raise ValueError("Text reading requires 'text_sequence' field in ground_truth")
    
    elif task_type == "document_parsing":
        # Format: {"gt_parse": {...}} - any structured JSON
        return {"gt_parse": ground_truth}
    
    else:
        raise ValueError(f"Unsupported task type: {task_type}. Supported types: document_parsing, document_classification, document_vqa, text_reading")


def process_batch_json_files(
    input_dir: str, 
    output_dir: str, 
    task_type: str = "document_parsing",
    split_name: str = "train"
) -> List[str]:
    """
    Process multiple JSON files in a directory and create a combined metadata.jsonl file.
    
    Args:
        input_dir (str): Directory containing JSON files to process
        output_dir (str): Directory where metadata.jsonl will be saved
        task_type (str): Type of task for all files
        split_name (str): Name of the split
        
    Returns:
        List[str]: List of processed file paths
    """
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all JSON files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    if not json_files:
        raise ValueError(f"No JSON files found in {input_dir}")
    
    # Create output directory
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    metadata_filepath = os.path.join(split_dir, "metadata.jsonl")
    processed_files = []
    
    # Process each JSON file and append to metadata.jsonl
    with open(metadata_filepath, 'w', encoding='utf-8') as f:
        for json_file in json_files:
            input_path = os.path.join(input_dir, json_file)
            
            try:
                # Load the JSON file
                with open(input_path, 'r', encoding='utf-8') as json_f:
                    input_data = json.load(json_f)
                
                # Extract and format ground truth
                if 'ground_truth' not in input_data:
                    print(f"⚠️  Skipping {json_file}: missing ground_truth key")
                    continue
                
                ground_truth_data = input_data['ground_truth']
                
                # Convert to proper format
                if isinstance(ground_truth_data, str):
                    try:
                        ground_truth_json = json.loads(ground_truth_data)
                    except json.JSONDecodeError:
                        if task_type == "text_reading":
                            ground_truth_json = {"text_sequence": ground_truth_data}
                        else:
                            print(f"⚠️  Skipping {json_file}: invalid JSON in ground_truth")
                            continue
                elif isinstance(ground_truth_data, dict):
                    ground_truth_json = ground_truth_data
                else:
                    print(f"⚠️  Skipping {json_file}: unsupported ground_truth type")
                    continue
                
                # Format for task
                formatted_ground_truth = _format_ground_truth_for_task(ground_truth_json, task_type)
                
                # Create metadata entry
                file_name = input_data.get('file_name', f"{os.path.splitext(json_file)[0]}.jpg")
                metadata_entry = {
                    "file_name": file_name,
                    "ground_truth": json.dumps(formatted_ground_truth, ensure_ascii=False)
                }
                
                # Write to metadata.jsonl
                json.dump(metadata_entry, f, ensure_ascii=False)
                f.write('\n')
                
                processed_files.append(input_path)
                
            except Exception as e:
                print(f"❌ Error processing {json_file}: {str(e)}")
                continue
    
    print(f"✅ Successfully processed {len(processed_files)} files")
    print(f"📄 Created metadata.jsonl at: {metadata_filepath}")
    
    return processed_files


# Example usage and testing
if __name__ == "__main__":
    # Example 1: Process a single JSON file
    example_input = {
        "file_name": "receipt_001.jpg",
        "ground_truth": {
            "menu": [
                {"name": "Coffee", "count": "2", "price": "3.50"},
                {"name": "Cake", "count": "1", "price": "5.00"}
            ],
            "total": "8.50"
        }
    }
    
    # Save example input
    with open("example_input.json", "w") as f:
        json.dump(example_input, f, indent=2)
    
    # Process the example
    try:
        result_path = process_json_to_metadata_jsonl(
            input_json_path="example_input.json",
            output_dir="./dataset",
            task_type="document_parsing",
            split_name="train"
        )
        print(f"✅ Single file processing completed: {result_path}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Example 2: Process multiple files
    # Create example batch
    os.makedirs("example_batch", exist_ok=True)
    for i in range(3):
        example_data = {
            "file_name": f"document_{i:03d}.jpg",
            "ground_truth": {
                "class": f"document_type_{i % 3}"
            }
        }
        with open(f"example_batch/sample_{i}.json", "w") as f:
            json.dump(example_data, f, indent=2)
    
    try:
        processed_files = process_batch_json_files(
            input_dir="example_batch",
            output_dir="./dataset",
            task_type="document_classification",
            split_name="train"
        )
        print(f"✅ Batch processing completed: {len(processed_files)} files")
    except Exception as e:
        print(f"❌ Error: {str(e)}") 