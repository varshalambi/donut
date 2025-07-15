#!/usr/bin/env python3
"""
Example usage of the data processor functions for Donut metadata.jsonl creation.
"""

from data_processor import process_json_to_metadata_jsonl, process_batch_json_files
import json
import os


def create_example_json_files():
    """Create example JSON files for demonstration."""
    
    # Example 1: Document parsing (receipt)
    receipt_data = {
        "file_name": "receipt_001.jpg",
        "ground_truth": {
            "menu": [
                {"name": "Coffee", "count": "2", "price": "3.50"},
                {"name": "Cake", "count": "1", "price": "5.00"}
            ],
            "total": "8.50",
            "tax": "0.85",
            "tip": "1.70"
        }
    }
    
    # Example 2: Document classification
    classification_data = {
        "file_name": "document_001.jpg",
        "ground_truth": {
            "class": "invoice"
        }
    }
    
    # Example 3: Document VQA
    vqa_data = {
        "file_name": "document_002.jpg",
        "ground_truth": {
            "question": "What is the total amount?",
            "answer": "8.50"
        }
    }
    
    # Example 4: Text reading
    text_reading_data = {
        "file_name": "document_003.jpg",
        "ground_truth": "This is a sample document with text content that needs to be extracted."
    }
    
    # Save examples
    examples = [
        ("receipt_example.json", receipt_data),
        ("classification_example.json", classification_data),
        ("vqa_example.json", vqa_data),
        ("text_reading_example.json", text_reading_data)
    ]
    
    for filename, data in examples:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Created {filename}")
    
    return [filename for filename, _ in examples]


def demonstrate_single_file_processing():
    """Demonstrate processing a single JSON file."""
    print("\n" + "="*50)
    print("📄 SINGLE FILE PROCESSING DEMO")
    print("="*50)
    
    # Process document parsing example
    try:
        result_path = process_json_to_metadata_jsonl(
            input_json_path="receipt_example.json",
            output_dir="./dataset",
            task_type="document_parsing",
            split_name="train"
        )
        print(f"✅ Document parsing completed: {result_path}")
        
        # Show the created metadata.jsonl content
        with open(result_path, 'r') as f:
            content = f.read()
            print(f"📋 Generated metadata.jsonl content:\n{content}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    # Process document classification example
    try:
        result_path = process_json_to_metadata_jsonl(
            input_json_path="classification_example.json",
            output_dir="./dataset",
            task_type="document_classification",
            split_name="train"
        )
        print(f"✅ Document classification completed: {result_path}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def demonstrate_batch_processing():
    """Demonstrate processing multiple JSON files."""
    print("\n" + "="*50)
    print("📦 BATCH PROCESSING DEMO")
    print("="*50)
    
    # Create a batch directory with multiple files
    batch_dir = "example_batch"
    os.makedirs(batch_dir, exist_ok=True)
    
    # Create multiple classification examples
    for i in range(5):
        data = {
            "file_name": f"document_{i:03d}.jpg",
            "ground_truth": {
                "class": f"document_type_{i % 3}"  # 3 different classes
            }
        }
        with open(f"{batch_dir}/sample_{i}.json", 'w') as f:
            json.dump(data, f, indent=2)
    
    print(f"✅ Created {batch_dir} directory with 5 sample files")
    
    # Process the batch
    try:
        processed_files = process_batch_json_files(
            input_dir=batch_dir,
            output_dir="./dataset",
            task_type="document_classification",
            split_name="train"
        )
        print(f"✅ Batch processing completed: {len(processed_files)} files processed")
        
        # Show the created metadata.jsonl content
        metadata_path = "./dataset/train/metadata.jsonl"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                lines = f.readlines()
                print(f"📋 Generated metadata.jsonl has {len(lines)} entries")
                print("First entry:")
                print(lines[0].strip())
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def demonstrate_different_task_types():
    """Demonstrate different task types."""
    print("\n" + "="*50)
    print("🎯 DIFFERENT TASK TYPES DEMO")
    print("="*50)
    
    task_examples = [
        ("receipt_example.json", "document_parsing", "Document Parsing"),
        ("classification_example.json", "document_classification", "Document Classification"),
        ("vqa_example.json", "document_vqa", "Document VQA"),
        ("text_reading_example.json", "text_reading", "Text Reading")
    ]
    
    for json_file, task_type, task_name in task_examples:
        if os.path.exists(json_file):
            try:
                result_path = process_json_to_metadata_jsonl(
                    input_json_path=json_file,
                    output_dir=f"./dataset_{task_type}",
                    task_type=task_type,
                    split_name="train"
                )
                print(f"✅ {task_name} completed: {result_path}")
                
            except Exception as e:
                print(f"❌ {task_name} failed: {str(e)}")


def cleanup_example_files():
    """Clean up example files created during demonstration."""
    print("\n" + "="*50)
    print("🧹 CLEANING UP EXAMPLE FILES")
    print("="*50)
    
    files_to_remove = [
        "receipt_example.json",
        "classification_example.json", 
        "vqa_example.json",
        "text_reading_example.json"
    ]
    
    dirs_to_remove = [
        "example_batch",
        "dataset",
        "dataset_document_parsing",
        "dataset_document_classification", 
        "dataset_document_vqa",
        "dataset_text_reading"
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️  Removed {file}")
    
    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            import shutil
            shutil.rmtree(dir_path)
            print(f"🗑️  Removed directory {dir_path}")


if __name__ == "__main__":
    print("🚀 DONUT DATA PROCESSOR DEMONSTRATION")
    print("="*50)
    
    # Create example files
    print("📝 Creating example JSON files...")
    create_example_json_files()
    
    # Demonstrate single file processing
    demonstrate_single_file_processing()
    
    # Demonstrate batch processing
    demonstrate_batch_processing()
    
    # Demonstrate different task types
    demonstrate_different_task_types()
    
    # Clean up
    cleanup_example_files()
    
    print("\n✅ Demonstration completed!")
    print("\n📚 Usage Summary:")
    print("1. Single file: process_json_to_metadata_jsonl(input_json_path, output_dir, task_type)")
    print("2. Batch files: process_batch_json_files(input_dir, output_dir, task_type)")
    print("3. Supported task types: document_parsing, document_classification, document_vqa, text_reading") 