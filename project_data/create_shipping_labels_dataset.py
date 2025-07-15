#!/usr/bin/env python3
"""
Script to create Donut training dataset for shipping labels.
Creates train and validation metadata.jsonl files from existing image folders.
"""

import os
import json
from data_processor import create_train_val_metadata_from_folders


def create_shipping_labels_dataset():
    """
    Create shipping labels dataset with the correct Donut structure.
    """
    
    # Configuration
    train_folder_path = "/Users/varsha/Desktop/Dori/ocr/shipping_labels2/train1"
    val_folder_path = "/Users/varsha/Desktop/Dori/ocr/shipping_labels2/val1"
    output_dir = "./shipping_labels_dataset"
    
    # Ground truth content for shipping labels (same for all images)
    # Based on the existing metadata content you showed
    ground_truth_content = {
        "sender": "CARDHOLDER SERVICES\nPERSONALIZATION DEPT\n345 KAUTZ RD\nSAINT CHARLES IL 60174",
        "recipient": "Kamlesh Srivastava\n4390 US Highway 1 Ste 312\nPrinceton NJ 08540-5747"
    }
    
    # Task type for shipping labels (document parsing)
    task_type = "document_parsing"
    
    print("🚀 Creating Shipping Labels Dataset for Donut Training")
    print("="*60)
    print(f"📁 Train folder: {train_folder_path}")
    print(f"📁 Val folder: {val_folder_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Task type: {task_type}")
    print(f"📋 Ground truth content: {list(ground_truth_content.keys())}")
    print()
    
    try:
        # Create the dataset
        results = create_train_val_metadata_from_folders(
            train_folder_path=train_folder_path,
            val_folder_path=val_folder_path,
            output_dir=output_dir,
            ground_truth_content=ground_truth_content,
            task_type=task_type,
            image_extensions=['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        )
        
        print("\n✅ Dataset creation completed successfully!")
        print("="*60)
        print("📊 Dataset Structure:")
        print(f"   ├── {output_dir}/")
        print(f"   │   ├── train/")
        print(f"   │   │   ├── metadata.jsonl")
        print(f"   │   │   └── [image files]")
        print(f"   │   └── validation/")
        print(f"   │       ├── metadata.jsonl")
        print(f"   │       └── [image files]")
        print()
        print("📄 Created files:")
        for split, path in results.items():
            print(f"   ├── {split}: {path}")
        
        # Show sample metadata content
        print("\n📋 Sample metadata.jsonl content:")
        for split, path in results.items():
            if os.path.exists(path):
                with open(path, 'r') as f:
                    first_line = f.readline().strip()
                    print(f"   {split}: {first_line[:100]}...")
        
        print("\n🎯 Ready for Donut training!")
        print("You can now use this dataset with:")
        print(f"python train.py --config config/train_shipping_labels.yaml")
        print(f"--dataset_name_or_paths '[\"{output_dir}\"]'")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {str(e)}")
        print("Please check that the train and val folders exist and contain image files.")
    except ValueError as e:
        print(f"❌ Error: {str(e)}")
        print("Please check that the folders contain valid image files.")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")


def create_custom_shipping_labels_dataset(
    train_folder_path: str,
    val_folder_path: str,
    output_dir: str,
    ground_truth_content: dict,
    task_type: str = "document_parsing"
):
    """
    Create a custom shipping labels dataset with your own parameters.
    
    Args:
        train_folder_path (str): Path to train images folder
        val_folder_path (str): Path to validation images folder
        output_dir (str): Output directory for the dataset
        ground_truth_content (dict): Ground truth content for all images
        task_type (str): Task type (document_parsing, document_classification, etc.)
    """
    
    print(f"🚀 Creating Custom Shipping Labels Dataset")
    print("="*50)
    print(f"📁 Train folder: {train_folder_path}")
    print(f"📁 Val folder: {val_folder_path}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Task type: {task_type}")
    print(f"📋 Ground truth keys: {list(ground_truth_content.keys())}")
    
    try:
        results = create_train_val_metadata_from_folders(
            train_folder_path=train_folder_path,
            val_folder_path=val_folder_path,
            output_dir=output_dir,
            ground_truth_content=ground_truth_content,
            task_type=task_type
        )
        
        print(f"\n✅ Custom dataset created successfully at: {output_dir}")
        return results
        
    except Exception as e:
        print(f"❌ Error creating custom dataset: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the default shipping labels dataset creation
    create_shipping_labels_dataset()
    
    # Example of how to create a custom dataset
    print("\n" + "="*60)
    print("💡 Example: Creating a custom dataset")
    print("="*60)
    
    # Example custom ground truth for different shipping label format
    custom_ground_truth = {
        "carrier": "fedex",
        "tracking_number": "123456789012",
        "sender": {
            "name": "John Doe",
            "address": "123 Main St, New York, NY 10001"
        },
        "recipient": {
            "name": "Jane Smith", 
            "address": "456 Oak Ave, Los Angeles, CA 90210"
        },
        "package": {
            "weight": "2.5 lbs",
            "service_type": "Ground"
        }
    }
    
    # Uncomment to create a custom dataset
    # create_custom_shipping_labels_dataset(
    #     train_folder_path="/path/to/your/train/folder",
    #     val_folder_path="/path/to/your/val/folder", 
    #     output_dir="./custom_shipping_dataset",
    #     ground_truth_content=custom_ground_truth,
    #     task_type="document_parsing"
    # ) 