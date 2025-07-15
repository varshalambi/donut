#!/usr/bin/env python3
"""
Simple script to set up shipping labels dataset and start training.
"""

import os
import subprocess
import sys
from create_shipping_labels_dataset import create_shipping_labels_dataset


def main():
    """Main function to set up dataset and start training."""
    
    print("🚀 Shipping Labels Dataset Setup and Training")
    print("="*60)
    
    # Step 1: Create the dataset
    print("📁 Step 1: Creating dataset structure...")
    try:
        create_shipping_labels_dataset()
        print("✅ Dataset creation completed!")
    except Exception as e:
        print(f"❌ Error creating dataset: {str(e)}")
        return
    
    # Step 2: Check if dataset was created successfully
    dataset_path = "./shipping_labels_dataset"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        return
    
    # Check for required files
    train_metadata = os.path.join(dataset_path, "train", "metadata.jsonl")
    val_metadata = os.path.join(dataset_path, "validation", "metadata.jsonl")
    
    if not os.path.exists(train_metadata):
        print(f"❌ Train metadata not found: {train_metadata}")
        return
    
    if not os.path.exists(val_metadata):
        print(f"❌ Validation metadata not found: {val_metadata}")
        return
    
    print("✅ Dataset structure verified!")
    
    # Step 3: Show dataset statistics
    print("\n📊 Dataset Statistics:")
    try:
        with open(train_metadata, 'r') as f:
            train_count = len(f.readlines())
        with open(val_metadata, 'r') as f:
            val_count = len(f.readlines())
        
        print(f"   📁 Train images: {train_count}")
        print(f"   📁 Validation images: {val_count}")
        print(f"   📁 Total images: {train_count + val_count}")
        
    except Exception as e:
        print(f"⚠️  Could not count images: {str(e)}")
    
    # Step 4: Check if training config exists
    config_path = "./config/train_shipping_labels.yaml"
    if not os.path.exists(config_path):
        print(f"❌ Training config not found: {config_path}")
        return
    
    print("✅ Training configuration found!")
    
    # Step 5: Ask user if they want to start training
    print("\n🎯 Ready to start training!")
    print("="*60)
    print("Your dataset is now ready for Donut training.")
    print()
    print("To start training, run:")
    print(f"python train.py --config {config_path} --exp_version shipping_labels_v1")
    print()
    print("Or use this command:")
    print(f"python train.py --config {config_path} --dataset_name_or_paths '[\"{dataset_path}\"]' --exp_version shipping_labels_v1")
    print()
    
    # Ask if user wants to start training now
    response = input("Do you want to start training now? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\n🚀 Starting training...")
        try:
            # Start training
            cmd = [
                sys.executable, "train.py",
                "--config", config_path,
                "--dataset_name_or_paths", f'["{dataset_path}"]',
                "--exp_version", "shipping_labels_v1"
            ]
            
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed with error code: {e.returncode}")
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
        except Exception as e:
            print(f"❌ Unexpected error during training: {str(e)}")
    else:
        print("⏸️  Training not started. You can run it manually later.")
    
    print("\n✅ Setup completed!")
    print("📁 Dataset location: ./shipping_labels_dataset")
    print("⚙️  Config location: ./config/train_shipping_labels.yaml")


if __name__ == "__main__":
    main() 