import json
import os

def create_metadata_jsonl(train_folder_path, val_folder_path, ground_truth_content):
    """
    Simple function to create train and val metadata.jsonl files.
    
    Args:
        train_folder_path: Path to train images folder
        val_folder_path: Path to val images folder  
        ground_truth_content: Dict with the ground truth content (same for all images)
    """
    
    # Format ground truth for Donut
    formatted_gt = {"gt_parse": ground_truth_content}
    gt_string = json.dumps(formatted_gt, ensure_ascii=False)
    
    # Create train metadata.jsonl
    os.makedirs('train', exist_ok=True)
    train_images = [f for f in os.listdir(train_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    with open('train/metadata.jsonl', 'w') as f:
        for img in train_images:
            json.dump({"file_name": img, "ground_truth": gt_string}, f)
            f.write('\n')
    
    # Create val metadata.jsonl  
    os.makedirs('validation', exist_ok=True)
    val_images = [f for f in os.listdir(val_folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    with open('validation/metadata.jsonl', 'w') as f:
        for img in val_images:
            json.dump({"file_name": img, "ground_truth": gt_string}, f)
            f.write('\n')
    
    print(f"Created train/metadata.jsonl with {len(train_images)} entries")
    print(f"Created validation/metadata.jsonl with {len(val_images)} entries")

# Usage
ground_truth = {
    "sender": "CARDHOLDER SERVICES\nPERSONALIZATION DEPT\n345 KAUTZ RD\nSAINT CHARLES IL 60174",
    "recipient": "Kamlesh Srivastava\n4390 US Highway 1 Ste 312\nPrinceton NJ 08540-5747"
}

create_metadata_jsonl(
    "/Users/varsha/Desktop/Dori/ocr/shipping_labels2/train1",
    "/Users/varsha/Desktop/Dori/ocr/shipping_labels2/val1", 
    ground_truth
) 