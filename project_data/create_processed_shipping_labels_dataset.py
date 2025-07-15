import os
import shutil
import json

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG_TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'shipping_labels_dataset 2', 'train')
ORIG_VAL_DIR = os.path.join(BASE_DIR, 'dataset', 'shipping_labels_dataset 2', 'validation')
PROC_TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'shipping_labels_dataset_processed', 'train')
PROC_VAL_DIR = os.path.join(BASE_DIR, 'dataset', 'shipping_labels_dataset_processed', 'validation')

# Ground truth template (from your example)
gt_dict = {
    "gt_parse": {
        "sender": "CARDHOLDER SERVICES PERSONALIZATION DEPT 345 KAUTZ RD SAINT CHARLES IL 60174",
        "recipient": "Kamlesh Srivastava 4390 US Highway 1 Ste 312 Princeton NJ 08540-5747"
    }
}

def process_folder(orig_dir, proc_dir):
    os.makedirs(proc_dir, exist_ok=True)
    metadata_path = os.path.join(proc_dir, 'metadata.jsonl')
    entries = []
    for fname in os.listdir(orig_dir):
        if fname.endswith('.png') or fname.endswith('.jpg'):
            # Copy file
            src = os.path.join(orig_dir, fname)
            dst = os.path.join(proc_dir, fname)
            shutil.copy2(src, dst)
            # Prepare metadata entry
            entry = {
                "file_name": fname,
                "ground_truth": json.dumps(gt_dict)
            }
            entries.append(entry)
    # Write metadata.jsonl
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    print(f"Processed {len(entries)} files in {proc_dir}")

def main():
    process_folder(ORIG_TRAIN_DIR, PROC_TRAIN_DIR)
    process_folder(ORIG_VAL_DIR, PROC_VAL_DIR)

if __name__ == '__main__':
    main() 