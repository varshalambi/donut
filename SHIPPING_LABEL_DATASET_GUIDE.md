# 📦 Shipping Label Dataset Creation Guide for Donut

## **🎯 Overview**

Building a shipping label dataset for Donut requires understanding the structure of various carrier labels (Amazon, FedEx, UPS, USPS, DHL, etc.) and creating consistent JSON annotations. This guide covers everything from data collection to annotation strategies.

---

## **📊 Shipping Label Structure Analysis**

### **Common Shipping Label Elements**

```json
{
  "carrier": "fedex",
  "tracking_number": "123456789012345",
  "sender": {
    "name": "John Doe",
    "address": {
      "street": "123 Main St",
      "city": "New York",
      "state": "NY",
      "zip": "10001",
      "country": "USA"
    }
  },
  "recipient": {
    "name": "Jane Smith",
    "address": {
      "street": "456 Oak Ave",
      "city": "Los Angeles",
      "state": "CA",
      "zip": "90210",
      "country": "USA"
    }
  },
  "package": {
    "weight": "2.5 lbs",
    "dimensions": "12x8x4 in",
    "service_type": "Ground",
    "declared_value": "$100.00"
  },
  "shipping_date": "2024-01-15",
  "barcode": {
    "type": "Code128",
    "value": "123456789012345"
  }
}
```

### **Carrier-Specific Variations**

#### **Amazon Labels**

```json
{
  "carrier": "amazon",
  "amazon_order_id": "112-3456789-1234567",
  "tracking_number": "TBA123456789",
  "recipient": {
    "name": "Customer Name",
    "address": "123 Delivery St, City, State ZIP"
  },
  "package": {
    "weight": "1.2 lbs",
    "dimensions": "10x6x3 in"
  },
  "fulfillment_center": "BOS1",
  "shipment_id": "FBA123456789"
}
```

#### **FedEx Labels**

```json
{
  "carrier": "fedex",
  "tracking_number": "123456789012",
  "sender": {
    "name": "Sender Name",
    "address": "123 Sender St, City, State ZIP"
  },
  "recipient": {
    "name": "Recipient Name",
    "address": "456 Recipient Ave, City, State ZIP"
  },
  "package": {
    "weight": "3.5 lbs",
    "service_type": "Ground",
    "declared_value": "$150.00"
  },
  "barcode": {
    "type": "Code128",
    "value": "123456789012"
  }
}
```

#### **UPS Labels**

```json
{
  "carrier": "ups",
  "tracking_number": "1Z999AA1234567890",
  "sender": {
    "name": "Sender Name",
    "address": "123 Sender St, City, State ZIP"
  },
  "recipient": {
    "name": "Recipient Name",
    "address": "456 Recipient Ave, City, State ZIP"
  },
  "package": {
    "weight": "2.8 lbs",
    "service_type": "Ground",
    "reference_numbers": ["REF123", "REF456"]
  },
  "barcode": {
    "type": "Code128",
    "value": "1Z999AA1234567890"
  }
}
```

---

## **📸 Data Collection Strategies**

### **1. Real-World Collection**

```python
# Data collection script
import os
from PIL import Image
import json

def collect_shipping_labels():
    """Collect shipping label images from various sources"""

    # Sources to collect from:
    sources = {
        "amazon": "amazon_labels/",
        "fedex": "fedex_labels/",
        "ups": "ups_labels/",
        "usps": "usps_labels/",
        "dhl": "dhl_labels/",
        "other": "other_labels/"
    }

    # Collection methods:
    methods = [
        "photograph_packages",      # Take photos of actual packages
        "scan_labels",             # Scan labels from packages
        "download_samples",        # Download from carrier websites
        "crowdsource",             # Collect from multiple people
        "synthetic_generation"     # Generate synthetic labels
    ]

    return sources, methods
```

### **2. Synthetic Data Generation**

```python
# Synthetic label generator
import random
from faker import Faker

fake = Faker()

def generate_synthetic_label(carrier="fedex"):
    """Generate synthetic shipping label data"""

    label_data = {
        "carrier": carrier,
        "tracking_number": generate_tracking_number(carrier),
        "sender": {
            "name": fake.name(),
            "address": {
                "street": fake.street_address(),
                "city": fake.city(),
                "state": fake.state_abbr(),
                "zip": fake.zipcode(),
                "country": "USA"
            }
        },
        "recipient": {
            "name": fake.name(),
            "address": {
                "street": fake.street_address(),
                "city": fake.city(),
                "state": fake.state_abbr(),
                "zip": fake.zipcode(),
                "country": "USA"
            }
        },
        "package": {
            "weight": f"{random.uniform(0.5, 50):.1f} lbs",
            "dimensions": f"{random.randint(4, 24)}x{random.randint(4, 18)}x{random.randint(1, 12)} in",
            "service_type": random.choice(["Ground", "Express", "Priority", "Standard"]),
            "declared_value": f"${random.randint(10, 1000)}.00"
        }
    }

    return label_data

def generate_tracking_number(carrier):
    """Generate carrier-specific tracking numbers"""
    patterns = {
        "fedex": lambda: f"{random.randint(100000000000, 999999999999)}",
        "ups": lambda: f"1Z{random.randint(1000000000000000, 9999999999999999)}",
        "usps": lambda: f"{random.randint(9400000000000000000000, 9499999999999999999999)}",
        "amazon": lambda: f"TBA{random.randint(100000000, 999999999)}",
        "dhl": lambda: f"{random.randint(100000000000, 999999999999)}"
    }
    return patterns.get(carrier, patterns["fedex"])()
```

### **3. Data Augmentation**

```python
# Image augmentation for robustness
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def augment_shipping_label(image_path, output_dir):
    """Apply various augmentations to shipping label images"""

    image = Image.open(image_path)
    augmentations = []

    # 1. Rotation (slight angles)
    for angle in [-5, -2, 2, 5]:
        rotated = image.rotate(angle, expand=True)
        augmentations.append(rotated)

    # 2. Brightness variations
    for factor in [0.7, 0.85, 1.15, 1.3]:
        enhancer = ImageEnhance.Brightness(image)
        brightened = enhancer.enhance(factor)
        augmentations.append(brightened)

    # 3. Contrast variations
    for factor in [0.8, 0.9, 1.1, 1.2]:
        enhancer = ImageEnhance.Contrast(image)
        contrasted = enhancer.enhance(factor)
        augmentations.append(contrasted)

    # 4. Noise addition
    img_array = np.array(image)
    noise = np.random.normal(0, 10, img_array.shape).astype(np.uint8)
    noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    augmentations.append(Image.fromarray(noisy))

    # 5. Blur (simulate poor focus)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=1))
    augmentations.append(blurred)

    return augmentations
```

---

## **🏷️ Annotation Guidelines**

### **1. JSON Schema Definition**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "carrier": {
      "type": "string",
      "enum": ["amazon", "fedex", "ups", "usps", "dhl", "other"]
    },
    "tracking_number": {
      "type": "string",
      "pattern": "^[A-Z0-9]{10,20}$"
    },
    "sender": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "address": {
          "type": "object",
          "properties": {
            "street": { "type": "string" },
            "city": { "type": "string" },
            "state": { "type": "string" },
            "zip": { "type": "string" },
            "country": { "type": "string" }
          }
        }
      }
    },
    "recipient": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "address": {
          "type": "object",
          "properties": {
            "street": { "type": "string" },
            "city": { "type": "string" },
            "state": { "type": "string" },
            "zip": { "type": "string" },
            "country": { "type": "string" }
          }
        }
      }
    },
    "package": {
      "type": "object",
      "properties": {
        "weight": { "type": "string" },
        "dimensions": { "type": "string" },
        "service_type": { "type": "string" },
        "declared_value": { "type": "string" }
      }
    }
  },
  "required": ["carrier", "tracking_number", "recipient"]
}
```

### **2. Annotation Best Practices**

#### **Consistency Rules**

```python
# Annotation consistency guidelines
ANNOTATION_RULES = {
    "address_format": "Street, City, State ZIP",  # Consistent format
    "weight_format": "X.X lbs",                   # Always include units
    "dimensions_format": "LxWxH in",             # Length x Width x Height
    "date_format": "YYYY-MM-DD",                  # ISO format
    "currency_format": "$X.XX",                   # Include dollar sign
    "tracking_format": "Exact as shown",          # Preserve original format
    "name_format": "Full name as shown",          # Include titles if present
    "service_type": "Standardized names"          # Use consistent service names
}
```

#### **Handling Edge Cases**

```python
# Edge case handling strategies
EDGE_CASES = {
    "illegible_text": "Mark as [ILLEGIBLE]",
    "partial_information": "Include what's visible, mark missing as [MISSING]",
    "multiple_addresses": "Use primary/return address distinction",
    "international_labels": "Include country codes and customs info",
    "damaged_labels": "Annotate visible portions only",
    "handwritten_labels": "Transcribe exactly as written",
    "barcode_only": "Extract barcode data, mark other fields as [BARCODE_ONLY]"
}
```

### **3. Annotation Tools**

#### **Label Studio Configuration**

```yaml
# label_studio_config.xml
<View>
<Image name="image" value="$image"/>
<TextArea name="ground_truth" toName="image"
perRegion="true" displayMode="region-list">
<Label value="Shipping Label"/>
</TextArea>
<Choices name="carrier" toName="image" perRegion="true">
<Choice value="amazon"/>
<Choice value="fedex"/>
<Choice value="ups"/>
<Choice value="usps"/>
<Choice value="dhl"/>
<Choice value="other"/>
</Choices>
</View>
```

#### **Custom Annotation Script**

```python
# Custom annotation helper
class ShippingLabelAnnotator:
    def __init__(self):
        self.carrier_templates = self.load_carrier_templates()
        self.validation_rules = self.load_validation_rules()

    def annotate_label(self, image_path, carrier_type):
        """Annotate a shipping label image"""

        # 1. Load carrier-specific template
        template = self.carrier_templates[carrier_type]

        # 2. Extract text using OCR (for reference)
        ocr_text = self.extract_text(image_path)

        # 3. Manual annotation with template guidance
        annotation = self.manual_annotation(image_path, template, ocr_text)

        # 4. Validate annotation
        validation_result = self.validate_annotation(annotation)

        return annotation, validation_result

    def validate_annotation(self, annotation):
        """Validate annotation against schema and rules"""
        errors = []

        # Check required fields
        required_fields = ["carrier", "tracking_number", "recipient"]
        for field in required_fields:
            if field not in annotation:
                errors.append(f"Missing required field: {field}")

        # Validate tracking number format
        if "tracking_number" in annotation:
            if not self.is_valid_tracking_number(annotation["tracking_number"], annotation["carrier"]):
                errors.append("Invalid tracking number format")

        # Validate address format
        if "recipient" in annotation and "address" in annotation["recipient"]:
            if not self.is_valid_address(annotation["recipient"]["address"]):
                errors.append("Invalid address format")

        return {"valid": len(errors) == 0, "errors": errors}
```

---

## **📊 Dataset Organization**

### **1. Directory Structure**

```
shipping_labels_dataset/
├── images/
│   ├── train/
│   │   ├── amazon/
│   │   ├── fedex/
│   │   ├── ups/
│   │   ├── usps/
│   │   └── dhl/
│   ├── val/
│   └── test/
├── annotations/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── metadata/
│   ├── carrier_stats.json
│   ├── annotation_stats.json
│   └── quality_metrics.json
└── config/
    ├── schema.json
    ├── annotation_rules.json
    └── validation_rules.json
```

### **2. Dataset Statistics**

```python
# Dataset analysis script
def analyze_dataset(dataset_path):
    """Analyze shipping label dataset statistics"""

    stats = {
        "total_samples": 0,
        "carrier_distribution": {},
        "image_quality": {},
        "annotation_completeness": {},
        "tracking_number_formats": {},
        "address_formats": {},
        "service_types": {}
    }

    # Load annotations
    with open(f"{dataset_path}/annotations/train.jsonl", "r") as f:
        for line in f:
            annotation = json.loads(line)
            stats["total_samples"] += 1

            # Carrier distribution
            carrier = annotation.get("carrier", "unknown")
            stats["carrier_distribution"][carrier] = stats["carrier_distribution"].get(carrier, 0) + 1

            # Service types
            if "package" in annotation and "service_type" in annotation["package"]:
                service = annotation["package"]["service_type"]
                stats["service_types"][service] = stats["service_types"].get(service, 0) + 1

    return stats
```

---

## **🔧 Training Configuration**

### **1. Custom Config for Shipping Labels**

```yaml
# config/train_shipping_labels.yaml
resume_from_checkpoint_path: null
result_path: "./result"
pretrained_model_name_or_path: "naver-clova-ix/donut-base"

# Dataset configuration
dataset_name_or_paths: ["./shipping_labels_dataset"]
sort_json_key: true # Important for consistent JSON ordering

# Model configuration
input_size: [1280, 960] # Good for label readability
max_length: 1024 # Longer sequences for detailed labels
align_long_axis: true # Handle different label orientations

# Training configuration
train_batch_sizes: [4] # Smaller batches for high-res images
val_batch_sizes: [1]
lr: 2e-5 # Slightly lower learning rate
warmup_steps: 500 # More warmup for complex task
num_training_samples_per_epoch: 1000
max_epochs: 50
num_workers: 4
val_check_interval: 1.0
check_val_every_n_epoch: 2
gradient_clip_val: 1.0
verbose: true

# Shipping label specific
task_start_token: "<s_shipping_label>"
prompt_end_token: "</s_shipping_label>"
```

### **2. Custom Dataset Class**

```python
# Custom shipping label dataset
class ShippingLabelDataset(DonutDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.carrier_templates = self.load_carrier_templates()

    def load_carrier_templates(self):
        """Load carrier-specific annotation templates"""
        return {
            "amazon": {
                "required_fields": ["amazon_order_id", "tracking_number", "recipient"],
                "optional_fields": ["fulfillment_center", "shipment_id", "package"]
            },
            "fedex": {
                "required_fields": ["tracking_number", "sender", "recipient"],
                "optional_fields": ["package", "barcode", "shipping_date"]
            },
            "ups": {
                "required_fields": ["tracking_number", "sender", "recipient"],
                "optional_fields": ["package", "barcode", "reference_numbers"]
            }
        }

    def validate_annotation(self, annotation):
        """Validate shipping label annotation"""
        carrier = annotation.get("carrier")
        if carrier not in self.carrier_templates:
            return False, f"Unknown carrier: {carrier}"

        template = self.carrier_templates[carrier]
        for field in template["required_fields"]:
            if field not in annotation:
                return False, f"Missing required field: {field}"

        return True, "Valid annotation"
```

---

## **🎯 Quality Assurance**

### **1. Annotation Quality Metrics**

```python
# Quality metrics calculation
def calculate_annotation_quality(annotations):
    """Calculate annotation quality metrics"""

    metrics = {
        "completeness": 0.0,      # Percentage of required fields filled
        "consistency": 0.0,       # Consistency in formatting
        "accuracy": 0.0,          # Manual verification accuracy
        "inter_annotator_agreement": 0.0  # Agreement between annotators
    }

    total_fields = 0
    filled_fields = 0

    for annotation in annotations:
        # Check completeness
        required_fields = get_required_fields(annotation["carrier"])
        total_fields += len(required_fields)

        for field in required_fields:
            if field in annotation and annotation[field]:
                filled_fields += 1

    metrics["completeness"] = filled_fields / total_fields if total_fields > 0 else 0

    return metrics
```

### **2. Data Validation Pipeline**

```python
# Validation pipeline
def validate_dataset(dataset_path):
    """Comprehensive dataset validation"""

    validation_results = {
        "images": validate_images(dataset_path),
        "annotations": validate_annotations(dataset_path),
        "consistency": validate_consistency(dataset_path),
        "coverage": validate_coverage(dataset_path)
    }

    return validation_results

def validate_images(dataset_path):
    """Validate image quality and format"""
    issues = []

    for image_file in glob.glob(f"{dataset_path}/images/**/*.jpg"):
        try:
            img = Image.open(image_file)

            # Check resolution
            if img.size[0] < 800 or img.size[1] < 600:
                issues.append(f"Low resolution: {image_file}")

            # Check aspect ratio
            aspect_ratio = img.size[0] / img.size[1]
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                issues.append(f"Unusual aspect ratio: {image_file}")

        except Exception as e:
            issues.append(f"Corrupted image: {image_file} - {e}")

    return {"valid": len(issues) == 0, "issues": issues}
```

---

## **🚀 Best Practices Summary**

### **1. Data Collection**

- **Diverse Sources**: Collect from multiple carriers and scenarios
- **Quality Images**: Ensure good lighting and focus
- **Realistic Conditions**: Include damaged, folded, and partial labels
- **Augmentation**: Apply realistic augmentations for robustness

### **2. Annotation**

- **Consistent Schema**: Use standardized JSON structure
- **Quality Control**: Implement validation and review processes
- **Edge Cases**: Handle illegible text and missing information
- **Inter-Annotator Agreement**: Use multiple annotators for validation

### **3. Training**

- **Carrier-Specific Models**: Consider training separate models for major carriers
- **Transfer Learning**: Start with pretrained Donut model
- **Validation Strategy**: Use edit distance and JSON structure validation
- **Incremental Training**: Start with simple labels, add complexity

### **4. Evaluation**

- **Multiple Metrics**: Use edit distance, JSON accuracy, and field-level metrics
- **Real-World Testing**: Test on actual shipping labels
- **Error Analysis**: Analyze common failure cases
- **Continuous Improvement**: Iterate based on performance

This comprehensive guide should help you create a robust shipping label dataset for Donut training. The key is maintaining consistency in annotation while covering the diverse range of label formats and conditions you'll encounter in real-world applications.
