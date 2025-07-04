# 🔍 Donut Training Pipeline: Step-by-Step Guide

## **📋 Overview**

Donut (Document Understanding Transformer) is an end-to-end OCR-free document understanding system that combines:

- **SwinTransformer Encoder**: Processes document images into embeddings
- **BART Decoder**: Generates structured text from image embeddings
- **JSON Ground Truth**: Structured annotations for training

---

## **🎯 Step 1: Configuration Setup**

### **Config File Structure** (`config/train_cord.yaml`)

```yaml
# Model Configuration
pretrained_model_name_or_path: "naver-clova-ix/donut-base"
input_size: [1280, 960] # Image dimensions (height, width)
max_length: 768 # Maximum sequence length
align_long_axis: False # Image rotation handling

# Dataset Configuration
dataset_name_or_paths: ["naver-clova-ix/cord-v2"]
sort_json_key: False # JSON key sorting for consistency

# Training Configuration
train_batch_sizes: [8]
val_batch_sizes: [1]
lr: 3e-5 # Learning rate
warmup_steps: 300 # Learning rate warmup
max_epochs: 30 # Training epochs
num_workers: 8 # Data loading workers
```

### **What Each Config Parameter Does**:

- **`input_size`**: Resizes all images to this dimension
- **`max_length`**: Maximum tokens the decoder can generate
- **`align_long_axis`**: Rotates images to match target orientation
- **`sort_json_key`**: Ensures consistent JSON structure ordering

---

## **🖼️ Step 2: Dataset Loading & Processing**

### **Dataset Structure**

```python
# Each sample contains:
{
    "image": PIL.Image,           # Document image
    "ground_truth": "JSON string" # Structured annotation
}

# Example ground truth (CORD dataset):
{
    "gt_parse": {
        "menu": [
            {"name": "coffee", "count": "2", "price": "3.50"},
            {"name": "cake", "count": "1", "price": "5.00"}
        ],
        "total": "8.50"
    }
}
```

### **Dataset Processing Flow**

```python
# 1. Load dataset from HuggingFace
dataset = load_dataset("naver-clova-ix/cord-v2", split="train")

# 2. Create DonutDataset wrapper
donut_dataset = DonutDataset(
    dataset_name_or_path="naver-clova-ix/cord-v2",
    donut_model=model,
    max_length=768,
    lazy_loading=True  # Memory efficient
)

# 3. Ground truth tokenization (lazy loading)
def _get_gt_token_sequences(self, idx):
    sample = self.dataset[idx]
    ground_truth = json.loads(sample["ground_truth"])

    # Convert JSON to token sequence
    token_sequence = (
        "<s>" +                    # Task start token
        model.json2token(ground_truth["gt_parse"]) +  # JSON → tokens
        "</s>"                     # End token
    )
    return [token_sequence]
```

### **JSON to Token Conversion**

```python
# Example: JSON → Token Sequence
json_obj = {
    "menu": [
        {"name": "coffee", "count": "2"}
    ]
}

# Converts to:
"<s><s_menu><s_name>coffee</s_name><s_count>2</s_count></s_menu></s>"
```

---

## **🏗️ Step 3: Model Architecture**

### **DonutModel Components**

```python
class DonutModel(PreTrainedModel):
    def __init__(self, config):
        # 1. SwinTransformer Encoder
        self.encoder = SwinEncoder(
            input_size=[1280, 960],
            window_size=10,
            encoder_layer=[2, 2, 14, 2]
        )

        # 2. BART Decoder
        self.decoder = BARTDecoder(
            decoder_layer=4,
            max_position_embeddings=768
        )
```

### **Encoder: SwinTransformer**

```python
class SwinEncoder(nn.Module):
    def forward(self, x):
        # Input: (batch_size, 3, 1280, 960)
        x = self.model.patch_embed(x)    # Split into patches
        x = self.model.pos_drop(x)       # Add positional embeddings
        x = self.model.layers(x)         # SwinTransformer layers
        # Output: (batch_size, 768, 40, 30) - image embeddings
        return x
```

### **Decoder: BART**

```python
class BARTDecoder(nn.Module):
    def forward(self, input_ids, encoder_hidden_states, labels):
        # Input: tokenized ground truth
        # Cross-attention with encoder outputs
        outputs = self.model(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            labels=labels
        )
        return outputs  # Loss and logits
```

---

## **🔄 Step 4: Training Loop**

### **Forward Pass**

```python
def training_step(self, batch, batch_idx):
    # 1. Unpack batch
    image_tensors = batch[0]      # (batch_size, 3, 1280, 960)
    decoder_input_ids = batch[1]  # (batch_size, seq_len)
    decoder_labels = batch[2]     # (batch_size, seq_len)

    # 2. Encoder forward pass
    encoder_outputs = self.model.encoder(image_tensors)
    # encoder_outputs: (batch_size, 768, 40, 30)

    # 3. Decoder forward pass (teacher forcing)
    decoder_outputs = self.model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_outputs,
        labels=decoder_labels
    )

    # 4. Calculate loss
    loss = decoder_outputs.loss
    return loss
```

### **Data Flow Visualization**

```
📄 Document Image (1280×960)
    ↓
🖼️ Image Preprocessing
    ↓
🔢 SwinTransformer Encoder
    ↓
📊 Image Embeddings (768×40×30)
    ↓
🔤 BART Decoder (Cross-Attention)
    ↓
📝 Token Predictions
    ↓
📊 Loss Calculation
```

---

## **🔍 Step 5: Ground Truth Processing**

### **Ground Truth Structure**

```python
# Original JSON annotation
ground_truth = {
    "gt_parse": {
        "menu": [
            {"name": "coffee", "count": "2", "price": "3.50"},
            {"name": "cake", "count": "1", "price": "5.00"}
        ],
        "total": "8.50"
    }
}

# Converted to token sequence
token_sequence = (
    "<s>" +  # Task start
    "<s_menu>" +
        "<s_name>coffee</s_name>" +
        "<s_count>2</s_count>" +
        "<s_price>3.50</s_price>" +
    "</s_menu>" +
    "<s_menu>" +
        "<s_name>cake</s_name>" +
        "<s_count>1</s_count>" +
        "<s_price>5.00</s_price>" +
    "</s_menu>" +
    "<s_total>8.50</s_total>" +
    "</s>"  # End token
)
```

### **Tokenization Process**

```python
def json2token(self, obj):
    if isinstance(obj, dict):
        output = ""
        for key in sorted(obj.keys()):
            # Add special tokens for JSON keys
            self.decoder.add_special_tokens([f"<s_{key}>", f"</s_{key}>"])
            output += f"<s_{key}>" + self.json2token(obj[key]) + f"</s_{key}>"
        return output
    elif isinstance(obj, list):
        return "<sep/>".join([self.json2token(item) for item in obj])
    else:
        return str(obj)
```

---

## **🎯 Step 6: Validation Process**

### **Validation Step**

```python
def validation_step(self, batch, batch_idx):
    # 1. Get image and prompt
    image_tensors = batch[0]
    decoder_input_ids = batch[1]
    prompt_end_idxs = batch[2]
    answers = batch[3]

    # 2. Create prompts (up to prompt end token)
    decoder_prompts = pad_sequence([
        input_id[:end_idx + 1]
        for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)
    ], batch_first=True)

    # 3. Generate predictions
    preds = self.model.inference(
        image_tensors=image_tensors,
        prompt_tensors=decoder_prompts,
        return_json=False
    )["predictions"]

    # 4. Calculate edit distance scores
    scores = []
    for pred, answer in zip(preds, answers):
        # Clean up predictions and answers
        pred = TAG_CLEANUP_PATTERN.sub("", pred)
        answer = HTML_TAG_PATTERN.sub("", answer, count=1)

        # Calculate normalized edit distance
        score = edit_distance(pred, answer) / max(len(pred), len(answer))
        scores.append(score)

    return scores
```

### **Inference Process**

```python
def inference(self, image_tensors, prompt_tensors):
    # 1. Encode image
    encoder_outputs = self.encoder(image_tensors)

    # 2. Generate tokens autoregressively
    decoder_output = self.decoder.model.generate(
        decoder_input_ids=prompt_tensors,
        encoder_outputs=encoder_outputs,
        max_length=self.config.max_length,
        early_stopping=True,
        pad_token_id=self.decoder.tokenizer.pad_token_id,
        eos_token_id=self.decoder.tokenizer.eos_token_id
    )

    # 3. Decode tokens to text
    predictions = self.decoder.tokenizer.batch_decode(decoder_output.sequences)

    return {"predictions": predictions}
```

---

## **📊 Step 7: Loss Calculation**

### **Teacher Forcing Training**

```python
# During training, the model uses teacher forcing:
# - Input: Previous ground truth tokens
# - Target: Next ground truth tokens
# - Loss: Cross-entropy between predictions and targets

# Example:
input_ids = ["<s>", "<s_menu>", "<s_name>", "coffee"]  # Input tokens
labels = ["<s_menu>", "<s_name>", "coffee", "</s_name>"]  # Target tokens

# Model predicts next token given previous tokens
# Loss = CrossEntropy(predictions, labels)
```

### **Label Masking**

```python
# Labels are masked to ignore certain tokens:
labels = input_ids.clone()
labels[labels == pad_token_id] = ignore_id      # Ignore padding
labels[:prompt_end_idx + 1] = ignore_id         # Ignore prompt tokens
```

---

## **🔄 Step 8: Complete Training Flow**

### **Training Pipeline Summary**

```
1. 📋 Load Configuration
   ├── Model parameters (input_size, max_length)
   ├── Training parameters (lr, batch_size, epochs)
   └── Dataset parameters (dataset_path, sort_json_key)

2. 📊 Load Dataset
   ├── Download from HuggingFace
   ├── Create DonutDataset wrapper
   └── Setup lazy loading for memory efficiency

3. 🏗️ Initialize Model
   ├── Load pretrained DonutModel
   ├── Setup SwinTransformer encoder
   └── Setup BART decoder

4. 🔄 Training Loop
   ├── Load batch (images + ground truth)
   ├── Preprocess images (resize, normalize)
   ├── Tokenize ground truth JSON
   ├── Forward pass (encoder + decoder)
   ├── Calculate loss (teacher forcing)
   └── Backward pass (gradient update)

5. ✅ Validation
   ├── Load validation batch
   ├── Generate predictions (autoregressive)
   ├── Calculate edit distance scores
   └── Log validation metrics

6. 💾 Checkpointing
   ├── Save model weights
   ├── Save tokenizer
   └── Log training metrics
```

---

## **🎯 Key Concepts Explained**

### **1. Why No OCR?**

- Traditional OCR: Image → Text → Structure
- Donut: Image → Structure (direct)
- Eliminates OCR errors and intermediate steps

### **2. Cross-Attention Mechanism**

- Encoder provides image context
- Decoder attends to relevant image regions
- Enables understanding of visual layout

### **3. JSON as Ground Truth**

- Structured format for training
- Maintains hierarchical relationships
- Enables direct structured output

### **4. Teacher Forcing**

- Uses ground truth tokens as decoder input
- Ensures stable training
- Prevents error accumulation

### **5. Lazy Loading**

- Processes ground truth on-demand
- Reduces memory usage by 90%
- Enables training on large datasets

---

## **🔧 Configuration Tips**

### **Memory Optimization**

```yaml
# For limited GPU memory:
input_size: [1024, 768] # Smaller images
batch_size: 4 # Smaller batches
max_length: 512 # Shorter sequences
lazy_loading: true # Memory efficient loading
```

### **Speed Optimization**

```yaml
# For faster training:
num_workers: 8 # Parallel data loading
pin_memory: true # Faster GPU transfer
gradient_accumulation: 2 # Effective larger batches
```

### **Quality Optimization**

```yaml
# For better results:
input_size: [1280, 960] # Higher resolution
max_length: 768 # Longer sequences
warmup_steps: 300 # Proper warmup
gradient_clip_val: 1.0 # Stable training
```

This comprehensive guide shows how all components work together to create an end-to-end document understanding system that can extract structured information directly from document images without traditional OCR.
