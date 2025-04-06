# 🧠 QuantLLM: Lightweight Library for Quantized LLM Fine-Tuning and Deployment

## 📌 Overview

**QuantLLM** is a Python library designed for developers, researchers, and teams who want to fine-tune and deploy large language models (LLMs) **efficiently** using **4-bit and 8-bit quantization** techniques. It provides a modular and flexible framework for:

- **Loading and quantizing models** with advanced configurations
- **LoRA / QLoRA-based fine-tuning** with customizable parameters
- **Dataset management** with preprocessing and splitting
- **Training and evaluation** with comprehensive metrics
- **Model checkpointing** and versioning
- **Hugging Face Hub integration** for model sharing

The goal of QuantLLM is to **democratize LLM training**, especially in low-resource environments, while keeping the workflow intuitive, modular, and production-ready.

## 🎯 Key Features

| Feature                          | Description |
|----------------------------------|-------------|
| ✅ Quantized Model Loading       | Load any HuggingFace model in 4-bit or 8-bit precision with customizable quantization settings |
| ✅ Advanced Dataset Management   | Load, preprocess, and split datasets with flexible configurations |
| ✅ LoRA / QLoRA Fine-Tuning      | Memory-efficient fine-tuning with customizable LoRA parameters |
| ✅ Comprehensive Training        | Advanced training loop with mixed precision, gradient accumulation, and early stopping |
| ✅ Model Evaluation             | Flexible evaluation with custom metrics and batch processing |
| ✅ Checkpoint Management        | Save, resume, and manage training checkpoints with versioning |
| ✅ Hub Integration              | Push models and checkpoints to Hugging Face Hub with authentication |
| ✅ Configuration Management     | YAML/JSON config support for reproducible experiments |
| ✅ Logging and Monitoring       | Comprehensive logging and Weights & Biases integration |

## 🚀 Getting Started

### 🔧 Installation

```bash
pip install quantllm
```

### 📦 Basic Usage

```python
from quantllm import (
    ModelLoader,
    DatasetLoader,
    DatasetPreprocessor,
    DatasetSplitter,
    FineTuningTrainer,
    ModelEvaluator,
    HubManager,
    CheckpointManager,
)
import os
from quantllm.finetune import TrainingLogger
from quantllm.config import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
)

# Initialize logger
logger = TrainingLogger()

# 1. Initialize hub manager first
hub_manager = HubManager(
    model_id="your-username/llama-2-imdb",
    token=os.getenv("HF_TOKEN")
)

# 2. Model Configuration and Loading
model_config = ModelConfig(
    model_name="meta-llama/Llama-3.2-3B",
    load_in_4bit=True,
    use_lora=True,
    hub_manager=hub_manager
)

model_loader = ModelLoader(model_config)
model = model_loader.get_model()
tokenizer = model_loader.get_tokenizer()

# 3. Dataset Configuration and Loading
dataset_config = DatasetConfig(
    dataset_name_or_path="imdb",
    dataset_type="huggingface",
    text_column="text",
    label_column="label",
    max_length=512,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    hub_manager=hub_manager
)

# Load and prepare dataset
dataset_loader = DatasetLoader(logger)
dataset = dataset_loader.load_hf_dataset(dataset_config)

# Split dataset
dataset_splitter = DatasetSplitter(logger)
train_dataset, val_dataset, test_dataset = dataset_splitter.train_val_test_split(
    dataset,
    train_size=dataset_config.train_size,
    val_size=dataset_config.val_size,
    test_size=dataset_config.test_size
)

# 4. Dataset Preprocessing
preprocessor = DatasetPreprocessor(tokenizer, logger)
train_dataset, val_dataset, test_dataset = preprocessor.tokenize_dataset(
    train_dataset, val_dataset, test_dataset,
    max_length=dataset_config.max_length,
    text_column=dataset_config.text_column,
    label_column=dataset_config.label_column
)

# Create data loaders
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4
)

# 5. Training Configuration
training_config = TrainingConfig(
    learning_rate=2e-4,
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=50,
    eval_steps=200,
    save_steps=500,
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)

# Initialize checkpoint manager
checkpoint_manager = CheckpointManager(
    output_dir="./checkpoints",
    save_total_limit=3
)

# 6. Initialize Trainer
trainer = FineTuningTrainer(
    model=model,
    training_config=training_config,
    train_dataloader=train_dataloader,
    eval_dataloader=val_dataloader,
    logger=logger,
    checkpoint_manager=checkpoint_manager,
    hub_manager=hub_manager,
    use_wandb=True,
    wandb_config={
        "project": "quantllm-imdb",
        "name": "llama-2-imdb-finetuning"
    }
)

# 7. Train the model
trainer.train()

# 8. Evaluate on test set
evaluator = ModelEvaluator(
    model=model,
    eval_dataloader=test_dataloader,
    metrics=[
        lambda preds, labels, _: (preds.argmax(dim=-1) == labels).float().mean().item()  # Accuracy
    ],
    logger=logger
)

test_metrics = evaluator.evaluate()

# 9. Save final model
trainer.save_model("./final_model")

# 10. Push to Hub if logged in
if hub_manager.is_logged_in():
    hub_manager.push_model(
        model,
        commit_message=f"Final model with test accuracy: {test_metrics.get('accuracy', 0):.4f}"
    )
```

### ⚙️ Advanced Usage

#### Configuration Files

Create a config file (e.g., `config.yaml`):
```yaml
model:
  model_name: "meta-llama/Llama-3.2-3B"
  load_in_4bit: true
  use_lora: true
  lora_config:
    r: 16
    lora_alpha: 32
    target_modules: ["q_proj", "v_proj"]

dataset:
  dataset_name_or_path: "imdb"
  text_column: "text"
  label_column: "label"
  max_length: 512
  train_size: 0.8
  val_size: 0.1
  test_size: 0.1

training:
  learning_rate: 2e-4
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  warmup_steps: 100
  logging_steps: 50
  eval_steps: 200
  save_steps: 500
  early_stopping_patience: 3
  early_stopping_threshold: 0.01
```

## 📚 Documentation

### Model Loading

```python
model_config = ModelConfig(
    model_name="meta-llama/Llama-3.2-3B",
    load_in_4bit=True,
    use_lora=True,
    hub_manager=hub_manager
)
```

### Dataset Management

```python
dataset_config = DatasetConfig(
    dataset_name_or_path="imdb",
    dataset_type="huggingface",
    text_column="text",
    label_column="label",
    max_length=512,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    hub_manager=hub_manager
)
```

### Training Configuration

```python
training_config = TrainingConfig(
    learning_rate=2e-4,
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_steps=50,
    eval_steps=200,
    save_steps=500,
    early_stopping_patience=3,
    early_stopping_threshold=0.01
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [HuggingFace](https://huggingface.co/) for their amazing Transformers library
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for quantization
- [PEFT](https://github.com/huggingface/peft) for parameter-efficient fine-tuning
- [Weights & Biases](https://wandb.ai/) for experiment tracking 