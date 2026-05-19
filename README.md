# GPT-2 Custom Implementation

A custom implementation of GPT-2 architecture supporting model training, hyperparameter optimization, and text generation functionalities.

## Key Features

- **Model Training**: Support training GPT-2 model from scratch or fine-tuning pretrained models
- **Hyperparameter Optimization**: Integrated with Optuna for hyperparameter search and optimization
- **Text Generation**: Text generation based on trained models (GPU via vLLM, CPU via Transformers pipeline)
- **Reproducibility**: Fixed random seeds to ensure reproducible experimental results

## Directory Structure

```
.
├── configs/                          # Configuration files
│   ├── model_parameters.example.yml  # Model parameter configuration template
│   ├── hparam_tuning.example.yml     # HPO configuration template
│   ├── logging_config.example.json   # vLLM logging configuration template
│   └── gpt2-original/               # GPT-2 tokenizer configuration files
├── custom/                           # User-defined custom modules (gitignored)
├── src/                              # Source code
│   ├── dataset/                      # Data loading and tokenizer
│   │   ├── data_processing.py
│   │   ├── tokenizer.py
│   │   └── data_collator.py
│   ├── training/                     # Training and HPO
│   │   ├── train.py
│   │   ├── trainer_builder.py
│   │   └── hpo.py
│   ├── model/                        # Model construction
│   │   └── model_utils.py
│   ├── generation/                   # Text generation
│   │   └── generate.py
│   └── utils/                        # Utilities
│       ├── setup_seed.py
│       ├── utils.py
│       └── custom_module_loader.py
├── outputs/                          # Runtime outputs (gitignored)
│   ├── training/{jobtype}/{time}/
│   ├── generation/{jobtype}/{time}/
│   └── hpo/{jobtype}/{time}/
├── main.py                           # Program entry point
└── requirements.txt                  # Project dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Configure Parameters

Copy and modify the parameter configuration files:

```bash
cp configs/model_parameters.example.yml configs/model_parameters.yml
cp configs/hparam_tuning.example.yml configs/hparam_tuning.yml
```

In `configs/model_parameters.yml`, set the following key parameters:
- `mode`: Running mode (training/fine-tuning/hpo/generation)
- `jobtype`: Job type name
- Dataset paths and other parameters

### 2. Run the Program

```bash
python main.py
```

Based on the configured `mode` parameter, the program will perform corresponding functions:

- **Training mode (training/fine-tuning)**: Train or fine-tune GPT-2 model
- **Hyperparameter optimization mode (hpo)**: Use Optuna for hyperparameter search
- **Generation mode (generation)**: Generate texts using trained model

## Component Descriptions

### Data Processing (src/dataset/)
Responsible for data loading, preprocessing, and tokenizer construction.

### Model Training (src/training/)
Contains model training and hyperparameter optimization functionality, using Hugging Face Transformers library for training.

### Text Generation (src/generation/)
Efficient text generation based on vLLM engine (GPU) or Transformers pipeline (CPU).

### Model Definition (src/model/)
GPT-2 model architecture definition and construction tools.

## Custom Module Extension

The framework supports user-defined Python modules to customize training behavior without modifying framework code. Set in `configs/model_parameters.yml`:

```yaml
custom_module: custom/custom_module.py
```

To get started, copy the template:

```bash
cp custom/custom_module.example.py custom/custom_module.py
```

A custom module can define any combination of the following (all optional):

| Name | Type | Purpose |
|------|------|---------|
| `tokenize(tokenizer, batch, target, context_length)` | function | Override default tokenization (e.g. label masking) |
| `tk_batched` | `bool` | Control `dataset.map(batched=?)` (default: `True`) |
| `compute_metrics(eval_pred)` | function | Custom evaluation metrics |
| `preprocess_logits_for_metrics(logits, labels)` | function | Pre-process logits before metrics (e.g. argmax) |
| `metric_for_best_model` | `str` | Metric name for best model selection (default: `"eval_loss"`) |
| `greater_is_better` | `bool` | Whether the metric is higher-is-better (default: `False`) |

When `custom_module` is `null` (default), all behavior is identical to the standard training pipeline.

**Key behaviors:**
- If the custom `tokenize` returns a `labels` column, the framework automatically switches to a data collator that pads labels with `-100`.
- The tokenizer is injected into the custom module at runtime (`custom.tokenizer`), making it available for `compute_metrics`.
- The custom module file is included in the dataset cache hash, so modifying it automatically invalidates the tokenization cache.

## Dependencies

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Optuna (for hyperparameter optimization)
- vLLM (for efficient text generation)
- Pandas & NumPy (data processing)

For detailed version information, please refer to `requirements.txt`.
