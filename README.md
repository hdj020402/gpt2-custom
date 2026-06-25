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
├── custom/                           # User-defined custom modules
├── scripts/                          # Utility scripts
│   └── scan_runs.py                  # Training run comparison tool
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

**HPO usage notes:**

- Use `warmup_steps` instead of the deprecated `warmup_ratio`.
- `per_device_train_batch_size` and `gradient_accumulation_steps` are bounded by GPU memory — set them to the maximum your GPU supports; there is no need to include them in the search space.
- When using `GridSampler`, every hyperparameter in `hparam_tuning.yml` must be `type: categorical` with an explicit `choices` list. Non-categorical parameters will cause an early error with a clear message.
- The YAML `sampler` and `pruner` sections are fully wired — any optuna sampler/pruner class can be used by setting `type` (e.g. `TPESampler`, `MedianPruner`) plus its constructor arguments.
- To resume an interrupted study, set `continue_trials.continue: true` and point `storage` / `study_name` to the existing database.
- **Generation mode (generation)**: Generate texts using trained model

### 3. Compare Training Runs

```bash
# CSV (all params + metrics, auto-discovered) — open in Excel / pandas
python scripts/scan_runs.py > runs.csv

# Terminal table (compact subset)
python scripts/scan_runs.py --table --filter 1p0kcal --last 10

# Filter + sort
python scripts/scan_runs.py --filter inchi_3M --sort best_eval_loss > runs.csv

# Custom columns for table mode
python scripts/scan_runs.py --table --cols learning_rate,dropout,best_eval_loss
```

`scan_runs.py` scans all training runs under `outputs/training/`, extracts hyperparameters
from `model_parameters.yml` and metrics from training logs, and emits a unified CSV or
terminal table.  All parameter names are preserved verbatim from the YAML config.
See `--help` for all options.

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

The framework supports user-defined Python modules to customize tokenization and evaluation without modifying framework code. Two separate module files can be configured in `configs/model_parameters.yml`:

```yaml
custom_tokenize: custom/custom_tokenize.py  # changes invalidate tokenization cache
custom_metrics: custom/custom_metrics.py    # changes do NOT invalidate cache
```

To get started, copy the templates:

```bash
cp custom/custom_tokenize.example.py custom/custom_tokenize.py
cp custom/custom_metrics.example.py custom/custom_metrics.py
```

### custom_tokenize module

| Name | Type | Purpose |
|------|------|---------|
| `tokenize(tokenizer, batch, target, context_length)` | function | Override default tokenization (e.g. label masking) |
| `tk_batched` | `bool` | Control `dataset.map(batched=?)` (default: `True`) |

### custom_metrics module

| Name | Type | Purpose |
|------|------|---------|
| `compute_metrics(eval_pred)` | function | Custom evaluation metrics |
| `preprocess_logits_for_metrics(logits, labels)` | function | Pre-process logits before metrics (e.g. argmax) |
| `metric_for_best_model` | `str` | Metric name for best model selection (default: `"eval_loss"`) |
| `greater_is_better` | `bool` | Whether the metric is higher-is-better (default: `False`) |

When both are `null` (default), all behavior is identical to the standard training pipeline.

**Key behaviors:**
- If the custom `tokenize` returns a `labels` column, the framework automatically switches to a data collator that pads labels with `-100`.
- The tokenizer is injected into the custom metrics module at runtime, accessible via `sys.modules[__name__].tokenizer` in `compute_metrics`.
- Only the `custom_tokenize` file is included in the dataset cache hash. Modifying `custom_metrics` does **not** trigger re-tokenization.

## Dependencies

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Optuna (for hyperparameter optimization)
- vLLM (for efficient text generation)
- Pandas & NumPy (data processing)

For detailed version information, please refer to `requirements.txt`.
