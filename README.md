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
├── src/                              # Source code
│   ├── dataset/                      # Data loading and tokenizer
│   │   ├── data_processing.py
│   │   └── tokenizer.py
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
│       └── utils.py
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

## Dependencies

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Optuna (for hyperparameter optimization)
- vLLM (for efficient text generation)
- Pandas & NumPy (data processing)

For detailed version information, please refer to `requirements.txt`.
