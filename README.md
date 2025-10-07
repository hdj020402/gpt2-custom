# GPT-2 Custom Implementation

A custom implementation of GPT-2 architecture supporting model training, hyperparameter optimization, and text generation functionalities.

## Key Features

- **Model Training**: Support training GPT-2 model from scratch or fine-tuning pretrained models
- **Hyperparameter Optimization**: Integrated with Optuna for hyperparameter search and optimization
- **Text Generation**: Text generation based on trained models
- **Reproducibility**: Fixed random seeds to ensure reproducible experimental results

## Directory Structure

```
.
├── dataset/                 # Data processing related code
│   ├── data_processing.py   # Data loading and preprocessing
│   └── tokenizer.py         # Tokenizer related functions
├── generation/              # Text generation related code
│   ├── generate.py          # Main text generation program
│   └── logging_config.json  # Logging configuration file
├── gpt2-original/           # Original GPT-2 model configuration files
├── model/                   # Model definition related code
│   └── model_utils.py       # Model construction utilities
├── training/                # Training related code
│   ├── hpo.py               # Hyperparameter optimization
│   ├── train.py             # Main training program
│   └── trainer_builder.py   # Trainer builder
├── utils/                   # Utility functions
│   ├── setup_seed.py        # Random seed setup
│   └── utils.py             # Other utility functions
├── main.py                  # Program entry point
├── model_parameters_example.yml  # Model parameter configuration example
├── hparam_tuning_example.yml     # Hyperparameter optimization configuration example
└── requirements.txt         # Project dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Configure Parameters

Copy and modify the parameter configuration files:

```bash
cp model_parameters_example.yml model_parameters.yml
cp hparam_tuning_example.yml hparam_tuning.yml
```

In [model_parameters.yml](file:///home2/hdj/Toolkits/Machine_Learning/gpt2-custom/model_parameters.yml), set the following key parameters:
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

### Data Processing ([dataset/](file:///home2/hdj/Toolkits/Machine_Learning/gpt2-custom/dataset))
Responsible for data loading, preprocessing, and tokenizer construction.

### Model Training ([training/](file:///home2/hdj/Toolkits/Machine_Learning/gpt2-custom/training))
Contains model training and hyperparameter optimization functionality, using Hugging Face Transformers library for training.

### Text Generation ([generation/](file:///home2/hdj/Toolkits/Machine_Learning/gpt2-custom/generation))
Efficient text generation based on vLLM engine.

### Model Definition ([model/](file:///home2/hdj/Toolkits/Machine_Learning/gpt2-custom/model))
GPT-2 model architecture definition and construction tools.

## Dependencies

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Optuna (for hyperparameter optimization)
- vLLM (for efficient text generation)
- Pandas & NumPy (data processing)

For detailed version information, please refer to [requirements.txt](file:///home2/hdj/Toolkits/Machine_Learning/gpt2-custom/requirements.txt).