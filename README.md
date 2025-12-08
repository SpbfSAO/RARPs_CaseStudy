# Solar Activity Forecasting Based on Radio Data from RATAN-600

## Project Overview

This project investigates the potential of **ground-based radio telescope observations** for predicting solar flares. Unlike traditional approaches that rely on data from space-based observatories (e.g., HMI/SHARP on the SDO satellite), we explore the feasibility of using two-channel radio scans from the **RATAN-600** complex.

Our research demonstrates that radio data can potentially serve as an **effective and independent alternative** for extracting key features necessary for forecasting. The project implements a two-stage pipeline:

1.  **Feature Extraction**: Using a Convolutional Autoencoder (ConvAE) trained on two-channel radio scans, we compress the data into informative latent representations (embeddings).
2.  **Forecasting**: Based on these embeddings, as well as traditional HMI/SHARP data (as a control group), a logistic regression model is trained to predict solar flares.

-----

## Project Structure

```
.
├── config/                     # YAML configuration files for pipelines
│   ├── ae_config.yaml          # Settings for the autoencoder pipeline
│   ├── modelling_config.yaml   # Settings for the logistic regression pipeline
│   └── main_config.yaml        # Flags for launching pipelines
├── data/                       # Data storage directories
│   ├── external/               # External data (e.g., synchronization files)
│   ├── processed/              # Processed data (embeddings, predictions)
│   └── raw/                    # Raw source data (RATAN scans, SHARP data)
├── LICENSE                     # MIT License file
├── log/                        # Script execution logs
├── main.py                     # Main executable script
├── notebooks/                  # Jupyter notebooks for experiments and analysis
│   └── EvolutionCurves.ipynb   # Active Region Evolution Curves visualization example
├── output                      # Computing results
│   ├── reports/                # Reports and generated plots
│   └── ae/                     # AutoEncoder computing results (figures, inference, scores)    
├── requirements.txt            # List of Python dependencies
└── src/                        # Project source code
    ├── config_models.py        # Pydantic models for configuration validation
    ├── constants.py            # Constants and directory paths
    ├── dataset.py              # PyTorch Dataset class for data loading
    ├── logger.py               # Logging setup
    ├── models.py               # ConvAE architecture definition
    ├── plot.py                 # Visualization functions
    ├── scripts/                # Modular steps for pipelines
    │   ├── embeddings.py       # Autoencoder pipeline steps
    │   └── modelling.py        # ML (Logistic regression) pipeline steps
    └── utils.py                # Utility functions
```

-----

## Project Configuration

All project parameters, from data paths to model hyperparameters, are defined in YAML files within the `config/` directory. This allows for easy reproduction of experiments and project customization.

*   `main_config.yaml`: Determines which of the two main pipelines (`ae` and `logreg`) will be executed.
*   `ae_config.yaml`: Contains all settings for the autoencoder pipeline, including training parameters, model architecture, and visualization options.
*   `modelling_config.yaml`: Contains settings for the ML pipeline, including the list of features, target variables, and model parameters.

-----

## Running the Project

Follow these steps to reproduce the results:

### 1. Environment Setup

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/mezhbert/flare_radio_forecasting.git
    cd flare_radio_forecasting
    ```

2.  **Create and activate a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

### 2. Running Pipelines

The main executable script is `main.py`. It launches pipelines based on the flags set in `config/main_config.yaml`.

```bash
python main.py
```

After execution, all results, including saved models, plots, and prediction tables, will be available in the `models/`, `reports/`, and `data/processed/` directories.

-----

## Link to the article

The Ratan Active Region Patches (RARPs) Database: A New Database of Solar Active Region Radio Signatures from the RATAN-600 Telescope:

https://arxiv.org/abs/2512.05702


-----

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.



