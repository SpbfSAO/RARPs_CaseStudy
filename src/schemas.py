from typing import List
from pydantic import BaseModel


class AEDataConfig(BaseModel):
    download_fits: bool
    preprocess_fits: bool
    folder_name: str
    preprocess_fits_name: str
    years: List[str]
    height: int
    width: int
    channels: int
    clip_factor: float

class AEModelConfig(BaseModel):
    latent_dim: int
    epochs: int
    batch_size: int
    lr: float
    save_path: str
    train_model: bool

class AEOutputConfig(BaseModel):
    embeddings_csv: str
    loss_plot: str
    num_visual_examples: int
    grayscale: bool
    train_color: str
    val_color: str
    plot_losses: bool
    extract_embeddings: bool
    visualize_reconstructions: bool
    evaluate_reconstructions: bool

class AEInterimFilesConfig(BaseModel):
    original_data_for_viz_path: str
    reconstructed_data_save_path: str
    visualization_filenames_path: str
    train_losses_path: str
    val_losses_path: str
    filepaths_path: str

class AEConfig(BaseModel):
    data: AEDataConfig
    model: AEModelConfig
    output: AEOutputConfig
    interim_files: AEInterimFilesConfig
    device: str


class DataPaths(BaseModel):
    sharps_train_save: str
    sharps_test_save: str
    ratan_train_save: str
    ratan_test_save: str
    ratan_embeddings: str

class DataConfig(BaseModel):
    sharp_years_train: List[int]
    sharp_years_test: List[int]
    sharp_features: List[str]
    ratan_features: List[str]
    target_cols: List[str]
    data_paths: DataPaths

class LogRegModelParams(BaseModel):
    class_weight: str
    random_state: int

class ModelConfig(BaseModel):
    name: str
    params: LogRegModelParams
    cv_splits: int

class OutputTitles(BaseModel):
    is_C_24: str
    is_C_48: str
    Mplus_24: str
    Mplus_48: str

class OutputConfig(BaseModel):
    prediction_csv_sharp: str
    prediction_csv_ratan: str
    thresholds: List[float]
    titles: OutputTitles
    color: bool
    show_fig: bool

class MetricsConfig(BaseModel):
   compute_ci: bool
   n_bootstrap: int
   ci_percentile: int 

class StepsConfig(BaseModel):
    prepare_data: bool
    train_and_predict: bool
    evaluate_and_visualize: bool

class ModellingConfig(BaseModel):
    data: DataConfig
    model: ModelConfig
    metrics: MetricsConfig
    output: OutputConfig
    steps: StepsConfig
    device: str
