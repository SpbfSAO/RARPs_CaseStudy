import os
import re
import ast
import yaml
from collections import defaultdict
from astropy.io import fits
from tqdm import tqdm
from datetime import datetime
from typing import List, Tuple, Any
from pydantic import BaseModel, ValidationError

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import scipy.ndimage as ndimage
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, brier_score_loss, roc_auc_score, roc_curve

from src.logger import get_logger
from src.schemas import AEConfig, ModellingConfig


logger = get_logger(__file__)


def load_config(path: str) -> Any:
    """
    Loads configuration from a YAML file.


    This function loads the content of a YAML file and returns it
    as a dictionary or other object. No data validation is performed.


    Args:
        path (str): Path to the YAML file.


    Returns:
        Any: Content of the file, typically as a dictionary.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_ae_config(path: str) -> AEConfig:
    """
    Loads and validates the autoencoder configuration file.


    Args:
        path (str): Path to the YAML file.


    Returns:
        AEConfig: Validated configuration object.
    
    Raises:
        ValidationError: If data in the file does not match the model.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        return AEConfig(**raw_config)
    except ValidationError as e:
        logger.error(f"Validation error in ae_config.yaml:\n{e}")
        raise


def load_logreg_config(path: str) -> ModellingConfig:
    """
    Loads and validates the logistic regression configuration file.


    Args:
        path (str): Path to the YAML file.


    Returns:
        ModellingConfig: Validated configuration object.
    
    Raises:
        ValidationError: If data in the file does not match the model.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        return ModellingConfig(**raw_config)
    except ValidationError as e:
        logger.error(f"Validation error in logreg_config.yaml:\n{e}")
        raise


def plot_losses(
    train_losses: List[float], val_losses: List[float], config: BaseModel, path: str
):
    """
    Plots and saves training and validation losses.


    Args:
        train_losses (List[float]): List of loss values on the training set.
        val_losses (List[float]): List of loss values on the validation set.
        config (BaseModel): Configuration object for color and style settings.
        path (str): Path to save the plot.
    """
    color_train = config.output.train_color
    color_val = config.output.val_color
    grayscale = config.output.grayscale


    if grayscale:
        color_train = "black"
        color_val = "dimgray"


    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linestyle='-', color=color_train)
    plt.plot(
        val_losses, label='Validation Loss', linestyle='--', color=color_val
    )
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value (MSE)')


    if len(train_losses) <= 20:
        plt.xticks(range(1, len(train_losses) + 1))
    else:
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


    plt.legend()
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    logger.info(f"Loss plot saved to: {path}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: str,
) -> Tuple[List[float], List[float]]:
    """
    Trains the autoencoder model.


    Args:
        model (nn.Module): Autoencoder model.
        train_loader (DataLoader): Data loader for the training set.
        val_loader (DataLoader): Data loader for the validation set.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        num_epochs (int): Number of epochs for training.


    Returns:
        tuple: A tuple containing lists of training and validation losses.
    """
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for data in tqdm(train_loader,
                         desc=f"Epoch {epoch + 1}/{num_epochs} [Training]"):
            data = data.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)


        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for data in tqdm(val_loader,
                             desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]"):
                data = data.to(device)
                reconstructed, _ = model(data)
                val_loss = criterion(reconstructed, data)
                val_running_loss += val_loss.item()
        val_loss = val_running_loss / len(val_loader)
        val_losses.append(val_loss)


        logger.info(
            f"Epoch {epoch + 1}: Training Loss = {train_loss:.6f}, "
            f"Validation Loss = {val_loss:.6f}"
        )


    return train_losses, val_losses


def creat_sharp_df(years: list[int], sharps_dir: str, event_history_path: str) -> pd.DataFrame:
    """works only with Sharps csv files"""


    dfs = []
    for y in years:
        sharp_csv = os.path.join(sharps_dir, f'{y}Sharp.csv')
        dfs.append(pd.read_csv(sharp_csv))
    df = pd.concat(dfs)[['ratan_filename', 'NOAA_num','harp','T_REC', 'CRVAL1', 'CRLN_OBS', 'USFLUX', 'MEANGBT', 'MEANJZH',
        'MEANPOT', 'SHRGT45', 'TOTUSJH', 'MEANGBH', 'MEANALP', 'MEANGAM',
        'MEANGBZ', 'MEANJZD', 'TOTUSJZ', 'SAVNCPP', 'TOTPOT', 'MEANSHR',
        'AREA_ACR', 'R_VALUE', 'ABSNJZH']]
    df_events = pd.read_csv(event_history_path)
    data_df = df[~df['T_REC'].isna()].join(df_events.set_index('key'), on='ratan_filename')
    data_df["T_REC"] = data_df["T_REC"].str.replace("_TAI", "", regex=False)
    data_df["T_REC"] = pd.to_datetime(data_df ["T_REC"], format="%Y.%m.%d_%H:%M:%S")
    data_df = data_df.dropna().sort_values("T_REC").reset_index(drop=True)
    
    logger.info(f"SHARP df for years {years} is {data_df.shape}")


    return data_df


def creat_ratan_embeddings_df(ratan_embeddings_path: str, sync_df_path: str):
    sync_df = pd.read_csv(sync_df_path)
    all_embeddings_df = pd.read_csv(ratan_embeddings_path)


    data_df = all_embeddings_df.copy()


    data_df['filepath'] = data_df['filepath'].apply(lambda x: x.split('/')[-1])


    logger.info(f"embeddings full shape: {all_embeddings_df.shape}")


    data_df = data_df[data_df['filepath'].isin(sync_df['ratan_filename'])]


    data_df = pd.merge(data_df, sync_df[['ratan_filename', 'day', 'day after']],
                    left_on='filepath', right_on='ratan_filename',
                    how='left')
    data_df = data_df.drop(columns=['filepath'])


    logger.info(f"ratan embeddings data shape for : {data_df.shape}")


    return data_df


def parse_spectrum_filename(filename):
    """
    Parses the spectrum filename to extract date, time, and active region number.
    Example filename: 20110103_075046_AR1140_20.0.fits
    """
    import re
    from datetime import datetime


    match = re.match(r"(\d{8})_(\d{6})_AR(\d+)_.*\.fits", filename)
    if not match:
        raise ValueError(f"Wrong file name format: {filename}")
    
    date_str, time_str, ar_str = match.groups()
    
    spectrum_datetime_str = date_str + time_str
    spectrum_dt = datetime.strptime(spectrum_datetime_str, "%Y%m%d%H%M%S")
    ar_number = int('1'+ar_str)
    
    return spectrum_dt, ar_number


def num_flare_class(flares: List[str], f_cl ='M'):
    """
    If there is C and M or X
    Transform list of flares ['C1.1', 'M1.0', 'X2.1'] → [1, 1]
    """
    if isinstance(flares, str):
        flares = ast.literal_eval(flares)


    return 1*(sum(f_cl in str(event) for event in flares)>0)


def create_target_cols(data_df: pd.DataFrame) -> pd.DataFrame:
    data_df = data_df.copy()
    data_df['is_C_24'] = data_df['day'].map(lambda x: num_flare_class(x, f_cl ='C'))
    data_df['is_M_24'] = data_df['day'].map(lambda x: num_flare_class(x, f_cl ='M'))
    data_df['is_X_24'] = data_df['day'].map(lambda x: num_flare_class(x, f_cl ='X'))
    data_df['is_C_48'] = data_df['day after'].map(lambda x: num_flare_class(x, f_cl ='C'))
    data_df['is_M_48'] = data_df['day after'].map(lambda x: num_flare_class(x, f_cl ='M'))
    data_df['is_X_48'] = data_df['day after'].map(lambda x: num_flare_class(x, f_cl ='X'))
    data_df['Mplus_24'] =  1.*(data_df['is_M_24']+data_df['is_X_24']>0)
    data_df['Mplus_48'] =  1.*(data_df['is_M_48']+data_df['is_X_48']>0)


    return data_df


def compute_stats(TP, TN, FP, FN):
    TSS = (TP/(TP+FN))-(FP/(FP+TN))
    POD = TP/(TP+FN)
    if (FP+TP)==0:
        FAR = 0
    else:
        FAR = FP/(FP+TP)


    return TSS, POD, FAR


def get_stat_by_df(data_dataframe: pd.DataFrame, 
                   y_pred_colname: str, 
                   y_true_colname: str, 
                   thres=0.4):
    
    tn, fp, fn, tp = confusion_matrix(data_dataframe[y_true_colname], 
                                      1.*(data_dataframe[y_pred_colname]>thres)).ravel()
    TSS, POD, FAR = compute_stats(tp, tn, fp, fn)


    return {'TSS': TSS, 'POD': POD, 'FAR': FAR}



def get_brier_score(data_dataframe: pd.DataFrame, 
                    y_pred_colname: str, 
                    y_true_colname: str):


    return brier_score_loss(data_dataframe[y_true_colname], data_dataframe[y_pred_colname])



def get_metrics_for_all_thresholds(thresholds: List[int], 
                                   data_dataframe: pd.DataFrame, 
                                   y_pred_colname: str, 
                                   y_true_colname: str,
                                   return_lists=True):
    """
    If return_lists (by default: True) -> return lists for scores 
    in the following order: TSS, FAR, POD


    else -> return a dict of scores where: 
        key: thresh 
        value: a dict of scores returned by get_stat_by_df  
    """
    y_true = data_dataframe[y_true_colname].values
    y_pred = data_dataframe[y_pred_colname].values

    TSS_list, FAR_list, POD_list, HSS_list = [], [], [], []

    for thr in thresholds:
        y_bin = (y_pred >= thr).astype(int)

        TP = np.sum((y_bin == 1) & (y_true == 1))
        TN = np.sum((y_bin == 0) & (y_true == 0))
        FP = np.sum((y_bin == 1) & (y_true == 0))
        FN = np.sum((y_bin == 0) & (y_true == 1))

        pod = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        far = FP / (FP + TP) if (FP + TP) > 0 else 0.0
        tss = pod - (FP / (FP + TN) if (FP + TN) > 0 else 0.0)

        # Heidke Skill Score (binary)
        denom = ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
        if denom > 0:
            hss = 2 * (TP * TN - FP * FN) / denom
        else:
            hss = 0.0

        # Logging per-threshold metrics
        logger.info(
            f"[thr={thr:.2f}] "
            f"POD={pod:.3f}, FAR={far:.3f}, TSS={tss:.3f}, HSS={hss:.3f} "
            f"(TP={TP}, FP={FP}, TN={TN}, FN={FN})"
        )
        POD_list.append(pod)
        FAR_list.append(far)
        TSS_list.append(tss)
        HSS_list.append(hss)

    return TSS_list, FAR_list, POD_list, HSS_list

def draw_stat_plot(thrs, TSS, POD, FAR, HSS, title: str =None, suptitle: str =None, use_colors: bool = False):
    """
    Plots statistical scores over different thresholds.
    """
    if use_colors:
        colors = plt.cm.Set1.colors
        tss_color = colors[0]
        pod_color = colors[1]
        far_color = colors[2]
        hss_color = colors[3]
    else:
        tss_color = pod_color = far_color = hss_color = 'k'

    plt.plot(thrs, POD, color=pod_color, linestyle='-',  marker='o',
             label='Probability of detection (POD)')
    plt.plot(thrs, FAR, color=far_color, linestyle='-.', marker='*',
             label='False alarm ratio (FAR)')
    plt.plot(thrs, TSS, color=tss_color, linestyle='--', marker='s',
             label='True skill statistic (TSS)')
    plt.plot(thrs, HSS, color=hss_color, linestyle=':', marker='^',
             label='Heidke skill score (HSS)')

    max_pod = max(POD)
    max_far = max(FAR)
    max_tss = max(TSS)


    plt.axhline(y=max_pod, color=pod_color, linestyle='dotted', alpha=0.6);
    plt.axhline(y=max_far, color=far_color, linestyle='dotted', alpha=0.6);
    plt.axhline(y=max_tss, color=tss_color, linestyle='dotted', alpha=0.6);


    plt.xlabel('Probability threshold', fontsize=14);
    plt.ylabel('Score', fontsize=14);
    plt.xticks(fontsize=14);
    plt.yticks(fontsize=14);


    plt.grid();
    plt.legend(fontsize=12);
    plt.title(f"{title}", fontsize=14);
    plt.suptitle(f"{suptitle}", fontsize=16);


    plt.ylim(1, 0)
    plt.gca().invert_yaxis()


def plot_for_flare_type(data_dataframe: pd.DataFrame,
                        y_pred_colname: str, 
                        y_true_colname: str, 
                        threholds: List[int],
                        title: str,
                        save_to: str = None,
                        use_colors: bool = False):
    """
    Calculates and plots statistical scores for a specific flare type.


    Parameters
    ----------
    data_dataframe : pd.DataFrame
        DataFrame containing predicted and true values.
    y_pred_colname : str
        Column name for predicted probabilities.
    y_true_colname : str
        Column name for true labels.
    threholds : List[int]
        List of probability thresholds.
    title : str
        Title for the plot.
    save_to : str, optional
        Path to save the plot. If None, the plot is displayed.
    use_colors : bool, optional
        If True, a colorful palette is used. Defaults to False.
    """
    logger.info(
            f"plotting metrices for {title} | across threholds: {threholds}"
        )
    TSS, FAR, POD, HSS  = get_metrics_for_all_thresholds(thresholds=threholds,
                                                    data_dataframe=data_dataframe, 
                                                    y_pred_colname=y_pred_colname, 
                                                    y_true_colname=y_true_colname)

    y_true = data_dataframe[y_true_colname].values
    y_pred = data_dataframe[y_pred_colname].values

    # 1) Brier Score
    BS = brier_score_loss(y_true, y_pred)



    # 2) Bootstrap CI для Brier Score (можно взять параметры из config, если хочешь)
    bs_stats = calculate_bootstrap_metrics(
        y_true=y_true,
        y_pred=y_pred,
        n_iterations=1000,
        ci_width=95
    )
    bs_ci_low, bs_ci_high = bs_stats['brier_ci']
    
    logger.info(
            f"brier_score_loss: {BS} | CI: {bs_ci_low} - {bs_ci_high}"
        )
    
    # 3) Reference climatology and Brier Skill Score
    p_clim = y_true.mean()
    bs_ref = brier_score_loss(y_true, np.full_like(y_true, p_clim, dtype=float))
    BSS = 1.0 - BS / bs_ref

    title_line = (f"Brier Score = {BS:.3f} "
              f"[95% CI: {bs_ci_low:.3f}–{bs_ci_high:.3f}]")
    full_title = f"{title_line}"

    draw_stat_plot(thrs=threholds, 
                    TSS=TSS, 
                    FAR=FAR, 
                    POD=POD,
                    HSS=HSS, 
                    suptitle=title, 
                    title=full_title,
                    use_colors=use_colors)
    if save_to:
        plt.savefig(save_to, dpi=600) 
        plt.close() 
    else:
        plt.show()


def save_predictions_csv(
    data_dataframe: pd.DataFrame, predictions_dict: dict, output_path: str
):
    """
    Saves predictions to a CSV file.


    The function creates a DataFrame with true values and predicted
    logits, and then saves it to the specified path.


    Args:
        data_dataframe (pd.DataFrame): Original DataFrame containing data.
        predictions_dict (dict): Dictionary where key is the target variable,
                                 and value is the predicted logits.
        output_path (str): Path for saving the CSV file.
    """
    results_df = pd.DataFrame()


    for target_col, preds_logit in predictions_dict.items():
        results_df[f'logit_{target_col}'] = preds_logit
        results_df[f'real_{target_col}'] = data_dataframe[target_col]


    if 'filename' in data_dataframe.columns:
        results_df['filename'] = data_dataframe['filename']
    elif 'ratan_filename' in data_dataframe.columns:
        results_df['ratan_filename'] = data_dataframe['ratan_filename']
        
    results_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


def gather_fits_files(base_data_dir: str, target_years: list = None, default_year_subdir: str = "2025") -> list:
    """
    Gathers all .fits file paths from specified year subdirectories.
    """
    fits_files = []
    
    if not target_years:
        years_to_scan = [default_year_subdir]
    else:
        years_to_scan = target_years


    for year in years_to_scan:
        year_dir = os.path.join(base_data_dir, year)
        if os.path.isdir(year_dir):
            logger.info(f"Scanning for FITS files in: {year_dir}")
            for filename in sorted(os.listdir(year_dir)):
                if filename.lower().endswith((".fits", ".fit")):
                    fits_files.append(os.path.join(year_dir, filename))
        else:
            logger.warning(f"Directory not found - {year_dir}")
            
    if not fits_files:
        logger.warning(f"No FITS files found in specified directories: {(', '.join(years_to_scan)) if years_to_scan else 'default'}")
    else:
        logger.info(f"Found {len(fits_files)} FITS files.")
    return fits_files


def get_classifier_from_config(model_config):
    """
    Creates a classifier object based on the configuration.
    """
    model_name = model_config.name
    model_params = model_config.params
    
    if hasattr(model_params, 'to_dict'):
        model_params_dict = model_params.to_dict()
    else:
        model_params_dict = {
            'class_weight': model_params.class_weight,
            'random_state': model_params.random_state,
        }


    if model_name == "LogisticRegression":
        return LogisticRegression(**model_params_dict)
    else:
        raise ValueError(f"Unsupported classifier: {model_name}")


def calculate_bootstrap_metrics(y_true, y_pred, n_iterations=1000, ci_width=95, random_state=42):
    """
    Computes metrics with Confidence Intervals based on config parameters.
    ci_width: Width of the interval (e.g., 95 for 95% CI).
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    
    # Calculate boundaries based on width (e.g., 95 -> 2.5th and 97.5th percentiles)
    lower_p = (100 - ci_width) / 2.0
    upper_p = 100 - lower_p
    
    auc_scores = []
    brier_scores = []
    
    for i in range(n_iterations):
        # Resample with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        
        # Ensure we have both classes in bootstrap sample to avoid ROC errors
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        auc = roc_auc_score(y_true[indices], y_pred[indices])
        brier = brier_score_loss(y_true[indices], y_pred[indices])
        
        auc_scores.append(auc)
        brier_scores.append(brier)
    
    return {
        'auc_mean': np.mean(auc_scores),
        'auc_ci': (np.percentile(auc_scores, lower_p), np.percentile(auc_scores, upper_p)),
        'brier_mean': np.mean(brier_scores),
        'brier_ci': (np.percentile(brier_scores, lower_p), np.percentile(brier_scores, upper_p))
    }


def plot_roc_auc(pred_df_ratan,
                 pred_df_sharp,
                 target_col,
                 title, save_path, 
                 show_fig=False, use_colors=True,
                 compute_ci=False, n_bootstrap=1000, ci_width=95):
    """
    Plots ROC curves for SHARP vs RATAN with optional Confidence Intervals.
    """
    # Data Extraction
    y_sharp_true = pred_df_sharp[f"real_{target_col}"].values
    y_sharp_pred = pred_df_sharp[f"logit_{target_col}"].values
    y_ratan_true = pred_df_ratan[f"real_{target_col}"].values
    y_ratan_pred = pred_df_ratan[f"logit_{target_col}"].values
    
    # --- 1. Calculate "Real" AUC (The one that matches the line) ---
    sharp_auc_real = roc_auc_score(y_sharp_true, y_sharp_pred)
    ratan_auc_real = roc_auc_score(y_ratan_true, y_ratan_pred)

    # --- 2. Calculate Statistics (Bootstrap) ---
    if compute_ci:
        logger.info(f"Calculating {ci_width}% CI for {title} ({n_bootstrap} iterations)...")
        
        sharp_stats = calculate_bootstrap_metrics(y_sharp_true, y_sharp_pred, n_bootstrap, ci_width)
        ratan_stats = calculate_bootstrap_metrics(y_ratan_true, y_ratan_pred, n_bootstrap, ci_width)
        
        # LABEL FORMAT: Real Score + (CI Range)
        sharp_label = (f'SHARP | AUC = {sharp_auc_real:.3f} '
                       f'[{ci_width}% CI: {sharp_stats["auc_ci"][0]:.3f}-{sharp_stats["auc_ci"][1]:.3f}]')
        ratan_label = (f'RATAN | AUC = {ratan_auc_real:.3f} '
                       f'[{ci_width}% CI: {ratan_stats["auc_ci"][0]:.3f}-{ratan_stats["auc_ci"][1]:.3f}]')
        
        # Log extra metrics
        logger.info(f"[{title}] SHARP Brier: {sharp_stats['brier_mean']:.3f} CI: {sharp_stats['brier_ci']}")
        logger.info(f"[{title}] RATAN Brier: {ratan_stats['brier_mean']:.3f} CI: {ratan_stats['brier_ci']}")
        
    else:
        sharp_label = f'SHARP (AUC = {sharp_auc_real:.3f})'
        ratan_label = f'RATAN (AUC = {ratan_auc_real:.3f})'

    # --- 3. Calculate Curves (Actual Data) ---
    fpr_sharp, tpr_sharp, _ = roc_curve(y_sharp_true, y_sharp_pred)
    fpr_ratan, tpr_ratan, _ = roc_curve(y_ratan_true, y_ratan_pred)

    # --- 4. Setup Styling ---
    if use_colors:
        colors = plt.cm.Set2.colors
        sharp_color = colors[0]
        ratan_color = colors[1]
    else:
        sharp_color = 'black'
        ratan_color = 'black'

    # --- 5. Plotting ---
    plt.figure(figsize=(6, 6))
    plt.plot(fpr_sharp, tpr_sharp, color=sharp_color, linestyle='-', lw=2, label=sharp_label)
    plt.plot(fpr_ratan, tpr_ratan, color=ratan_color, linestyle='--', lw=2, label=ratan_label)
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1)
    
    plt.title(f'ROC Curve for: {title}', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=600)
    
    if show_fig:
        plt.show()
    else:
        plt.close()
        
    logger.info(f"ROC curve for {title} saved to {save_path}")


def extract_azimuth_from_filename(filename: str) -> float:
    """
    Extracts azimuth from the filename (before .fits).
    Example: "..._12.0.fits" -> 12.0
    """
    base = os.path.basename(filename)
    name = base.replace(".fits", "")
    az_str = name.split("_")[-1]
    return float(az_str)


def denoise_by_azimuth(values, azimuths):
    """Removes the azimuthal trend using a linear model."""
    df = pd.DataFrame({"value": values, "az": azimuths})
    X = pd.get_dummies(df["az"], drop_first=True)
    model = LinearRegression(fit_intercept=True)
    model.fit(X, df["value"])
    return df["value"] - model.predict(X)

def denoise_maps_by_azimuth(all_maps: np.ndarray, azimuths: np.ndarray) -> np.ndarray:
    """
    Remove an azimuth-dependent offset from a stack of 2D maps.

    Parameters
    ----------
    all_maps : np.ndarray
        Array of shape (N, H, W) containing N 2D maps (e.g. Stokes I or V).
    azimuths : np.ndarray
        Array of shape (N,) with the telescope azimuth value corresponding to each map.

    Returns
    -------
    np.ndarray
        Array of shape (N, H, W) with an estimated azimuth-dependent mean offset removed
        from each map, while preserving the spatial and spectral structure.
    """
    if all_maps.ndim != 3:
        raise ValueError("all_maps must have shape (N, H, W)")
    if azimuths.shape[0] != all_maps.shape[0]:
        raise ValueError("azimuths length must match number of maps")

    means = all_maps.mean(axis=(1, 2))

    df = pd.DataFrame({"value": means, "az": azimuths})
    X = pd.get_dummies(df["az"], drop_first=True)
    model = LinearRegression(fit_intercept=True)
    model.fit(X, df["value"])
    pred = model.predict(X)

    offsets = pred - pred.mean()

    corrected = all_maps - offsets[:, None, None]
    return corrected


def sqrt_formatter(x, pos):
    """
    Reverses the square root transformation for axis labels.
    """
    return f'{x**2:.0e}'


def log_transform(x, epsilon=1e-12):
    """
    Applies logarithmic transformation, protecting against zero values.
    """
    return np.log10(x + epsilon)


def log_formatter(x, pos):
    """
    Returns the original value using the inverse logarithmic transformation.
    """
    original_value = 10**x
    return f'{original_value:.0e}'


def _normalize_ar_num(ar_num: str) -> str:
    if not re.fullmatch(r"\d{4,5}", ar_num):
        raise ValueError("ar_num must consist of 4 or 5 digits")
    if len(ar_num) == 5:
        if not ar_num.startswith("1"):
            raise ValueError("If ar_num has 5 digits, the first must be '1'")
        ar_num = ar_num[1:]
    return f"_AR{ar_num}_"


def _normalize_azimuth(azimuth: float) -> str:
    azimuth = float(azimuth)
    if not (-30 <= azimuth <= 30):
        raise ValueError("azimuth must be in range [-30, 30]")
    return f"_{azimuth:.1f}.fits"


def filter_by_ar_num(files, ar_num):
    ar_num = _normalize_ar_num(ar_num=ar_num)
    return [path for path in files if ar_num in path.split("/")[-1]]


def filter_by_azimuth(files, azimuth):
    azimuth = _normalize_azimuth(azimuth=azimuth)
    return [path for path in files if azimuth in path.split("/")[-1]] 


def sort_path_by_date(files):
    return sorted(files, key=lambda x: os.path.basename(x))


def thin_out_labels(all_labels, max_visible_labels=40):
    """
    Reduces the number of visible labels by replacing some with empty strings.


    Args:
        all_labels (list): The complete list of string labels.
        max_visible_labels (int): The maximum number of labels you want to see.


    Returns:
        list: A new list of labels with some "muted" (set to '').
    """
    n = len(all_labels)
    if n <= max_visible_labels:
        return all_labels # No need to thin out if already sparse


    # Calculate the step: show one label for every 'step' labels
    step = round(n / max_visible_labels)
    
    # Create the new list: keep a label if its index is a multiple of the step
    thinned_labels = [label if i % step == 0 else '' for i, label in enumerate(all_labels)]
    
    return thinned_labels


def plot_evolution_curves(files, ar_number: str, flare_counts: dict, bars_width=0.25, 
                          channel: int = 0, azimuths: list[str] = None, freq_indices: list = None, 
                          cmap_grayscale: bool = False, denoise: bool = False):
    """
    Plot evolutionary curves for amplitude and area over time for selected frequencies.
    
    Parameters
    ----------
    files : list of str
        List of paths to the FITS files.
    ar_number : str
        The active region number to be included in the plot title.
    channel : int, optional
        Channel index (0 - I, 1 - V). Defaults to 0.
    azimuths : list of str, optional
        If provided, files will be filtered by these strings in their names.
    freq_indices : list of int, optional
        List of frequency indices to plot. If None, the user will be prompted.
    cmap_grayscale : bool, optional
        If True, the colormap will be set to grayscale. Defaults to False.
    """
    
    if not files:
        print("No files to process.")
        return
    
    filtered_files = files  # Corrected typo: filetered_files -> filtered_files


    # Filter by azimuth
    if azimuths:
        tmp = []
        for azim in azimuths:
            tmp.extend(filter_by_azimuth(files=files, azimuth=azim))
    
        filtered_files = tmp


    # Sort files
    filtered_files = sort_path_by_date(files=filtered_files)


    if not filtered_files:
        print("No files to process after filtration.")
        return


    # Get frequencies from the first file to display to the user
    try:
        with fits.open(filtered_files[0]) as hdul:
            frequencies = hdul['FREQ'].data
    except Exception as e:
        print(f"Error reading frequencies from the first file: {e}")
        return


    # Determine the minimum number of frequencies across all files
    min_freq_count = len(frequencies)
    for fpath in filtered_files:
        with fits.open(fpath) as hdul:
            if 'FREQ' in hdul:
                current_freq_count = len(hdul['FREQ'].data)
            else:
                current_freq_count = hdul[0].data.shape[1]
            
            if current_freq_count < min_freq_count:
                min_freq_count = current_freq_count


    # Truncate the frequency list to the minimum count
    safe_frequencies = frequencies[:min_freq_count]
    
    # If freq_indices are not provided, prompt the user
    if freq_indices is None:
        print(f"Available frequencies and their indices (total {min_freq_count}):")
        for i, freq in enumerate(safe_frequencies):
            print(f"  Index {i}: {freq:.2f} MHz")
        
        while True:
            try:
                user_input = input(f"Enter frequency indices separated by commas, e.g., '0, 5, 10' (range 0-{min_freq_count-1}): ")
                freq_indices = [int(idx.strip()) for idx in user_input.split(',')]
                
                if all(0 <= idx < min_freq_count for idx in freq_indices):
                    break
                else:
                    print(f"One or more indices are outside the valid range (0-{min_freq_count-1}). Please try again.")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas.")
    
    timestamps = []
    amplitudes_by_freq = {idx: [] for idx in freq_indices}
    areas_by_freq = {idx: [] for idx in freq_indices}


    for fpath in filtered_files:
        fname = os.path.basename(fpath)
        timestamp = fname.split("_AR")[0]
        timestamps.append(timestamp)


        with fits.open(fpath) as hdul:
            data = hdul[0].data
        
        for idx in freq_indices:
            freq_data = data[channel, idx, :]


            amplitude = np.max(freq_data)
            area = np.sum(freq_data)
            
            amplitudes_by_freq[idx].append(amplitude)
            areas_by_freq[idx].append(area)
    if denoise:
        azimuths_numeric = [extract_azimuth_from_filename(f) for f in filtered_files]
        for idx in freq_indices:
            amplitudes_by_freq[idx] = denoise_by_azimuth(amplitudes_by_freq[idx], azimuths_numeric)
            areas_by_freq[idx] = denoise_by_azimuth(areas_by_freq[idx], azimuths_numeric)


    # Determine channel name for title
    channel_name = 'I' if channel == 0 else 'V'
    dates = [datetime.strptime(ts, '%Y%m%d_%H%M%S') for ts in timestamps]


    # --- Start of the section for formatting tick labels ---
    formatted_labels = [dt.strftime('%b %Y %d %H:%M') for dt in dates]


    # Main labels (day + hour)
    improved_labels = [dt.strftime('%d %H:%M') for dt in dates]


    # Find month boundaries
    change_indices = [0]
    for i in range(1, len(dates)):
        if (dates[i].month != dates[i-1].month) or (dates[i].year != dates[i-1].year):
            change_indices.append(i)


    # Determine the center of each month for label placement
    month_labels_pos = []
    month_labels_text = []
    for i in range(len(change_indices)):
        start_idx = change_indices[i]
        end_idx = change_indices[i+1] if i + 1 < len(change_indices) else len(dates)
        center_idx = (start_idx + end_idx) // 2
        month_labels_pos.append(center_idx)
        month_labels_text.append(dates[center_idx].strftime('%B %Y'))
        # --- End of the section for formatting tick labels ---
    # Common flare plotting setup
    all_flare_types = ['C', 'M', 'X']


    # Detect which flare types are actually present in your flare_counts
    flare_types = [
        ftype for ftype in all_flare_types
        if any(ftype in flare_counts[ts] for ts in flare_counts)
    ]


    # If nothing is found, fallback to all_flare_types (or skip plotting)
    if not flare_types:
        print("No flare data found, skipping flare bars...")
        flare_types = []



    # Convert to datetime for closest matching
    timestamps_dt = [datetime.strptime(ts, '%Y%m%d_%H%M%S') for ts in timestamps]
    flare_times_dt = [datetime.strptime(ts, '%b %Y %d %H:%M') for ts in flare_counts.keys()]


    # Find closest x indices
    x_flares = [np.argmin([abs(ft - ts) for ts in timestamps_dt]) for ft in flare_times_dt]


    # Build counts_matrix (using raw values)
    if flare_types:
        counts_matrix = np.array([
            [flare_counts[ts].get(ftype, 0) for ftype in flare_types]
            for ts in flare_counts.keys()
        ])
    else:
        counts_matrix = np.zeros((len(flare_counts), 0)) 


    # Optional: Scale up the matrix for better visibility (e.g., if fluxes are tiny)
    # counts_matrix *= 1e6  # Uncomment to make bars taller


    width = bars_width


    # Get a list of colors, or a colormap
    if cmap_grayscale:
        # Create a linear grayscale colormap for the lines
        cmap_lines = plt.get_cmap('gray')
        colors_lines = cmap_lines(np.linspace(0.1, 0.9, len(freq_indices)))
        # Create a linear grayscale colormap for the bars
        cmap_bars = plt.get_cmap('Greys')
        colors_bars = cmap_bars(np.linspace(0.4, 0.8, len(flare_types)))
    else:
        # Use a default Matplotlib colormap for color
        cmap_lines = plt.get_cmap('viridis')
        colors_lines = cmap_lines(np.linspace(0.1, 0.9, len(freq_indices)))
        # Use a default colormap for bars
        cmap_bars = plt.get_cmap('plasma')
        colors_bars = cmap_bars(np.linspace(0.1, 0.9, len(flare_types)))


    # -------------------------------------------------------------------------
    # Amplitude Plot
    # -------------------------------------------------------------------------
    x = np.arange(len(timestamps))
    fig_amp, ax = plt.subplots(figsize=(16, 6))


    # Plot amplitude lines on primary axis
    for i, idx in enumerate(freq_indices):
        amps = amplitudes_by_freq[idx]
        freq_value = safe_frequencies[idx]
        ax.plot(x, amps, '-o', label=f'{freq_value:.2f} MHz', markersize=2, color=colors_lines[i])


    ax.set_xlabel(f'{month_labels_text[0]}\n\nTime', fontsize=15)
    ax.set_ylabel('Amplitude (Maximum)', fontsize=15)
    ax.set_title(f'AR {ar_number} | Channel {channel_name} | Amplitude Evolution with Flares', fontsize=18)


    # Create secondary y-axis for flares
    ax2 = ax.twinx()
    ax2.set_ylabel('Flare Flux (W/m²)', fontsize=15)  # Adjust label based on your data
    ax2.set_yscale("log")


    for i, ft in enumerate(flare_types):
        scaled_vals = counts_matrix[:, i]
        ax2.bar(
            np.array(x_flares) + i * width + 0.5,
            scaled_vals,
            width=width,
            alpha=0.5,
            label=f'Flare {ft}',
            color=colors_bars[i]
        )
    # X-axis labels (thinned out for readability)
    ax.set_xticks(x + width * (len(flare_types) / 2))  # Center ticks under bar groups
    ax.set_xticklabels(thin_out_labels(improved_labels), rotation=45, ha='right')


    # Add the central month labels
    ax.set_xticks(month_labels_pos, minor=True)
    ax.set_xticklabels(month_labels_text, rotation=0, ha='center', minor=True)
    ax.tick_params(axis='x', which='minor', pad=47) # Push minor ticks down


    # Legends for both axes
    ax.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    ax2.set_ylabel('Flare Flux (scaled W/m²)', fontsize=15)
    #ax2.yaxis.set_major_formatter(FuncFormatter(log_formatter))
    max_val = np.max(scaled_vals[np.isfinite(scaled_vals)])
    min_val = np.min(scaled_vals[np.isfinite(scaled_vals)])
    ax2.set_ylim(min_val - 1, max_val + 1)
    # Set tick label font size for both axes
    # --- Set all tick font sizes to 14 ---
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='x', which='minor', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax2.tick_params(axis='y', labelsize=13)




    plt.tight_layout()
    plt.show()


    # -------------------------------------------------------------------------
    # Area Plot (Separate Figure with Same Bars)
    # -------------------------------------------------------------------------
    fig_area, ax = plt.subplots(figsize=(16, 6))


    # Plot area lines on primary axis
    for i, idx in enumerate(freq_indices):
        areas = areas_by_freq[idx]
        freq_value = safe_frequencies[idx]
        ax.plot(x, areas, '-o', label=f'{freq_value:.2f} MHz', markersize=2, color=colors_lines[i])
    
    ax.set_xlabel(f'{month_labels_text[0]}\n\nTime', fontsize=15)
    ax.set_ylabel('Area (Sum)', fontsize=15)
    ax.set_title(f'AR {ar_number} | Channel {channel_name} | Area Evolution with Flares', fontsize=18)


    # Create secondary y-axis for flares (same as amplitude plot)
    ax2 = ax.twinx()
    ax2.set_ylabel('Flare Flux (W/m²)', fontsize=15)  
    ax2.set_yscale("log")


    for i, ft in enumerate(flare_types):
        scaled_vals = counts_matrix[:, i]
        #scaled_vals[np.isinf(scaled_vals)] = -6
        ax2.bar(
            np.array(x_flares) + i * width + 0.5,
            scaled_vals,
            width=width,
            alpha=0.5,
            label=f'Flare {ft}',
            color=colors_bars[i]
        )
    # X-axis labels (thinned out for readability)
    ax.set_xticks(x + width * (len(flare_types) / 2))  # Center ticks under bar groups
    ax.set_xticklabels(thin_out_labels(improved_labels), rotation=45, ha='right')
    
    # Add the central month labels
    ax.set_xticks(month_labels_pos, minor=True)
    ax.set_xticklabels(month_labels_text, rotation=0, ha='center', minor=True)
    ax.tick_params(axis='x', which='minor', pad=15) # Push minor ticks down


    # Legends for both axes 
    ax.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    #ax2.yaxis.set_major_formatter(FuncFormatter(log_formatter))
    max_val = np.max(scaled_vals[np.isfinite(scaled_vals)])
    min_val = np.min(scaled_vals[np.isfinite(scaled_vals)])
    ax2.set_ylim(min_val - 1, max_val + 1)


    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='x', which='minor', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax2.tick_params(axis='y', labelsize=13)



    plt.tight_layout()
    plt.show()



def round_sig(x, sig=2):
    return float(f"{x:.{sig}g}")


def preprocess_flare_data(flare_data_raw):
    """
    Aggregates flare data by day. For each day, takes the maximum flux
    for each flare category ('C', 'M', 'X') in absolute units W/m².
    
    Args:
        flare_data_raw (dict): Source dictionary where keys are filenames.
    
    Returns:
        dict: Keys are date/time strings, values are a dictionary with maximum flux
              by flare class.
              Example: {'Oct 2014 21 08:58': {'C': 9.4e-6, 'M': 2.2e-5, 'X': 2.2e-4}}
    """
    
    flare_scale = {'C': 1e-6, 'M': 1e-5, 'X': 1e-4}  # absolute flux W/m²
    
    processed_flares = {}
    
    for filename, flare_list in flare_data_raw.items():
        # Get timestamp
        timestamp_str = filename.split("_AR")[0]
        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        ts_str = timestamp.strftime('%b %Y %d %H:%M')
        
        max_flux = {}
        for f in flare_list.get('day', []):
            cls = f[0]  # C, M, X
            try:
                val = float(f[1:])
            except ValueError:
                continue
            abs_flux = round_sig(val * flare_scale[cls], 2)
            
            if cls not in max_flux or abs_flux > max_flux[cls]:
                max_flux[cls] = abs_flux
        
        if max_flux:  # if there is at least one flare
            processed_flares[ts_str] = max_flux
    
    return processed_flares


def reduce_flares_to_daily_representative(processed_flares):
    """
    Accepts a dictionary {str: {'C':..}} and keeps only one representative
    entry for each day, preserving the original strings.
    """
    grouped_by_day = defaultdict(list)


    # 1. Convert string back to datetime for grouping
    for ts_str, flare_counts in processed_flares.items():
        ts_dt = datetime.strptime(ts_str, '%b %Y %d %H:%M')
        grouped_by_day[ts_dt.date()].append((ts_dt, ts_str, flare_counts))


    final_reduced_flares = {}


    # 2. For each group, choose the entry closest to the mean time
    for day_key, entries in grouped_by_day.items():
        if len(entries) == 1:
            _, chosen_str, chosen_counts = entries[0]
        else:
            times_in_seconds = [e[0].hour*3600 + e[0].minute*60 + e[0].second for e in entries]
            mean_time = np.mean(times_in_seconds)
            closest_index = np.argmin([abs(t - mean_time) for t in times_in_seconds])
            _, chosen_str, chosen_counts = entries[closest_index]


        final_reduced_flares[chosen_str] = chosen_counts


    return final_reduced_flares



def create_flares_raw_dict(event_csv_path:str, ar_paths_dict: dict, column_name: str = "day"):
    events = pd.read_csv(event_csv_path)
    events_dict = {}
    for ar in ar_paths_dict:
        ar_name = ar.split("/")[-1]


        # take the string that looks like a list and turn it into a real list
        day_str = events.loc[events["key"] == ar_name, "day"].tolist()[0]
        day_lists = ast.literal_eval(day_str)


        days_after_str = events.loc[events["key"] == ar_name, column_name].tolist()[0]
        days_after_lists = ast.literal_eval(days_after_str)


        events_dict[ar_name] = {
            "day": day_lists,
            "days after": days_after_lists
        }
    return events_dict



def clipped_zoom(img, zoom_factor, **kwargs):


    h, w = img.shape[:2]


    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)


    # Zooming out
    if zoom_factor < 1:


        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2


        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = ndimage.zoom(img, zoom_tuple, **kwargs)


    # Zooming in
    elif zoom_factor > 1:


        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2


        out = ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)


        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out


def count_freqs_per_fit(
    fits_paths: list[str],
    ar_num: str = None,
    show_summary: bool = True,
    return_paths: bool = True
):
    tmp = {}
    for fpath in fits_paths:
        with fits.open(fpath) as hdul:
            num_freqs = len(hdul['FREQ'].data)
            tmp[fpath] = num_freqs

    # sort items by frequency count
    tmp_sorted = dict(sorted(tmp.items(), key=lambda item: item[1]))

    # print summary
    if show_summary:
        if ar_num is not None:
            print(f"{ar_num}:")
        for fpath, count in tmp_sorted:
            print(f"  {fpath} -> {count}")

        print(f"Total files: {len(tmp_sorted)}")
        print(f"Min freqs: {min(tmp.values())}, Max freqs: {max(tmp.values())}")

    if return_paths:
        return tmp_sorted