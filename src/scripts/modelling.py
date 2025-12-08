import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from src import utils
from src.logger import get_logger
from src.constants import (RAW_DATA_DIR, EXTERNAL_DATA_DIR,
                           INTERIM_DATA_DIR, PROCESSED_DATA_DIR, PLOTS_DIR)

logger = get_logger(__file__)


def prepare_data(config):
    """
    Creates and caches dataframes for the logistic regression pipeline.

    This function checks for the existence of processed SHARP and RATAN dataframes.
    If files are not found, it creates them from raw data, adds target columns,
    and saves them to the 'interim' directory for future use.

    Args:
        config (dict): Dictionary with all pipeline settings loaded from the YAML file.
    """
    logger.info("Preparing and caching data...")

    sharp_train_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.sharps_train_save)
    sharp_test_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.sharps_test_save)
    ratan_train_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.ratan_train_save)
    ratan_test_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.ratan_test_save)

    if not os.path.exists(sharp_train_save_path) or not os.path.exists(sharp_test_save_path):
        logger.info("SHARP dataframes not found. Creating and saving...")
        sharp_train_df = utils.creat_sharp_df(
            config.data.sharp_years_train,
            sharps_dir=os.path.join(RAW_DATA_DIR, "sharps"),
            event_history_path=os.path.join(RAW_DATA_DIR, "events", "events_history.csv")
        )
        sharp_test_df = utils.creat_sharp_df(
            config.data.sharp_years_test,
            sharps_dir=os.path.join(RAW_DATA_DIR, "sharps"),
            event_history_path=os.path.join(RAW_DATA_DIR, "events", "events_history.csv")
        )
        
        sharp_train_df = utils.create_target_cols(sharp_train_df)
        sharp_test_df = utils.create_target_cols(sharp_test_df)
        
        sharp_train_df.to_csv(sharp_train_save_path, index=False)
        sharp_test_df.to_csv(sharp_test_save_path, index=False)
        logger.info("SHARP dataframes saved.")
    else:
        logger.info("SHARP dataframes found. Skipping step.")
    
    if not os.path.exists(ratan_train_save_path) or not os.path.exists(ratan_test_save_path):
        logger.info("RATAN dataframes not found. Creating and saving...")
        ratan_train_df = utils.creat_ratan_embeddings_df(
            ratan_embeddings_path=os.path.join(PROCESSED_DATA_DIR, config.data.data_paths.ratan_embeddings),
            sync_df_path=os.path.join(EXTERNAL_DATA_DIR, "sync_train.csv")
        )
        ratan_test_df = utils.creat_ratan_embeddings_df(
            ratan_embeddings_path=os.path.join(PROCESSED_DATA_DIR, config.data.data_paths.ratan_embeddings),
            sync_df_path=os.path.join(EXTERNAL_DATA_DIR, "sync_test.csv")
        )
        ratan_train_df = utils.create_target_cols(ratan_train_df)
        ratan_test_df = utils.create_target_cols(ratan_test_df)
        
        ratan_train_df.to_csv(ratan_train_save_path, index=False)
        ratan_test_df.to_csv(ratan_test_save_path, index=False)
        logger.info("RATAN dataframes saved.")
    else:
        logger.info("RATAN dataframes found. Skipping step.")
    
    


def train_and_predict(config):
    """
    Trains a logistic regression model and makes predictions.

    The function first checks if prediction files exist. If so, the training step
    is skipped. Otherwise, it loads prepared data, trains the model with
    TimeSeriesSplit cross-validation, makes final predictions on the test set,
    and saves results to CSV files.

    Args:
        config (dict): Dictionary with all pipeline settings.
    """
    logger.info("Training model and predicting...")

    sharp_train_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.sharps_train_save)
    sharp_test_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.sharps_test_save)
    ratan_train_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.ratan_train_save)
    ratan_test_save_path = os.path.join(INTERIM_DATA_DIR, config.data.data_paths.ratan_test_save)
    output_csv_sharp = os.path.join(PROCESSED_DATA_DIR, config.output.prediction_csv_sharp)
    output_csv_ratan = os.path.join(PROCESSED_DATA_DIR, config.output.prediction_csv_ratan)
        
    sharp_train_df = pd.read_csv(sharp_train_save_path)
    sharp_test_df = pd.read_csv(sharp_test_save_path)
    ratan_train_df = pd.read_csv(ratan_train_save_path)
    ratan_test_df = pd.read_csv(ratan_test_save_path)


    logger.info(
        f"[SHARP] train: {len(sharp_train_df)} samples, "
        f"{sharp_train_df['AR_NUM'].nunique()} unique ARs"
        if 'AR_NUM' in sharp_train_df.columns else
        f"[SHARP] train: {len(sharp_train_df)} samples"
    )
    logger.info(
        f"[SHARP] test:  {len(sharp_test_df)} samples, "
        f"{sharp_test_df['AR_NUM'].nunique()} unique ARs"
        if 'AR_NUM' in sharp_test_df.columns else
        f"[SHARP] test:  {len(sharp_test_df)} samples"
    )
    logger.info(
        f"[RATAN] train: {len(ratan_train_df)} samples, "
        f"{ratan_train_df['AR_NUM'].nunique()} unique ARs"
        if 'AR_NUM' in ratan_train_df.columns else
        f"[RATAN] train: {len(ratan_train_df)} samples"
    )
    logger.info(
        f"[RATAN] test:  {len(ratan_test_df)} samples, "
        f"{ratan_test_df['AR_NUM'].nunique()} unique ARs"
        if 'AR_NUM' in ratan_test_df.columns else
        f"[RATAN] test:  {len(ratan_test_df)} samples"
    )

    if os.path.exists(output_csv_sharp) and os.path.exists(output_csv_ratan):
        logger.info("Prediction files already exist. Skipping step.")
        return
    
    feature_cols_sharp = config.data.sharp_features
    feature_cols_ratan = config.data.ratan_features
    target_cols = config.data.target_cols
    
    base_clf = utils.get_classifier_from_config(config.model)
    base_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', base_clf),
    ])
    
    tscv = TimeSeriesSplit(n_splits=config.model.cv_splits)
    
    logit_res_sharp = {}
    logit_res_ratan = {}

    for target_col in target_cols:
        logger.info(f'\n==== Processing target: {target_col} ====')
        
        n_train_pos = int(sharp_train_df[target_col].sum())
        n_test_pos  = int(sharp_test_df[target_col].sum())
        logger.info(
            f"[{target_col}] positives: train={n_train_pos}, test={n_test_pos}"
        )
        sharp_cv_scores = []
        for train_idx, val_idx in tscv.split(sharp_train_df):
            X_train, X_val = sharp_train_df.iloc[train_idx][feature_cols_sharp], sharp_train_df.iloc[val_idx][feature_cols_sharp]
            y_train, y_val = sharp_train_df.iloc[train_idx][target_col], sharp_train_df.iloc[val_idx][target_col]
            model = clone(base_pipe)
            model.fit(X_train, y_train)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_val_proba)
            sharp_cv_scores.append(score)
        logger.info(f'[SHARP] CV ROC-AUC scores: {sharp_cv_scores}')
        logger.info(f'[SHARP] Mean: {np.mean(sharp_cv_scores):.3f}, Std Dev: {np.std(sharp_cv_scores):.3f}')
        
        final_model_sharp = clone(base_pipe)
        final_model_sharp.fit(sharp_train_df[feature_cols_sharp], sharp_train_df[target_col])
        y_sharp_proba = final_model_sharp.predict_proba(sharp_test_df[feature_cols_sharp])[:, 1]
        logit_res_sharp[target_col] = y_sharp_proba

        ratan_cv_scores = []
        for train_idx, val_idx in tscv.split(ratan_train_df):
            X_train, X_val = ratan_train_df.iloc[train_idx][feature_cols_ratan], ratan_train_df.iloc[val_idx][feature_cols_ratan]
            y_train, y_val = ratan_train_df.iloc[train_idx][target_col], ratan_train_df.iloc[val_idx][target_col]
            model = clone(base_pipe)
            model.fit(X_train, y_train)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_val_proba)
            ratan_cv_scores.append(score)
        logger.info(f'[RATAN] CV ROC-AUC scores: {ratan_cv_scores}')
        logger.info(f'[RATAN] Mean: {np.mean(ratan_cv_scores):.3f}, Std Dev: {np.std(ratan_cv_scores):.3f}')

        final_model_ratan = clone(base_pipe)
        final_model_ratan.fit(ratan_train_df[feature_cols_ratan], ratan_train_df[target_col])
        y_ratan_proba = final_model_ratan.predict_proba(ratan_test_df[feature_cols_ratan])[:, 1]
        logit_res_ratan[target_col] = y_ratan_proba

    utils.save_predictions_csv(sharp_test_df.copy(), logit_res_sharp, output_csv_sharp)
    utils.save_predictions_csv(ratan_test_df.copy(), logit_res_ratan, output_csv_ratan)
    logger.info("Predictions saved.")

def evaluate_and_visualize(config):
    """
    Loads predictions and creates ROC curve plots and other metrics.

    The function checks if prediction files exist. If so, it loads them,
    calculates metrics, and generates plots for SHARP and RATAN data,
    saving them to the reports directory.
    
    Includes calculation of 95% Confidence Intervals (CI) if enabled in config.

    Args:
        config (dict): Dictionary with all pipeline settings.
    """
    logger.info("Evaluating and visualizing metrics...")

    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    output_csv_sharp = os.path.join(PROCESSED_DATA_DIR, config.output.prediction_csv_sharp)
    output_csv_ratan = os.path.join(PROCESSED_DATA_DIR, config.output.prediction_csv_ratan)
    
    if not os.path.exists(output_csv_sharp) or not os.path.exists(output_csv_ratan):
        logger.error("Prediction files not found. Please run the training step first.")
        return
        
    pred_df_sharp = pd.read_csv(output_csv_sharp)
    pred_df_ratan = pd.read_csv(output_csv_ratan)

    titles_name_dict = config.output.titles
    thresholds = config.output.thresholds

    
    metrics_output_dir = os.path.join(PLOTS_DIR, "Forecasting_Metrics")
    os.makedirs(metrics_output_dir, exist_ok=True)
    

    for key in config.data.target_cols:
                # --- 0. Brier Scores and Brier Skill Scores (RATAN vs SHARP) ---
        y_true_sharp = pred_df_sharp[f"real_{key}"].values
        y_true_ratan = pred_df_ratan[f"real_{key}"].values

        # They should be identical by construction; we use SHARP's as reference
        if not np.array_equal(y_true_sharp, y_true_ratan):
            logger.warning(f"[{key}] Mismatch between SHARP and RATAN y_true; using SHARP labels as reference.")
        y_true = y_true_sharp

        y_sharp_pred = pred_df_sharp[f"logit_{key}"].values
        y_ratan_pred = pred_df_ratan[f"logit_{key}"].values

        bs_sharp = brier_score_loss(y_true, y_sharp_pred)
        bs_ratan = brier_score_loss(y_true, y_ratan_pred)

        # Brier Skill Score of RATAN relative to SHARP
        if bs_sharp > 0:
            bss_ratan_vs_sharp = 1.0 - bs_ratan / bs_sharp
        else:
            bss_ratan_vs_sharp = np.nan

        logger.info(
            f"[{key}] Brier (SHARP) = {bs_sharp:.3f}, "
            f"Brier (RATAN) = {bs_ratan:.3f}, "
            f"BSS(RATAN | SHARP) = {bss_ratan_vs_sharp:.3f}"
        )

# --- 1. Standard Metrics (Forecasting Metrics) ---
        utils.plot_for_flare_type(
            pred_df_sharp,
            y_pred_colname=f"logit_{key}",
            y_true_colname=f"real_{key}",
            threholds=thresholds,
            title=getattr(titles_name_dict, key),
            save_to=os.path.join(metrics_output_dir, f"Sharp_{key}.png"),
            use_colors=config.output.color
        )
        utils.plot_for_flare_type(
            pred_df_ratan,
            y_pred_colname=f"logit_{key}",
            y_true_colname=f"real_{key}",
            threholds=thresholds,
            title=getattr(titles_name_dict, key),
            save_to=os.path.join(metrics_output_dir, f"Ratan_{key}.png"),
            use_colors=config.output.color
        )

    # --- 2. ROC Curves with Confidence Intervals (CI) ---
    roc_output_dir = os.path.join(PLOTS_DIR, "ROC_Curves")
    os.makedirs(roc_output_dir, exist_ok=True)

    
    for target_col in config.data.target_cols:
        # Extract numpy arrays for the specific target
        
        # Resolve title
        plot_title = getattr(titles_name_dict, target_col) if hasattr(titles_name_dict, target_col) else target_col

        # Call the extracted function
        utils.plot_roc_auc(
            pred_df_ratan=pred_df_ratan,
            pred_df_sharp=pred_df_sharp,
            target_col=target_col,
            title=plot_title,
            save_path=os.path.join(roc_output_dir, f"ROC_Curve_{target_col}.png"),
            show_fig=config.output.show_fig,
            use_colors=config.output.color,
            compute_ci=config.metrics.compute_ci,
            n_bootstrap=config.metrics.n_bootstrap,
            ci_width=config.metrics.ci_percentile
        )
