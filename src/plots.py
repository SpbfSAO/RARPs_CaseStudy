import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils import get_metrics_for_all_thresholds, get_brier_score
from src.logger import get_logger


logger = get_logger(__file__)


def plot_metrics(
    thresholds: List[float],
    tss: List[float],
    pod: List[float],
    far: List[float],
    title: Optional[str] = None,
    suptitle: Optional[str] = None,
):
    """
    Plots forecasting quality metrics.


    Args:
        thresholds (List[float]): List of threshold values.
        tss (List[float]): List of TSS values for each threshold.
        pod (List[float]): List of POD values for each threshold.
        far (List[float]): List of FAR values for each threshold.
        title (Optional[str]): Plot title.
        suptitle (Optional[str]): Plot suptitle.
    """
    plt.plot(thresholds, pod, "ko-", label="Probability of discovery")
    plt.plot(thresholds, far, "k*-", label="False alarm ratio")
    plt.plot(thresholds, tss, "k--", label="True skill statistics (TSS)")


    max_pod = max(pod)
    max_far = max(far)
    max_tss = max(tss)


    plt.axhline(y=max_pod, color='k', linestyle='dotted', alpha=0.6)
    plt.axhline(y=max_far, color='k', linestyle='dotted', alpha=0.6)
    plt.axhline(y=max_tss, color='k', linestyle='dotted', alpha=0.6)


    plt.xlabel('Probability threshold')
    plt.ylabel('Score')
    plt.grid()
    plt.legend()
    plt.title(f"{title}")
    plt.suptitle(f"{suptitle}")



def plot_metrics_for_flare_type(
    data_dataframe: pd.DataFrame,
    y_pred_colname: str,
    y_true_colname: str,
    thresholds: List[float],
    title: str,
    save_to: Optional[str] = None,
):
    """
    Plots and saves metrics for a specific flare type.


    Args:
        data_dataframe (pd.DataFrame): DataFrame with data.
        y_pred_colname (str): Name of column with predicted values.
        y_true_colname (str): Name of column with true values.
        thresholds (List[float]): List of threshold values.
        title (str): Title for the plot.
        save_to (Optional[str]): Path to save the file. If None, the plot
                                 will be shown on screen.
    """
    tss, far, pod = get_metrics_for_all_thresholds(
        thresholds=thresholds,
        data_dataframe=data_dataframe,
        y_pred_colname=y_pred_colname,
        y_true_colname=y_true_colname,
    )
    bs = get_brier_score(
        data_dataframe=data_dataframe,
        y_pred_colname=y_pred_colname,
        y_true_colname=y_true_colname,
    )


    plot_metrics(
        thresholds=thresholds,
        tss=tss,
        far=far,
        pod=pod,
        suptitle=title,
        title=f"Brier Score: {round(bs, 4)}",
    )
    if save_to:
        plt.savefig(save_to, dpi=600)
        plt.close()
    else:
        plt.show()



def visualize_samples(
    original_data_np: np.ndarray,
    preprocessed_data_np: np.ndarray,
    reconstructed_data_np: np.ndarray,
    selected_filepaths: list,
    output_dir_base: str,
):
    """
    Visualizes original, preprocessed, and reconstructed images for selected files.
    
    Args:
        original_data_np (np.ndarray): Source data for selected files.
        preprocessed_data_np (np.ndarray): Preprocessed data for selected files.
        reconstructed_data_np (np.ndarray): Reconstructed data for selected files.
        selected_filepaths (list): List of paths to source files for visualization.
        output_dir_base (str): Base directory for saving plots.
    """
    logger.info("Starting visualization of selected samples...")
    output_dir_visual = os.path.join(output_dir_base, "visual_reconstruction")
    os.makedirs(output_dir_visual, exist_ok=True)
    
    data_stages = {
        "Original": original_data_np,
        "Preprocessed": preprocessed_data_np,
        "Reconstructed": reconstructed_data_np,
    }


    for i in range(len(selected_filepaths)):
        filename_base = os.path.basename(selected_filepaths[i]).split(".")[0]
        logger.debug(f"Visualizing for file: {filename_base}, ordinal index: {i}")


        for stage_name, data_array in data_stages.items():
            if data_array is None:
                logger.warning(
                    f"Data for stage '{stage_name}' is missing for index {i}, "
                    f"skipping."
                )
                continue
            
            current_sample_i = data_array[i, 0]
            current_sample_v = data_array[i, 1]


            for cmap_name, suffix in [('viridis', ''), ('gray', '_gray')]:
                fig, axes = plt.subplots(1, 2, figsize=(11, 5))
                fig.suptitle(
                    f'{stage_name} - {filename_base} (I and V channels)',
                    fontsize=14,
                )


                ax = axes[0]
                im = ax.imshow(current_sample_i, cmap=cmap_name, origin='lower', aspect='auto')
                ax.set_title(f'{stage_name} I')
                ax.set_xlabel('Spatial Pixels')
                ax.set_ylabel('Frequency Channels')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


                ax = axes[1]
                im = ax.imshow(current_sample_v, cmap=cmap_name, origin='lower', aspect='auto')
                ax.set_title(f'{stage_name} V')
                ax.set_xlabel('Spatial Pixels')
                ax.set_ylabel('Frequency Channels')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                output_filename = f"{stage_name}_{filename_base}{suffix}.png"
                output_path = os.path.join(output_dir_visual, output_filename)
                plt.savefig(output_path, bbox_inches='tight', dpi=100)
                plt.close(fig)
                logger.debug(f"Saved: {output_path}")
    logger.info("Visualization of selected samples complete.")



def plot_spectral_comparison(
    preprocessed_data: np.ndarray,
    reconstructed_data: np.ndarray,
    selected_filepaths: list,   
    output_dir_base: str,
    target_height: int,
    target_width: int,
):
    """
    Plots spectral profile comparisons for selected files.


    Args:
        preprocessed_data (np.ndarray): Preprocessed data.
        reconstructed_data (np.ndarray): Reconstructed data.
        selected_filepaths (List[str]): List of paths to source files.
        output_dir_base (str): Base directory for saving plots.
        target_height (int): Image height (number of frequency channels).
        target_width (int): Image width (number of spatial pixels).
    """
    logger.info("Starting spectral comparison plotting...")
    output_dir_spectral = os.path.join(
        output_dir_base, "spectral_reconstruction_plots"
    )
    os.makedirs(output_dir_spectral, exist_ok=True)
    central_width_pixel = target_width // 2
    freq_bins = np.arange(target_height)


    for i in range(len(selected_filepaths)):
        filename_base = os.path.basename(selected_filepaths[i]).split(".")[0]
        logger.debug(
            f"Plotting spectral comparison for file: {filename_base}, "
            f"local index: {i}"
        )


        prep_spectrum_i = preprocessed_data[i, 0, :, central_width_pixel]
        recon_spectrum_i = reconstructed_data[i, 0, :, central_width_pixel]
        prep_spectrum_v = preprocessed_data[i, 1, :, central_width_pixel]
        recon_spectrum_v = reconstructed_data[i, 1, :, central_width_pixel]


        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(
            f'Spectral Profile Comparison - {filename_base}\n(Central Spatial Pixel)',
            fontsize=14,
        )


        axes[0].plot(
            freq_bins,
            prep_spectrum_i,
            color='k',
            linestyle='-',
            label='Preprocessed (Input)',
        )
        axes[0].plot(
            freq_bins, recon_spectrum_i, color='r', linestyle='--', label='Reconstructed'
        )
        axes[0].set_title(f'Channel I')
        axes[0].set_xlabel('Frequency channel')
        axes[0].set_ylabel('Normalized Value')
        axes[0].legend()
        axes[0].grid(True)


        axes[1].plot(
            freq_bins,
            prep_spectrum_v,
            color='k',
            linestyle='-',
            label='Preprocessed (Input)',
        )
        axes[1].plot(
            freq_bins, recon_spectrum_v, color='r', linestyle='--', label='Reconstructed'
        )
        axes[1].set_title(f'Channel V')
        axes[1].set_xlabel('Frequency channel')
        axes[1].set_ylabel('Normalized Value')
        axes[1].legend()
        axes[1].grid(True)


        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        output_filename = f"Spectrum_Comparison_{filename_base}.png"
        output_path = os.path.join(output_dir_spectral, output_filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        logger.debug(f"Saved: {output_path}")


        # Grayscale version
        fig_gray, axes_gray = plt.subplots(1, 2, figsize=(16, 6))
        fig_gray.suptitle(
            f'Spectral Profile Comparison (Grayscale) - {filename_base}\n'
            f'(Central Spatial Pixel)',
            fontsize=14,
        )


        axes_gray[0].plot(
            freq_bins,
            prep_spectrum_i,
            color='black',
            linestyle='-',
            label='Preprocessed (Input)',
        )
        axes_gray[0].plot(
            freq_bins,
            recon_spectrum_i,
            color='dimgray',
            linestyle='--',
            label='Reconstructed',
        )
        axes_gray[0].set_title(f'Channel I')
        axes_gray[0].set_xlabel('Frequency Bin (Height Dimension Index)')
        axes_gray[0].set_ylabel('Normalized Value')
        axes_gray[0].legend()
        axes_gray[0].grid(True)


        axes_gray[1].plot(
            freq_bins,
            prep_spectrum_v,
            color='black',
            linestyle='-',
            label='Preprocessed (Input)',
        )
        axes_gray[1].plot(
            freq_bins,
            recon_spectrum_v,
            color='dimgray',
            linestyle='--',
            label='Reconstructed',
        )
        axes_gray[1].set_title(f'Channel V')
        axes_gray[1].set_xlabel('Frequency Bin (Height Dimension Index)')
        axes_gray[1].set_ylabel('Normalized Value')
        axes_gray[1].legend()
        axes_gray[1].grid(True)


        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        output_filename_gray = f"Spectrum_Comparison_{filename_base}_gray.png"
        output_path_gray = os.path.join(output_dir_spectral, output_filename_gray)
        plt.savefig(output_path_gray, bbox_inches='tight', dpi=100)
        plt.close(fig_gray)
        logger.debug(f"Saved: {output_path_gray}")
    logger.info("Spectral comparison plotting complete.")
