import os
import matplotlib.pyplot as plt
from src.logger import get_logger
from src.constants import create_dirs, MODELLING_CFG_PATH, AE_CFG_PATH, MAIN_CFG_PATH

from src.utils import load_config, load_logreg_config, load_ae_config


plt.switch_backend('Agg')


logger = get_logger(__file__)



def logreg_pipe():
    """
    Main execution block for the logistic regression pipeline.


    This function manages the entire pipeline by loading the configuration
    and sequentially calling modular steps: data preparation,
    training/prediction, and evaluation/visualization.
    """
    from src.scripts import modelling as modelling_steps

    logger.info("Starting logistic regression pipeline.")


    # Load configuration
    config = load_logreg_config(MODELLING_CFG_PATH)

    if config.steps.prepare_data:
        # 1. Data preparation
        modelling_steps.prepare_data(config)

    if config.steps.train_and_predict:
        # 2. Training and prediction
        modelling_steps.train_and_predict(config)

    if config.steps.evaluate_and_visualize:
        # 3. Evaluation and visualization
        modelling_steps.evaluate_and_visualize(config)


    logger.info("Logistic regression pipeline completed successfully.")



def ae_pipe():
    """
    Main execution block for the autoencoder pipeline.


    This function manages the entire pipeline by loading the configuration
    and sequentially calling modular steps: data preparation,
    training/loading the model, and embedding extraction/quality evaluation.
    """
    logger.info("Starting autoencoder pipeline.")
    from src.scripts import embeddings as ae_pipeline_steps

    config_path = os.path.join(AE_CFG_PATH)
    config = load_ae_config(config_path)


    # 1. Data preparation: downloading, preprocessing, and saving
    ae_pipeline_steps.download_and_prepare_data(config)


    # 2.1 Train or load model
    if config.model.train_model:
        ae_pipeline_steps.train_ae_model(config)


    # 2.2 Plot loss function
    if config.output.plot_losses:
        ae_pipeline_steps.plot_losses_step(config)


    # 3.1 Extract embeddings
    if config.output.extract_embeddings:
        ae_pipeline_steps.extract_embeddings(config)


    # 3.2 Visualize reconstructions
    if config.output.visualize_reconstructions:
        ae_pipeline_steps.visualize_reconstructions(config)


    # 3.3 Evaluate reconstruction quality
    if config.output.evaluate_reconstructions:
        ae_pipeline_steps.evaluate_reconstruction_quality(config)


    logger.info("Autoencoder pipeline completed successfully.")



def main(config):
    """
    Main function to run the pipelines.
    """
    create_dirs()


    if config["run_ae_pipeline"]:
        ae_pipe()
    else:
        logger.info("AE pipeline skipped.")


    if config["run_logreg_pipeline"]:
        logreg_pipe()
    else:
        logger.info("LogReg pipeline skipped.")



if __name__ == "__main__":
    config = load_config(MAIN_CFG_PATH)
    main(config)
