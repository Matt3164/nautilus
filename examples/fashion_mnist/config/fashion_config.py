import os

from examples.fashion_mnist.config.models.bow import PCA_KMEANS, BOW_MODEL
from examples.fashion_mnist.config.models.bow_orb import BOW_ORB_KM
from examples.fashion_mnist.config.models.partial_rf import PARTIAL_RF
from examples.fashion_mnist.config.models.pca_nn import PCA_NN
from examples.fashion_mnist.config.models.rf_lbp import RF_LBP
from nautilus.config.buffer_config import BufferConfig
from nautilus.config.partial_trainer_config import PartialTrainerConfig


class MnistConfig:


    # config for training model

    config = BufferConfig()

    # Random forest based on LBP features
    RF_LBP = RF_LBP

    # PCA on patches and Nearest neighbours
    PCA_NN = PCA_NN


    # Random forest learned on different batches

    PARTIAL_RF = PARTIAL_RF

    partial_config = PartialTrainerConfig(
        bath_size=2048,
        n_epoch=3
    )


    # ----------------------------------------------

    # Config for bag of word model

    # ----------------------------------------------

    root_path = os.path.join(os.path.dirname(__file__), "..", "models_bin")

    model_path = os.path.join(root_path, "kmeans.bin")

    bow_model_path = os.path.join(root_path, "bow.bin")

    BOW_PCA_KMEANS = PCA_KMEANS

    BOW_MODEL = BOW_MODEL

    BOW_ORB_KM = BOW_ORB_KM

    orb_km_path = os.path.join(root_path, "kmeans_orb.bin")