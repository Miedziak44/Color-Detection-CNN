import os
# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from . import config

def get_datasets(data_dir=config.DATA_DIR, validation_split=0.2):
    """
    Creates training and validation datasets from a directory structure.

    Args:
        data_dir (str): Path to the dataset directory.
        validation_split (float): Fraction of data to reserve for validation (0.0 - 1.0).

    Returns:
        tuple: A pair of (train_ds, val_ds) TensorFlow datasets.
    """
    
    # Create training subset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=123,  # Fixed seed for reproducible splits
        image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE
    )

    # Create validation subset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE
    )
    
    return train_ds, val_ds