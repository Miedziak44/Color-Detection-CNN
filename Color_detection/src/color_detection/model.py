import os
# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models
from . import config

def build_model(num_classes):
    """
    Builds a custom CNN model optimized for color detection.

    The model focuses on color averaging rather than complex shape detection.
    It includes Data Augmentation, Normalization, Convolutional layers, 
    and Dropout for regularization.

    Args:
        num_classes (int): The number of output classes (colors).

    Returns:
        tf.keras.Model: The compiled Keras model architecture.
    """

    inputs = tf.keras.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3))

    # Data Augmentation pipeline
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"), 
        layers.RandomRotation(0.2),                   
        layers.RandomZoom(0.2),                      
        layers.RandomContrast(0.3),                   
        layers.RandomBrightness(0.3),               
    ], name="data_augmentation")
    
    x = data_augmentation(inputs)

    # Normalization [0, 255] -> [0, 1]
    x = layers.Rescaling(1./255)(x)

    # Layer 1: Conv + MaxPool
    x = layers.Conv2D(16, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    # Layer 2: Conv + MaxPool
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    # Layer 3: Conv + MaxPool
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.MaxPooling2D()(x)

    # Global Average Pooling (spatial reduction)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dropout for regularization
    x = layers.Dropout(0.3)(x)

    # Output Layer
    outputs = layers.Dense(num_classes)(x)

    model = models.Model(inputs, outputs)
    return model