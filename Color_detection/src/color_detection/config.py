"""
Configuration module for the Color Detection project.

This module contains global constants for image processing, model parameters,
and file system paths.
"""
import os

# --- Image Parameters ---
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
CHANNELS = 3

# --- Source Paths ---
# Directory containing the dataset sorted by class folders
DATA_DIR = r"C:\Users\Miedz\Desktop\Color_detection\dataset"

# Path to save/load the Keras model
MODEL_PATH = "best_model.keras"

# Path for the quantized TFLite model
TFLITE_MODEL_PATH = "model_quant_int8.tflite"