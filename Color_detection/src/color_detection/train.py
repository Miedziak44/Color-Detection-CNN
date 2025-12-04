import os
# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ModelCheckpoint

from . import config
from . import dataset
from . import model as model_module

def calculate_weights(train_ds, class_names):
    """
    Computes class weights to handle dataset imbalance.

    It also applies a manual penalty to the 'black' class if present, 
    to reduce its dominance.

    Args:
        train_ds (tf.data.Dataset): The training dataset.
        class_names (list): List of class names.

    Returns:
        dict: A dictionary mapping class indices to their calculated weights.
    """
    print("Calculating class weights...")
    train_labels = np.concatenate([y for x, y in train_ds], axis=0)
    
    # Compute balanced weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Manual adjustment for specific classes
    if 'black' in class_names:
        try:
            black_index = class_names.index('black')
            class_weight_dict[black_index] = class_weight_dict[black_index] * 0.4 
            print("Manually corrected the weight of 'black'.")
        except ValueError:
            pass
            
    return class_weight_dict

def run_training():
    """
    Main execution function for model training.
    
    Steps:
    1. Loads datasets.
    2. Builds and compiles the model.
    3. Calculates class weights.
    4. Sets up checkpoints.
    5. Runs the training loop.
    """
    try:
        train_ds, val_ds = dataset.get_datasets()
    except Exception as e:
        print(f"Data loading error: {e}")
        return

    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    # Build and Summary
    model = model_module.build_model(len(class_names))
    model.summary()

    # Compile
    model.compile(
        optimizer='adam', 
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Weights and Callbacks
    class_weights = calculate_weights(train_ds, class_names)

    checkpoint_cb = ModelCheckpoint(
        filepath=config.MODEL_PATH,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    print("Beginning training cycle...") 
    try:
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=100, 
            callbacks=[checkpoint_cb],
            class_weight=class_weights 
        )
    except Exception as e:
        print(f"Training error: {e}")

if __name__ == "__main__":
    run_training()