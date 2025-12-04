import os
# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from . import config
from . import dataset

def show_training_batch():
    """
    Visualizes a single batch of training data in a 3x3 grid.
    
    Useful for verifying data integrity and label correctness before training.
    """
    print("Loading training data...")
    try:
        train_ds, _ = dataset.get_datasets()
        class_names = train_ds.class_names
        print(f"CLasses found by Keras: {class_names}")
    except Exception as e:
        print(f"Data loading error: {e}")
        return

    print("Loading a batch of images...")
    # Iterate over one batch
    for images, labels in train_ds.take(1):
        plt.figure(figsize=(10, 10))
        
        # Plot first 9 images
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            
            # Convert tensor to uint8 for display
            img = images[i].numpy().astype("uint8")
            label_index = labels[i].numpy()
            label_name = class_names[label_index]
            
            plt.imshow(img)
            plt.title(f"Label: {label_name}")
            plt.axis("off")
        
        print("Showing a plot...")
        plt.show()
        break

if __name__ == "__main__":
    show_training_batch()