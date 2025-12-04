import os
# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import argparse
import numpy as np

import tensorflow as tf
from . import config

def get_class_names():
    """
    Retrieves class names from the dataset directory structure.

    Returns:
        list: Sorted list of class names (strings). Returns a default list if directory is missing.
    """
    if os.path.exists(config.DATA_DIR):
        classes = sorted([d for d in os.listdir(config.DATA_DIR) 
                          if os.path.isdir(os.path.join(config.DATA_DIR, d))])
        if classes:
            return classes
    
    print("Dataset file not found - using a backup list.")
    return ['black', 'blue', 'brown', 'green', 'grey', 'orange', 'red', 'violet', 'white', 'yellow']

def predict_images(input_paths, model_path=config.MODEL_PATH):
    """
    Runs inference on a list of image paths or directories.

    Args:
        input_paths (list): List of file or directory paths to process.
        model_path (str): Path to the trained Keras model file.
    """
    final_image_list = []
    
    # Aggregate all valid image files
    for path in input_paths:
        if os.path.isdir(path):
            print(f" Scanning the file contents: {path}...")
            files = [os.path.join(path, f) for f in os.listdir(path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            final_image_list.extend(files)
        elif os.path.isfile(path):
            final_image_list.append(path)
        else:
            print(f"Path doesn't exist: {path}")

    if not final_image_list:
        print("No files found to predict.")
        return

    print(f"--- Loading model from: {model_path} ---")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Critical error: Can't load model.\n{e}")
        return

    class_names = get_class_names()
    print(f"Integrated classes: {class_names}")
    print(f"Images to analyze: {len(final_image_list)}")
    print("-" * 50)

    # Inference Loop
    for img_path in final_image_list:
        try:
            # Preprocess image
            img = tf.keras.utils.load_img(
                img_path, target_size=(config.IMG_HEIGHT, config.IMG_WIDTH)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Add batch dimension

            # Predict
            predictions = model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0])

            # Process results
            predicted_class = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)

            file_name = os.path.basename(img_path)
            print(f"################################## {file_name:<20} -> |\_______( {predicted_class.upper()} ({confidence:.1f}%) )_______/|")

            print("     (Top 3):")  
            top_3_indices = np.argsort(score)[-3:][::-1] 
            
            for i in top_3_indices:
                print(f"   - {class_names[i]}: {100 * score[i]:.2f}%")

        except Exception as e:
            print(f"Error with {os.path.basename(img_path)}: {e}")

    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detecting colors in pictures or folders.")
    parser.add_argument("paths", nargs='+', help="Path to a file or folder")
    
    args = parser.parse_args()
    
    predict_images(args.paths)