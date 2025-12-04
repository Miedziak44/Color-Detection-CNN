import os
# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import tensorflow as tf
from PIL import Image
from . import config

def convert_png_to_jpg(directory=config.DATA_DIR, delete_original=True):
    """
    Converts all PNG images in a directory (recursive) to JPG format.

    Args:
        directory (str): The root directory to scan for images. Defaults to config.DATA_DIR.
        delete_original (bool): If True, deletes the original PNG file after successful conversion.
    """
    print(f"Beginning conversion PNG -> JPG in: {directory}")
    converted_count = 0
    
    # Traverse through directory tree
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.lower().endswith('.png'):
                file_path_png = os.path.join(root, filename)
                file_path_jpg = os.path.join(root, os.path.splitext(filename)[0] + '.jpg')
                
                try:
                    with Image.open(file_path_png) as img:
                        # Convert to RGB (removes alpha channel) and save as high-quality JPEG
                        rgb_img = img.convert('RGB')
                        rgb_img.save(file_path_jpg, 'JPEG', quality=95)
                        
                    converted_count += 1
                    
                    if delete_original:
                        os.remove(file_path_png)
                        
                except Exception as e:
                    print(f"Error with file {file_path_png}: {e}")
    print(f"Converted {converted_count} files.")

def find_corrupted_images(directory=config.DATA_DIR):
    """
    Scans the directory for images that cannot be opened by the PIL library.

    Args:
        directory (str): The root directory to scan.

    Returns:
        list: A list of file paths corresponding to corrupted images.
    """
    print("Scanning for corrupted files (PIL)...")
    corrupted_files = []
    
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                # Verify image integrity without decoding pixel data
                with Image.open(file_path) as img:
                    img.verify() 
            except Exception as e:            
                print(f"Corrupted file: {file_path} ({e})")
                corrupted_files.append(file_path)
    return corrupted_files

def validate_tf_decode(directory=config.DATA_DIR):
    """
    Validates if images can be decoded by TensorFlow's IO engine.
    
    This step is crucial before creating a TF Dataset to avoid runtime errors during training.

    Args:
        directory (str): The root directory to scan.
    """
    print("Scanning for TensorFlow decoding errors...")
    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            file_path = os.path.join(root, filename)
            try:
                # Attempt to read and decode the image
                raw_bytes = tf.io.read_file(file_path)
                tf.image.decode_image(raw_bytes, channels=3)
            except Exception as e:
                print(f" TensorFlow error for: {file_path} \nContents: {e.message}")

if __name__ == "__main__":
    convert_png_to_jpg()
    find_corrupted_images()
    validate_tf_decode()