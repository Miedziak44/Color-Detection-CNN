import os
# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from . import config
from . import dataset

def representative_data_gen():
    """
    Generates representative data for INT8 quantization calibration.

    This generator iterates through the training dataset, casting images to float32
    and expanding dimensions to match the model's expected input shape.

    Yields:
        list: A list containing a single image tensor of shape (1, height, width, 3).
    """
    # Gets the training dataset to use for calibration
    train_ds, _ = dataset.get_datasets() 
    
    print("Generator calibration started...")

    # Take first 20 batches for calibration
    for batch_images, _ in train_ds.take(20):      
        for i in range(batch_images.shape[0]):
            image = tf.cast(batch_images[i], tf.float32)
            image = tf.expand_dims(image, 0)
            yield [image]

def convert_model():
    """
    Loads a Keras model, applies INT8 quantization, and saves it as a TFLite model.

    The function performs the following steps:
    1. Loads the model from config.MODEL_PATH.
    2. Configures the TFLite converter for INT8 optimization.
    3. Uses representative_data_gen for calibration.
    4. Enforces integer-only operations for microcontroller compatibility.
    5. Saves the converted model to config.TFLite_MODEL_PATH.

    Raises:
        IOError: If the model path implies file access issues.
    """
    print(f"Loading Keras model from {config.MODEL_PATH}...")      
    model = tf.keras.models.load_model(config.MODEL_PATH)
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Ensure full integer quantization compatible with microcontrollers
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    
    print("Starting conversion...")
    tflite_model = converter.convert()    
    
    with open(config.TFLITE_MODEL_PATH, 'wb') as f:     
        f.write(tflite_model)
    print('\n#####################################################')
    print(f"TFLite model saved to: {config.TFLITE_MODEL_PATH}")

if __name__ == "__main__":
    convert_model()