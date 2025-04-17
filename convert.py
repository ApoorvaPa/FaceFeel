import tensorflow as tf
from tensorflow import keras
import os

def convert_keras_to_h5(input_path, output_path):
    """
    Converts a Keras model from .keras format to .h5 format.

    Parameters:
    - input_path: str, path to the input .keras model file.
    - output_path: str, path where the .h5 model will be saved.
    """
    try:
        # Check if the input file exists
        if not os.path.exists(input_path):
            print(f"Input file '{input_path}' does not exist.")
            return

        # Load the .keras model
        print(f"Loading model from {input_path}...")
        model = keras.models.load_model(input_path)
        print("Model loaded successfully.")

        # Save the model in .h5 format
        print(f"Saving model to {output_path}...")
        model.save(output_path, save_format='h5')
        print(f"Model successfully converted and saved to {output_path}")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")

if __name__ == "__main__":
    # Define the input and output paths
    input_model = 'trained_model.keras'  # Replace with your .keras file path
    output_model = 'SER_Model.h5'   # Desired .h5 file path

    convert_keras_to_h5(input_model, output_model)
