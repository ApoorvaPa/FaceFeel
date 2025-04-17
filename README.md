# FaceFeel: Facial Emotion Recognition with MediaPipe and 2D CNN
FaceFeel is a lightweight facial emotion recognition pipeline that combines facial landmark extraction using MediaPipe with a 2D Convolutional Neural Network (CNN). It supports dataset generation using a webcam or video, training a custom CNN model, and running real-time inference.
✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨ ✨
## Features
- Data Generation: Capture facial landmarks from webcam or video using MediaPipe and save them with emotion labels in a CSV file.
- Model Training: Train a 2D CNN model using the generated dataset, with checkpointing for best performance.
- Real-Time Inference: Detect emotions in real-time using a webcam and the trained CNN model.
- Model Conversion: Convert .keras models to .h5 format for compatibility.
- Model Validation: Verify the integrity of converted .h5 models.

## Project Structure

```bash
facefeel/
├── venv/                         # Virtual environment directory (excluded from Git)
├── datagen.py                    # Script to generate landmark data using webcam
├── train2dcnn.py                 # CNN model training script
├── test2dcnn.py                  # Real-time inference using webcam and trained model
├── convert.py                    # Converts .keras model to .h5 format
├── checksave.py                  # Verifies .h5 model integrity
├── data.csv                      # Generated dataset with landmark coordinates and labels
├── best_model_2dcnn.keras        # Model checkpoint (lowest validation loss)
├── final_model_2dcnn.keras       # Final saved model after training
├── final_model_2dcnn.h5          # (Optional) Converted .h5 model
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## Prerequisites
- Python 3.8 or higher
- Webcam (for data generation and real-time inference)
- Git (for cloning the repository)

## Getting Started
### 1. Clone and Set Up Virtual Environment
```bash
git clone https://github.com/your-username/facefeel.git
cd facefeel
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Training Data
#### 1. Open datagen.py and set the desired emotion class:
```python
class_name = 'happy'  # Change to 'sad', 'neutral', etc.
```
### python datagen.py
```python
python datagen.py
```
- The webcam will capture frames, and MediaPipe will extract 468 facial landmarks per frame.
- Landmark coordinates and the specified emotion label are saved in data.csv.
- Repeat for multiple emotions (e.g., 'happy', 'sad', 'neutral') to create a diverse dataset.

### 3. Train the model
Run the training script:
```bash
python train2dcnn.py
```
- Loads data.csv and trains a 2D CNN.
- Saves the best model (best_model_2dcnn.keras) based on validation loss and the final model (final_model_2dcnn.keras).

### 4. Real-Time Emotion Detection
Run the inference script:
```bash
python test2dcnn.py
```
- Opens the webcam, extracts facial landmarks, and predicts emotions using the trained model.
- Displays the predicted emotion in real-time.

### 5. (Optional) Convert .keras Model to .h5
To convert the .keras model to .h5 format:
```bash
python convert.py
```
- Edit the script if your model filenames differ from the defaults.

### 6. (Optional) Validate HDF5 Model
- To verify the integrity of the converted .h5 model:
```bash
python checksave.py
```
## Emotion Classes
Update the emotion_labels list in test2dcnn.py to match your dataset:
```python
emotion_labels = ['happy', 'neutral', 'sad']
```
Ensure the label order is consistent across datagen.py, train2dcnn.py, and test2dcnn.py.

## Dependencies
Install the required Python packages:
```bash
pip install -r requirements.txt
```
requirements.txt:
```plain
opencv-python
mediapipe
numpy
pandas
tensorflow
cvzone
scikit-learn
```
## Troubleshooting
- Webcam Issues: Ensure your webcam is connected and accessible.
- MediaPipe Errors: Verify that mediapipe is installed correctly (pip show mediapipe).
- Model Loading Errors: Ensure the model files (best_model_2dcnn.keras or final_model_2dcnn.keras) exist in the project directory.
- Dataset Issues: Check that data.csv contains sufficient samples for each emotion class.

#### Author
Built with ❤️ by Apoorva Parashar

### License
This project is licensed under the MIT License. See the LICENSE file for details.

### Acknowledgments
- MediaPipe for facial landmark detection
- TensorFlow for CNN implementation
- OpenCV for webcam handling
