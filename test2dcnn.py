import cvzone
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from cvzone.FaceMeshModule import FaceMeshDetector

# ----------------------------- Configuration ----------------------------- #

# Path to the trained model
MODEL_PATH = 'best_model_2dcnn.keras'  # Ensure this path is correct

# Define the emotion classes in the exact order as during training
# Based on np.unique sorted alphabetically
emotion_labels = ['happy', 'neutral', 'sad']  # Adjust if the order is different

# Video capture device (0 for default webcam)
CAMERA_INDEX = 0

# Confidence threshold (optional)
CONFIDENCE_THRESHOLD = 0.5  # Adjust as needed

# -------------------------------------------------------------------------- #

def main():
    # Load the trained model
    try:
        model = load_model(MODEL_PATH)
        print(f"Successfully loaded model from '{MODEL_PATH}'")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize Face Mesh Detector
    detector = FaceMeshDetector(maxFaces=1)  # Detect one face at a time

    # Initialize webcam
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting video stream. Press 'q' to exit.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # Optional: Resize frame for faster processing
        frame = cv2.resize(frame, (720, 480))

        # Detect face mesh
        img, faces = detector.findFaceMesh(frame, draw=False)  # Set draw=True to visualize landmarks

        if faces:
            # Process the first detected face
            face = faces[0]  # List of 468 (x, y) tuples

            # Convert landmarks to a NumPy array and flatten
            face_landmarks = np.array(face).flatten()  # Shape: (936,)

            # Ensure that we have exactly 936 values
            if face_landmarks.shape[0] != 936:
                print(f"Unexpected number of landmarks: {face_landmarks.shape[0]}")
                continue

            # Reshape to (468, 2)
            try:
                face_landmarks = face_landmarks.reshape(468, 2)
            except ValueError as ve:
                print(f"Reshape error: {ve}")
                continue

            # Expand dimensions to match model input: (1, 468, 2, 1)
            input_data = face_landmarks.reshape(1, 468, 2, 1).astype(np.float32)

            # If you applied normalization during training, apply it here
            # Example: input_data /= np.max(input_data)
            # Uncomment and modify the following line if needed
            # input_data /= 255.0  # Example normalization

            # Predict emotion
            predictions = model.predict(input_data)
            predicted_index = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_index]

            # Optional: Apply confidence threshold
            if confidence < CONFIDENCE_THRESHOLD:
                emotion = 'Uncertain'
            else:
                # Map predicted index to emotion label
                if predicted_index < len(emotion_labels):
                    emotion = emotion_labels[predicted_index]
                else:
                    emotion = 'Unknown'

            # Display the emotion and confidence on the frame
            cvzone.putTextRect(
                frame, 
                f'Emotion: {emotion}', 
                (50, 50), 
                scale=2, 
                thickness=2, 
                colorT=(0, 255, 0)
            )
            cvzone.putTextRect(
                frame, 
                f'Confidence: {confidence*100:.2f}%', 
                (50, 100), 
                scale=1.5, 
                thickness=2, 
                colorT=(0, 255, 0)
            )

            # Optional: Print to console
            print(f'Predicted Emotion: {emotion}, Confidence: {confidence*100:.2f}%')

        else:
            # Optional: Indicate that no face was detected
            cvzone.putTextRect(
                frame, 
                'No Face Detected', 
                (50, 50), 
                scale=2, 
                thickness=2, 
                colorT=(0, 0, 255)
            )

        # Display the frame
        cv2.imshow("Emotion Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
