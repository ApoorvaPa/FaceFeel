# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import ModelCheckpoint

# # Load the CSV data
# data = pd.read_csv('data.csv')

# # Separate features (landmark coordinates) and labels (emotion class)
# X = data.drop(columns=['Class']).values  # Drop 'Class' column, keeping only the landmark coordinates
# y = data['Class'].values  # Extract the emotion class labels

# # Convert the labels to integers (assuming there are multiple classes)
# unique_classes = np.unique(y)  # Extract unique classes
# class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}  # Map classes to indices
# y = np.array([class_to_index[cls] for cls in y])  # Convert class names to integer labels

# # One-hot encode the labels for multi-class classification
# y = to_categorical(y, num_classes=len(unique_classes))

# # Reshape X from (samples, 936) -> (samples, 468, 2)
# # 936 comes from 468 x-coordinates and 468 y-coordinates
# X = X.reshape(X.shape[0], 468, 2)

# # Split the data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# print(f"Shape of X_train: {X_train.shape}")  # Should output (samples, 468, 2)
# print(f"Shape of y_train: {y_train.shape}")

# # Build the 2D CNN model
# # model = Sequential([
# #     # Conv2D layer expects input in the shape (height, width, channels)
# #     Reshape((468, 2, 1), input_shape=(468, 2)),  # Reshape input for Conv2D
# #     Conv2D(64, kernel_size=(3, 2), activation='relu'),  # Filters move across x, y pairs
# #     Conv2D(128, kernel_size=(3, 2), activation='relu'),
# #     Flatten(),
# #     Dense(256, activation='relu'),
# #     Dense(len(unique_classes), activation='softmax')  # Output layer for multi-class classification
# # ])

# model = Sequential([
#     # Conv2D layer expects input in the shape (height, width, channels)
#     Reshape((468, 2, 1), input_shape=(468, 2)),  # Reshape input for Conv2D
#     Conv2D(64, kernel_size=(2, 2), activation='relu'),  # Filters move across x, y pairs
#     Conv2D(128, kernel_size=(1, 2), activation='relu'),  # Adjust kernel size to avoid negative dimensions
#     Flatten(),
#     Dense(256, activation='relu'),
#     Dense(len(unique_classes), activation='softmax')  # Output layer for multi-class classification
# ])
# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Print model summary
# model.summary()

# # Model checkpoint to save the best model
# checkpoint = ModelCheckpoint('best_model_2dcnn.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# # Train the model
# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=32, callbacks=[checkpoint])

# # Save the final trained model
# model.save('final_model_2dcnn.keras')

# print("Training complete and model saved as 'final_model_2dcnn.keras'.")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Load the CSV data
data = pd.read_csv('data.csv')

# Separate features (landmark coordinates) and labels (emotion class)
X = data.drop(columns=['Class']).values  # Drop 'Class' column, keeping only the landmark coordinates
y = data['Class'].values  # Extract the emotion class labels

# Convert the labels to integers (assuming there are multiple classes)
unique_classes = np.unique(y)  # Extract unique classes
class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}  # Map classes to indices
y = np.array([class_to_index[cls] for cls in y])  # Convert class names to integer labels

# One-hot encode the labels for multi-class classification
y = to_categorical(y, num_classes=len(unique_classes))

# Reshape X from (samples, 936) -> (samples, 468, 2)
# 936 comes from 468 x-coordinates and 468 y-coordinates
X = X.reshape(X.shape[0], 468, 2)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Shape of X_train: {X_train.shape}")  # Should output (samples, 468, 2)
print(f"Shape of y_train: {y_train.shape}")

# Build the 2D CNN model with padding='same'
model = Sequential([
    # Conv2D layer expects input in the shape (height, width, channels)
    Reshape((468, 2, 1), input_shape=(468, 2)),  # Reshape input for Conv2D
    Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'),  # Preserves spatial dimensions
    Conv2D(128, kernel_size=(1, 2), activation='relu', padding='same'),  # Adjusted to use padding='same'
    Flatten(),
    Dense(256, activation='relu'),
    Dense(len(unique_classes), activation='softmax')  # Output layer for multi-class classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Model checkpoint to save the best model
checkpoint = ModelCheckpoint('best_model_2dcnn.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model
model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=25, 
    batch_size=32, 
    callbacks=[checkpoint]
)

# Save the final trained model
model.save('final_model_2dcnn.keras')

print("Training complete and model saved as 'final_model_2dcnn.keras'.")
