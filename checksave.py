import h5py

model_path = 'final_model_2dcnn.h5'  # Path to your newly saved .h5 model

try:
    with h5py.File(model_path, 'r') as f:
        print("File opened successfully. It's a valid HDF5 file.")
        print("HDF5 Keys:", list(f.keys()))
except OSError as e:
    print("Error:", e)
    print("The file is not a valid HDF5 file.")
