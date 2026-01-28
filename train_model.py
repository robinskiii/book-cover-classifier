import numpy as np

def load_processed_data():
    """
    getting the X and y data from the numpy files
    """
    print("\nLoading data from 'numpy_data.npz'...")
    
    # Load the file
    data = np.load('numpy_data.npz')
    
    # Extract the arrays
    X = data['X']
    y = data['y']
    classes = data['class_names']
    
    print(f"Loaded {len(X)} images. Input features per image: {X.shape[1]}\n")
    return X, y, classes


X, y, classes = load_processed_data()


def train_model(X,y):
    return 0



