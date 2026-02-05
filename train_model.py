import numpy as np

# model consists of 1 hidden layer with 64 neurons (Relu activation) and output layer with 5 possible outputs

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

    np.random.seed(0)
    p = np.random.permutation(len(y)) #to shuffle the data into a random order
    X = X[p]
    y = y[p]

    return X, y, classes

def initialise_parameters():
    """
    initial values for weights and biases using He Initialization
    """
    np.random.seed(0) 
    W1 = np.random.randn(122880, 64) * np.sqrt(2. / 122880)
    b1 = np.zeros(64)
    W2 = np.random.randn(64, 5) * np.sqrt(2. / 64)
    b2 = np.zeros(5)
    return W1, b1, W2, b2

def sigmoid(x):
    return (1/(1 + np.exp(-x)))

def relu(x):
    for i in range(len(x)):
        if x[i]<0:
            x[i]=0
    return x

def prediction(x, W1, b1, W2, b2):
    """
    Calculates the models prediction for a given book.
    Model uses one hidden layer
    Input:
        x    --> vector with 122880 parametera
        W1   --> 122880 x 64 matrix for layer 1
        b1   --> 64 vector for layer 1  
        W2   --> 64 x 5 matrix for output
        b2   --> 5 vector for output
    Output:
        y    --> prediction vector with probability for each 5 genres
    """
    z = np.matmul(x, W1) + b1 #layer 1 calculation
    z = relu(z) #layer 1 Relu activation 
    z = np.matmul(z, W2) + b2 #output layer calculation
    z = np.exp(z) #softmax part 1
    return z / np.sum(z) #softmax part 2

    




def compute_cost(X, y, W1, b1, W2, b2):
    """
    Computes the cost over a set of training examples.
    We want to find W that minimises cost.
    """
    return 0 #to finish (apply to neural network w multiclass)

def compute_gradient(X, y, W_1, W_2):
    return 0 #to finish



def main():
    X, y, classes = load_processed_data()
    W1, b1, W2, b2 = initialise_parameters()
    print(X.shape)
    print(y.shape)
    print(classes,"\n")
    print(W1.shape)
    print(b1.shape)
    print(W2.shape)
    print(b2.shape)
    print(prediction(X[0], W1, b1, W2, b2))
    

if __name__ == "__main__":
    main()