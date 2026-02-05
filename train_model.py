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
    return np.maximum(0, x)

def prediction(X, W1, b1, W2, b2):
    """
    Calculates the models prediction for a given book.
    Model uses one hidden layer
    Input:
        X     --> matrix of vectors with 122880 parameters each
        W1    --> 122880 x 64 matrix for layer 1
        b1    --> 64 vector for layer 1  
        W2    --> 64 x 5 matrix for output
        b2    --> 5 vector for output
    Output:
        y_hat --> prediction matrix with probability vectors for the 5 genres
    """
    y_hat = []

    for i in range(len(X)):
        z = np.matmul(X[i], W1) + b1 #layer 1 calculation
        z = relu(z) #layer 1 Relu activation 
        z = np.matmul(z, W2) + b2 #output layer calculation
        z = np.exp(z) #softmax part 1
        z /= np.sum(z) #softmax part 2
        y_hat.append(z)

    return y_hat

def compute_cost(y_hat, y, W1, W2, lambda_):
    """
    Computes the cost over a set of training examples.
    We want to find W1, b1, W2, b2 that minimises cost.
    Inputs:
        y_hat   --> matrix containing prediction vector for each book
        y       --> vector containing label genre for each book (as numbers 0-5)
        lambda_ --> parameter for tweaking regularization
    Output:
        J       --> sum square error loss + regularization term
    """
    J = 0
    m = len(y) #number of training examples
    
    for i in range(m):
        J += -np.log(y_hat[i][y[i]]+ 1e-8) #adding 1e-8 to avoid crashes (log(0))
    
    #regularization term:
    regularization = np.sum(np.square(W1))+np.sum(np.square(W2))
    regularization *= lambda_
    
    J += regularization
    J /= m 
    return J

def compute_gradient(X, y, W1, b1, W2, b2):
    return  #to finish

def apply_gradient_descent(X):
    return 



def main():
    X, y, classes = load_processed_data()
    W1, b1, W2, b2 = initialise_parameters()
    y_hat = prediction(X, W1, b1, W2, b2)
    print(compute_cost(y_hat, y, W1, W2, lambda_= 0))
    

if __name__ == "__main__":
    main()