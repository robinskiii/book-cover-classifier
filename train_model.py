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
        Y_hat --> prediction matrix with probability vectors for the 5 genres
        Z1    --> intermediate matrix needed for back-propagation (1st layer before ReLU)
        A1    --> intermediate matrix needed for back-propagation (1st layer after ReLU)
    """
    #layer 1
    Z1 = np.matmul(X, W1) + b1

    #layer 1 ReLU activation
    A1 = np.maximum(0, Z1)

    #layer 2
    Z2 = np.matmul(A1, W2) + b2

    #softmax
    Z2_shifted = Z2 - np.max(Z2, axis=1, keepdims=True) #to avoid big exponentials that previously caused bugs 
    exp_Z = np.exp(Z2_shifted)
    Y_hat = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    return Y_hat, Z1, A1

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

def compute_gradient(X, y, W1, b1, W2, b2, Y_hat, Z1, A1, lambda_):
    """
    Input: 
        X, y            --> training data
        W1, b1, W2, b2  --> current model parameters
        Y_hat, Z1, A1   --> variables calculated from prediction (matrices)
        lambda_         --> regularization constant
    Output:
        d_W1, d_W2      --> gradient of weights
        d_b1, d_b2      --> gradient of biases
    """
    m = len(y)
    
    #convert y to matrix of 1-hot encoded vectors (example: [0, 0, 1, 0, 0] instead of 2)
    Y_1hot = np.zeros_like(Y_hat)
    Y_1hot[np.arange(m), y] = 1

    d_Z2 = Y_hat - Y_1hot

    d_W2 = (1/m) * np.matmul(A1.T ,d_Z2) + (lambda_/m) * W2
    d_b2 = (1/m) * np.sum(d_Z2, axis=0)

    d_Z1 = np.matmul(d_Z2, W2.T) * (Z1 > 0)

    d_W1 = (1/m) * np.matmul(X.T, d_Z1) + (lambda_/m) * W1
    d_b1 = (1/m) * np.sum(d_Z1, axis = 0)

    return d_W1, d_b1, d_W2, d_b2

#training the model
def main():
    X, y, classes = load_processed_data()
    W1, b1, W2, b2 = initialise_parameters()

    #PARAMETERS
    LEARNING_RATE = 0.04 #learning strength
    EPOCHS = 5000 #learning iterations
    LAMBDA_ = 0  #regularization term
    
    print(f"\nStarting training on {len(X)} images...")
    
    for i in range(EPOCHS):

        Y_hat, Z1, A1 = prediction(X, W1, b1, W2, b2)
        
        #message every n iterations to signal cost
        if i % 50 == 0:
            cost = compute_cost(Y_hat, y, W1, W2, LAMBDA_)
            print(f"Epoch {i}: Cost = {cost}")
        
        #calculate gradients
        d_W1, d_b1, d_W2, d_b2 = compute_gradient(X, y, W1, b1, W2, b2, Y_hat, Z1, A1, LAMBDA_)

        #every m iterations, we cut in half the learning rate
        if i > 0 and i % 500 == 0:
            LEARNING_RATE *= 0.8
            print(f"--- Reducing learning rate to {LEARNING_RATE} ---")
        
        #update weights
        W1 = W1 - LEARNING_RATE * d_W1
        b1 = b1 - LEARNING_RATE * d_b1
        W2 = W2 - LEARNING_RATE * d_W2
        b2 = b2 - LEARNING_RATE * d_b2


if __name__ == "__main__":
    main()