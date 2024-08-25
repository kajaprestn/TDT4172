import numpy as np

class LinearRegression():
    
    def __init__(self, lr = 0.0001, iterations = 1000):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.lr = lr
        self.iterations = iterations
        self.w = None
        self.b = None

        pass
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """

        if len(X.shape) == 1:
            X = X.values.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.iterations):
            y_pred = np.dot(X, self.w) + self.b

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """

        if len(X.shape) == 1:
            X = X.values.reshape(-1, 1)
    
        y_pred = np.dot(X, self.w) + self.b
        return y_pred
