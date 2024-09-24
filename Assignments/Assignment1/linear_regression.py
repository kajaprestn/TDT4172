import numpy as np
import pandas as pd

class LinearRegression():
    
    def __init__(self, lr=0.0001, iterations=1000):
        self.lr = lr # hvor stor skrittstørrelse for å oppdatere vektene
        self.iterations = iterations # hvor mange ganger skal vi iterere over treningsdata
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        if isinstance(X, pd.Series):
            X = X.values # dersom X er en pandas Series, konverter til numpy array
        if len(X.shape) == 1: 
            X = X.reshape(-1, 1) # dersom X er en vektor, konverter til matrise
        
        n_samples, n_features = X.shape # hente antall eksempler og antall features
        self.w = np.zeros(n_features) # initialiser vekter
        self.b = 0 # initialiser bias

        for _ in range(self.iterations): # for hvert steg i gradient descent algoritmen    
            y_pred = self.predict(X) # prediker verdier basert på nåværende vekter og bias

            dW = (1 / n_samples) * np.dot(X.T, (y_pred - y)) # bruk gradient descent for å oppdatere vektene
            db = (1 / n_samples) * np.sum(y_pred - y)# bruk gradient descent for å oppdatere bias

            self.w = self.w - self.lr * dW # oppdater vektene
            self.b = self.b - self.lr * db # oppdater bias

        return self

    def predict(self, X):
        if isinstance(X, pd.Series):
            X = X.values # dersom X er en pandas Series, konverter til numpy array
        if len(X.shape) == 1:
            X = X.reshape(-1, 1) # dersom X er en vektor, konverter til matrise
        
        return X.dot(self.w) + self.b # matrisemultiplikasjon mellom X og w, og legger til bias
