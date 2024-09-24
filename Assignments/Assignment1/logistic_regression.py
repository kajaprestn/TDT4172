import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib import pyplot as plt

class LogisticRegression():
    
    def __init__(self, lr=0.001, iterations=10000):
        self.lr = lr # hvor stor skrittstørrelse for å oppdatere vektene
        self.iterations = iterations # hvor mange ganger skal vi iterere over treningsdata
        self.w = None # vekter
        self.b = None # bias
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z)) # konverterer z til en verdi mellom 0 og 1
    
    def compute_loss(self, y, y_pred):
        loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9)) # beregner log loss
        return loss
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.losses = []

        for _ in range(self.iterations): # for hvert steg i gradient descent algoritmen
            z = X.dot(self.w) + self.b # lineær kombinasjon av vektene og input
            y_pred = self.sigmoid(z) # konverterer lineær kombinasjon til en verdi mellom 0 og 1
            
            dW = (1 / n_samples) * np.dot(X.T, (y_pred - y)) # bruk gradient descent for å oppdatere vektene
            dB = (1 / n_samples) * np.sum(y_pred - y)# bruk gradient descent for å oppdatere bias
            
            self.w -= self.lr * dW # oppdater vektene
            self.b -= self.lr * dB # oppdater bias
            
            loss = self.compute_loss(y, y_pred) # beregn loss
            self.losses.append(loss) # lagre loss
        
        # plotter loss over iterasjoner for å se om modellen konvergerer
        plt.plot(range(self.iterations), self.losses)
        plt.title('Loss over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.show()

        return self

    
    def predict_proba(self, X): # prediker sannsynligheten for at en sample tilhører klasse 1
        return self.sigmoid(np.dot(X, self.w) + self.b) # konverterer lineær kombinasjon til en verdi mellom 0 og 1
    
    def predict(self, X, threshold=0.51): # prediker klasse basert på sannsynlighet
        return [1 if i >= threshold else 0 for i in self.predict_proba(X)] # konverterer sannsynlighet til klasse