from sklearn.linear_model import LogisticRegression
import joblib

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        self.model.fit(X, y)
        joblib.dump(self.model, 'models/logistic_regression.pkl')  # Save for later use

    def predict(self, X):
        model = joblib.load('models/logistic_regression.pkl')  # Load saved model
        return model.predict(X)
