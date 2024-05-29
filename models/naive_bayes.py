from sklearn.naive_bayes import GaussianNB
import joblib

class NaiveBayesModel:
    def __init__(self):
        self.model = GaussianNB()

    def fit(self, X, y):
        self.model.fit(X, y)
        joblib.dump(self.model, 'models/naive_bayes.pkl')

    def predict(self, X):
        model = joblib.load('models/naive_bayes.pkl')
        return model.predict(X)

    def predict_proba(self, X):
        model = joblib.load('models/naive_bayes.pkl')
        return model.predict_proba(X)

