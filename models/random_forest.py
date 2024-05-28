from sklearn.ensemble import RandomForestClassifier
import joblib

class RandomForestModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def fit(self, X, y):
        self.model.fit(X, y)
        joblib.dump(self.model, 'models/random_forest.pkl')

    def predict(self, X):
        model = joblib.load('models/random_forest.pkl')
        return model.predict(X)

