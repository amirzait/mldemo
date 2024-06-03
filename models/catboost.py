from catboost import CatBoostClassifier
import joblib

class CatBoostModel:
    def __init__(self):
        self.model = CatBoostClassifier(iterations=100)  # Adjust parameters as needed

    def fit(self, X, y):
        self.model.fit(X, y)
        joblib.dump(self.model, 'models/catboost.pkl')

    def predict(self, X):
        model = joblib.load('models/catboost.pkl')
        return model.predict(X)

    def predict_proba(self, X):
        model = joblib.load('models/catboost.pkl')
        return model.predict_proba(X)