from xgboost import XGBClassifier
import joblib

class XGBoostModel:
    def __init__(self):
        self.model = XGBClassifier(n_estimators=100)

    def fit(self, X, y):
        self.model.fit(X, y)
        joblib.dump(self.model, 'models/xgboost.pkl')

    def predict(self, X):
        model = joblib.load('models/xgboost.pkl')
        return model.predict(X)

    def predict_proba(self, X):
        model = joblib.load('models/xgboost.pkl')
        return model.predict_proba(X)