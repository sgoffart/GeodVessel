import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class CFC:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            verbose=0,
            class_weight="balanced"  # useful for imbalanced centerline data
        )
        self._is_fitted = False

    def fit(self, X, y):
        X = X.astype(np.float32)
        self.model.fit(X, y)
        self._is_fitted = True

    def predict_proba(self, X):
        X = X.astype(np.float32)
        return self.model.predict_proba(X)

    def predict(self, X):
        X = X.astype(np.float32)
        return self.model.predict(X)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"✅ Model saved to {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self._is_fitted = True
        print(f"✅ Model loaded from {path}")
