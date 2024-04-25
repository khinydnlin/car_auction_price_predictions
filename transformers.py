from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop):
        self.features_to_drop = features_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X.drop([feature for feature in self.features_to_drop if feature in X.columns], axis=1, inplace=True)
        return X

class YearToAgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, base_year=2019):
        self.base_year = base_year
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        current_year = self.base_year if self.base_year else pd.Timestamp.now().year
        if 'year' in X.columns:
            X['age'] = current_year - X['year']
            X.drop('year', axis=1, inplace=True)
        return X

class Regrouper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping, default_label='Other'):
        self.mapping = mapping
        self.default_label = default_label
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for column, map_dict in self.mapping.items():
            if column in X.columns:
                X[column] = X[column].map(map_dict).fillna(self.default_label)
        return X
