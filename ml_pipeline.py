import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Custom transformers for feature engineering
class YearToAgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, base_year=None):
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

class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_drop):
        self.features_to_drop = features_to_drop
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(self.features_to_drop, axis=1)

class Regrouper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping, default_label = 'Other'):
        self.mapping = mapping
        self.default_label = default_label

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        for column, map_dict in self.mapping.items():
            if column in X.columns:
                X[column] = X[column].map(map_dict).fillna('Other')
        return X

# Load data
def load_data():
    # Dummy load function, replace with actual data loading
    data = pd.read_csv('your_data.csv')
    y = data['target_column']  # Adjust the column name
    X = data.drop('target_column', axis=1)  # Drop the target column
    return X, y

# Build the pipeline
def build_pipeline():
    feature_dropper = FeatureDropper(features_to_drop=['car_id','colour', 'fuel_type', 'steering_position','price_mmk'])
    regrouper = Regrouper(mapping={
        'car_brand':{
            "toyota" : "high_end_brands",
            "honda" : "mid_range_brands",
            "nissan" : "mid_range_brands",
            "mitsubishi" : "mid_range_brands",
            "suzuki" : "low_end_brands",
            "daihatsu" : "low_end_brands"
        },

        'transmission': {
        "auto": "auto",
        "semi auto": "non_auto",
        "manual": "non_auto"
    }
    })
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'),['car_brand','car_make','transmission'])
        ], remainder='passthrough')


    params = {
        'max_depth': 7, 
        'max_features': None, 
        'max_samples': 0.9, 
        'min_samples_leaf': 2, 
        'n_estimators': 80
    }

    pipeline = Pipeline([
        ('drop_features', feature_dropper),
        ('regroup', regrouper),
        ('preprocess', preprocessor),
        ('rf', RandomForestRegressor(random_state=42, **params))
    ])
    
    return pipeline

# Main function to train and save the model
if __name__ == "__main__":
    X, y = load_data("\used_car_prices_dataset.csv")
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    joblib.dump(pipeline, 'model_pipeline.pkl')
