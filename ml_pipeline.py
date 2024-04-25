import sys
sys.path.append('/Car Auction Prices Prediction')

import os
print(os.getcwd())

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib
from transformers import FeatureDropper, YearToAgeTransformer, Regrouper

def load_data():
    data = pd.read_csv("C:\\used_car_prices_dataset.csv")
    y = data['price_usd']
    X = data.drop('price_usd', axis=1)
    return X, y

def build_pipeline():
    feature_dropper = FeatureDropper(features_to_drop=['car_id','colour', 'fuel_type', 'steering_position','price_mmk'])
    year_to_age = YearToAgeTransformer()
    regrouper = Regrouper(mapping={'car_brand': {"toyota": "high_end_brands", "honda": "mid_range_brands"}, 'transmission': {"auto": "auto", "manual": "non_auto"}})

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['car_brand', 'car_make','transmission','body_type']),
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
        ('year_to_age', year_to_age),
        ('regroup', regrouper),
        ('preprocess', preprocessor),
        ('rf', RandomForestRegressor(random_state=42, **params))
    ])

    return pipeline

if __name__ == "__main__":
    X, y = load_data()
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    joblib.dump(pipeline, 'model_pipeline.pkl')
