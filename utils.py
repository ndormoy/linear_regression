import pandas as pd
import os
import numpy as np

def load_csv(path: str) -> pd.DataFrame:
    try:
        if not isinstance(path, str):
            return None
        elif not os.path.exists(path):
            return None
        elif not path.lower().endswith('.csv'):
            return None
        df = pd.read_csv(path)
        print("Csv file loaded")
    except Exception as e:
        print(e)
    return df


def predict(theta0, theta1, mileage):
    price = int(theta0 + theta1 * mileage)
    return price


def normalize_data(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data


def denormalize_data(data, theta):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    denormalized_theta = theta * (max_vals - min_vals) + min_vals
    return denormalized_theta