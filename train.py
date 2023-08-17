import pandas as pd
import numpy as np
import os
from utils import load_csv, predict, normalize_data, denormalize_data

num_iterations = 1000


def calculate_sigma0(tmptheta0, tmptheta1, learning_rate, mileage_i, price_i):
    estimate_price = predict(tmptheta0, tmptheta1, mileage_i)
    tmptheta0 = estimate_price - price_i
    return tmptheta0


def calculate_sigma1(tmptheta0, tmptheta1, learning_rate, mileage_i, price_i):
    tmptheta1 = calculate_sigma0(tmptheta0, tmptheta1, learning_rate, mileage_i, price_i) * mileage_i
    return tmptheta1

def gradient_descent(df, learning_rate):

    tmptheta0 = 0
    tmptheta1 = 0
    for _ in range(num_iterations):
        sigma0 = 0
        sigma1 = 0
        for i in range(df.shape[0]):
            sigma0 += calculate_sigma0(tmptheta0, tmptheta1,
                                          learning_rate, df.iloc[i, 0], df.iloc[i, 1])
            sigma1 += calculate_sigma1(tmptheta0, tmptheta1,
                                          learning_rate, df.iloc[i, 0], df.iloc[i, 1])
        tmptheta0 -= (learning_rate * (sigma0 / (df.shape[0] - 1)))
        tmptheta1 -= (learning_rate * (sigma1 / (df.shape[0] - 1)))
        denormalized_tmptheta0 = denormalize_data(df, tmptheta0)
        denormalized_tmptheta1 = denormalize_data(df, tmptheta1)
        print(f"tmptheta0 = {tmptheta0}, tmptheta1 = {tmptheta1}")
        print(f"DENORMALIZE tmptheta0 = {denormalized_tmptheta0}, tmptheta1 = {denormalized_tmptheta1}")


def main():
    try:
        df = load_csv("data.csv")
        normalized_data = normalize_data(df)
        learning_rate = 0.0001
        print(df.shape)
        gradient_descent(normalized_data, learning_rate)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()







