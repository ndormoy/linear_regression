import pandas as pd
import numpy as np
import os
from utils import predict

theta0 = 0.060311
# 0.060311
theta1 = 0.015807
# 0.015807

def main():
    try:
        mileage = int(input("What is the mileage that you want to predict ? "))
        theta = np.array([[theta0],
                      [theta1]])
        price = predict(theta[0, 0], theta[1, 0], mileage)
        print(f"The predicted price for mileage = {mileage}, is  {price}")
    except:
        print("You can only have a integer in input")
        exit(1)

if __name__ == "__main__":
    main()