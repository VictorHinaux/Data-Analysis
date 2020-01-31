import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


def analyze(filename):
    # Import data from CSV
    data = pd.read_csv(filename)

    # Multi-indexing
    data.set_index(["AGE", "GENDER", "HEIGHT", "WEIGHT",
                    "SKIN", "SPORT"], inplace=True)

    # Dataset training/validation spltting
    training_set, validation_set = train_test_split(
        data, test_size=0.2, random_state=21)

    # Attribute/label splitting
    X_train = [x for x in training_set.index.values]
    Y_train = training_set.values
    X_val = [x for x in validation_set.index.values]
    Y_val = validation_set.values

    # Preprocessing: attributes scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)

    # Regression modelling
    regressor = RandomForestRegressor(n_estimators=2000, random_state=1)
    regressor.fit(X_train, Y_train)

    # Prediction test
    y_pred = regressor.predict(X_val)
    print(Y_val)
    print(y_pred)

    # Accuracy evaluation
    print('Root Mean Squared Error:', np.sqrt(
        metrics.mean_squared_error(Y_val, y_pred)))


def main():
    filename = "features_and_bpms.csv"
    return analyze(filename)


if __name__ == '__main__':
    exit(main())
