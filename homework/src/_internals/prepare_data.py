# descarga de datos
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")

    # preparacion de datos
    y = df["quality"]
    x = df.copy()
    x.pop("quality")

    # dividir los datos en entrenamiento y testing
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=123456,
    )

    return x_train, x_test, y_train, y_test
