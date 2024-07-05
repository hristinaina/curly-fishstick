import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import BaggingClassifier, VotingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import ADASYN


def load_data(file='data/healthcare-dataset-stroke-data.csv'):
    return pd.read_csv(file)


def separate_features(data):
    y = data['stroke']
    X = data.drop(columns=['stroke', 'id'], axis=1)
    return X, y


def handle_missing_values(df):
    df.dropna(inplace=True)
    return df


def remove_outliers(df):
    df = df[(df['bmi'] >= 13) & (df['bmi'] <= 55)]
    df = df[(df['gender'] != 'Other')]
    df = df[(df['work_type'] != 'Never_worked')]
    return df


def oversampling(X, y):
    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    # print(y_resampled.value_counts())
    return X_resampled, y_resampled


def encode(X):
    categ_columns = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categ_columns, drop_first=True)
    return X


def fit_model(X_train, y_train):
    model = VotingClassifier(
        estimators=[('rf1', RandomForestClassifier(max_depth=15, n_estimators=150, random_state=42))],
        voting='soft'
    )
    model.fit(X_train, y_train)
    return model


def reduction(x_train, x_test):
    pca = PCA(n_components=min(x_train.shape[0], x_train.shape[1]))
    x_train_reduced = pca.fit_transform(x_train)
    x_test_reduced = pca.transform(x_test)

    return x_train_reduced, x_test_reduced


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.tight_layout()
    plt.show()


def main():
    df = load_data()
    df = handle_missing_values(df)
    df = remove_outliers(df)

    X, y = separate_features(df)
    X = encode(X)
    X, y = oversampling(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    X_train, X_test = reduction(X_train, X_test)
    model = fit_model(X_train, y_train)

    evaluate(model, X_test, y_test)

    # feature_names = X.columns
    # plot_feature_importances(model, feature_names)


if __name__ == '__main__':
    main()
