import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def load_data(file='data/healthcare-dataset-stroke-data.csv'):
    return pd.read_csv(file)


def separate_features(data):
    y = data['stroke']
    X = data.drop(columns=['stroke', 'id'], axis=1)
    return X, y


def handle_missing_values(df):
    df.dropna(inplace=True)
    return df


#todo what to do with glucose
def remove_outliers(df):
    df = df[(df['bmi'] >= 13) & (df['bmi'] <= 55)]
    df = df[(df['gender'] != 'Other')]
    df = df[(df['work_type'] != 'Never_worked')]
    return df


def encode(X):
    encoder = LabelEncoder()
    for column in X.columns:
        if X[column].dtype == 'object':
            X[column] = encoder.fit_transform(X[column])
    return X


def reduction(x_train, x_test):
    pca = PCA(n_components=min(x_train.shape[0], x_train.shape[1]))
    x_train_reduced = pca.fit_transform(x_train)
    x_test_reduced = pca.transform(x_test)
    return x_train_reduced, x_test_reduced


def fit_model(x_train, y_train):
    model = SVC(kernel='linear', random_state=42)
    model.fit(x_train, y_train)
    return model


def evaluate(y_true, y_pred):
    print(classification_report(y_true, y_pred))


def main():
    df = load_data()
    df = handle_missing_values(df)
    df = remove_outliers(df)

    X, y = separate_features(df)
    X = encode(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    #X_train, X_test = reduction(X_train, X_test)
    model = fit_model(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluate(y_test, y_pred)


if __name__ == '__main__':
    main()
