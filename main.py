import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
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
    return X_resampled, y_resampled


def encode(X):
    encoder = LabelEncoder()
    for column in X.columns:
        if X[column].dtype == 'object':
            X[column] = encoder.fit_transform(X[column])
    return X


def fit_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced_subsample')
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


if __name__ == '__main__':
    main()
