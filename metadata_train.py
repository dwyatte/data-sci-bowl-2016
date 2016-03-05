from sklearn import linear_model, ensemble
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt


def get_training_data():
    # X
    features = pd.read_csv('age_sex_train.csv')
    lb = LabelEncoder()
    features['sex'] = lb.fit_transform(features['sex'])

    # y
    targets = pd.read_csv('data/train.csv')[['Systole', 'Diastole']]

    return features, targets


def get_validate_data():
    # X
    features = pd.read_csv('age_sex_validate.csv')
    lb = LabelEncoder()
    features['sex'] = lb.fit_transform(features['sex'])

    return features


def linear(features, targets, features_validate):
    # fit model
    clf = linear_model.LinearRegression(normalize=True)
    clf.fit(features, targets)

    # predictions & residuals
    predictions = pd.DataFrame(clf.predict(features), columns=['Sys_predict', 'Dias_predict'])
    actual = targets[:]
    actual.columns = ['Sys_actual', 'Dias_actual']
    residual_df = actual
    residual_df[['Sys_predict', 'Dias_predict']] = predictions[['Sys_predict', 'Dias_predict']]
    residual_df['Sys_residuals'] = residual_df['Sys_predict'] - residual_df['Sys_actual']
    residual_df['Dias_residuals'] = residual_df['Dias_predict'] - residual_df['Dias_actual']

    # make predictions on the validation data
    predictions_validate = pd.DataFrame(clf.predict(features_validate), columns=['Sys_predict', 'Dias_predict'])

    return clf, clf.coef_, clf.score(features, targets), residual_df, predictions_validate


def random_forest(features, targets):
    # fit model
    clf = ensemble.RandomForestRegressor()
    scores = cross_val_score(clf, features, targets)
    print scores


def plot_residuals(residual_df):
    residual_df.plot(kind='scatter', x='Sys_residuals', y='Dias_residuals')
    plt.show()


def write_predictions_to_file(predictions):
    print predictions
    predictions.to_csv('predictions.csv', index=False)


if __name__ == '__main__':
    # get data
    features, targets = get_training_data()
    features_validate = get_validate_data()

    # train a linear model
    model, coef, score, residuals, predictions = linear(features, targets, features_validate)
    plot_residuals(residuals)
    write_predictions_to_file(predictions)

    # train a random forest
    '''
    random_forest(features, targets)'''