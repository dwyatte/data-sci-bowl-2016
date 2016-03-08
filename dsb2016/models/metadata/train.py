import pickle
import pandas as pd
import numpy as np
from utils import ColumnExtractor
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.cross_validation import train_test_split


TRAIN_PCT = 0.8
PROTO_MODEL = Pipeline([
    ('ce', ColumnExtractor()),
    ('ss', StandardScaler()),
    ('gbr', GradientBoostingRegressor(random_state=0))
])


def extract_age_years(s):
    """
    extracts age in years from age string in format '012Y' or '005M' or '002W'
    """
    t = s[-1]
    if t == 'Y':
        return int(s[:-1])
    elif t == 'M':
        return int(s[:-1]) / 12.0
    elif t == 'W':
        return int(s[:-1]) / 52.0


def extract_features(df):
    """
    core feature extraction function
    """
    # get the Id from ImagePath
    features = pd.DataFrame()
    features['Id'] = df['ImagePath'].apply(lambda x: int(x.split('/')[2]))

    # age/sex
    features['Age'] = df['PatientAge'].apply(extract_age_years)
    sex = pd.get_dummies(df['PatientSex'], prefix='Sex')
    features[sex.columns] = sex

    # interactions
    age_sex = pd.DataFrame(np.array([features['Age'].values * features[col].values for col in sex.columns]).T)
    age_sex.columns = ['Age*'+col for col in sex.columns]
    features[age_sex.columns] = age_sex
    features['Age*Age'] = features['Age'] * features['Age']

    # just use the unique ids
    features = features.drop_duplicates()
    return features


def root_mean_squared_error(y_true, y_pred):
    """
    RMSE loss function
    """
    return np.sqrt(np.mean(np.square(y_pred - y_true)))


def train_model(model, X, y, train_pct):
    """
    Train model
    """
    X_train, X_test = train_test_split(X, test_size=1.0-train_pct, random_state=0)
    y_train, y_test = train_test_split(y, test_size=1.0-train_pct, random_state=0)
    model = clone(model).fit(X_train, y_train)
    if train_pct < 1.0:
        loss = root_mean_squared_error(y_test, model.predict(X_test))
        return model, loss
    else:
        return model, None


def select_features_from_tree(features, tree):
    """
    select features (df) from tree
    """
    return list(features.columns[tree.feature_importances_ > np.mean(tree.feature_importances_)])


def save_model(filename, model):
    """
    Save model (pickle format)
    """
    with open(filename, 'w') as f:
        pickle.dump(model, f)
        print 'Wrote %s' % filename


if __name__ == '__main__':
    metadata = pd.read_csv('data/metadata_train.csv', usecols=['ImagePath', 'PatientAge', 'PatientSex'])
    features = extract_features(metadata).set_index('Id').sort_index()
    outputs = pd.read_csv('data/train.csv', index_col='Id').sort_index()

    diastole_model, diastole_loss = train_model(PROTO_MODEL, features, outputs['Diastole'], TRAIN_PCT)
    systole_model, systole_loss = train_model(PROTO_MODEL, features, outputs['Systole'], TRAIN_PCT)
    if diastole_loss and systole_loss:
        print 'Before feature selection\nDiastole=%f\nSystole=%f\n' % (diastole_loss, systole_loss)

    diastole_columns = list(features.columns[SelectFromModel(diastole_model.steps[-1][-1], prefit=True).get_support()])
    systole_columns = list(features.columns[SelectFromModel(systole_model.steps[-1][-1], prefit=True).get_support()])

    print 'Selected features\nDiastole: %s\nSystole: %s\n' % (diastole_columns, systole_columns)

    diastole_model = diastole_model.set_params(ce__columns=diastole_columns)
    systole_model = systole_model.set_params(ce__columns=systole_columns)

    diastole_model, diastole_loss = train_model(diastole_model, features, outputs['Diastole'], TRAIN_PCT)
    systole_model, systole_loss = train_model(systole_model, features, outputs['Systole'], TRAIN_PCT)
    if diastole_loss and systole_loss:
        print 'After feature selection\nDiastole=%f\nSystole=%f\n' % (diastole_loss, systole_loss)

    save_model('diastole.pkl', diastole_model)
    save_model('systole.pkl', systole_model)
