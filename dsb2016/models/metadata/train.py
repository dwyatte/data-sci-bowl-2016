
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder as _LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from utils import real_to_cdf
from copy import deepcopy

# VALIDATE_PCT = 0.2
np.random.seed(seed=0)

class ColumnExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, column=None):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, df):
        return df[self.column].values.reshape(-1, 1)


class AgeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n = map(lambda x: int(x[0][:-1]), X.tolist())
        t = map(lambda x: x[0][-1], X.tolist())
        years = []
        for n, t in zip(n, t):
            if t == 'M':
                years.append(int(n) / 12.0)
            elif t == 'Y':
                years.append(n)
            elif t == 'W':
                years.append(int(n) / 52.0)
        return np.array(years).reshape(-1, 1)


class LabelEncoder(_LabelEncoder):
    def __init__(self):
        super(LabelEncoder, self).__init__()

    def fit(self, X, y=None):
        return super(LabelEncoder, self).fit(np.ravel(X))

    def transform(self, X):
        Xt = super(LabelEncoder, self).transform(np.ravel(X))
        return Xt.reshape(-1, 1)

    def fit_transform(self, X, y=None, **fit_params):
        Xt = super(LabelEncoder, self).fit_transform(np.ravel(X))
        return Xt.reshape(-1, 1)


features = FeatureUnion([
    ('age', Pipeline([
        ('ce', ColumnExtractor(column='PatientAge')),
        ('at', AgeTransformer())
     ])),
    ('sex', Pipeline([
        ('ce', ColumnExtractor(column='PatientSex')),
        ('le', LabelEncoder()),
        ('oh', OneHotEncoder(sparse=False))
    ]))
])

lr_clf = Pipeline([
    ('pre', StandardScaler()),
    ('grb', LinearRegression())
])

gbr_clf = Pipeline([
    ('pre', StandardScaler()),
    ('grb', GradientBoostingRegressor())
])

dr_clf = Pipeline([
    ('pre', StandardScaler()),
    ('drt', DecisionTreeRegressor())
])

metadata_train = pd.read_csv('data/metadata_train.csv', usecols=['ImagePath', 'PatientAge', 'PatientSex'])
metadata_train['Id'] = map(lambda x: x.split('/')[1], metadata_train['ImagePath'].values)
metadata_train = metadata_train.drop('ImagePath', axis=1)
metadata_train = metadata_train.drop_duplicates().reset_index(drop=True)

train = pd.read_csv('data/train.csv', dtype={'Id': 'object'})
train = pd.merge(metadata_train, train, on='Id')

# shuffle
# train = train.reindex(np.random.permutation(train.index))
# split our train data into a validation set since it actually has labels
# train, validate = train_test_split(train, test_size=VALIDATE_PCT)

X_train = features.fit_transform(train)
y_train_systole = train['Systole'].values
y_train_diastole = train['Diastole'].values
y_train = train[['Systole', 'Diastole']].values

# X_validate = features.fit_transform(validate)
# y_validate_systole = validate['Systole'].values
# y_validate_diastole = validate['Diastole'].values
# y_validate = validate[['Systole', 'Diastole']].values

lr_clf_systole = lr_clf.fit(X_train, y=y_train_systole)
lr_clf_diastole = lr_clf.fit(X_train, y=y_train_diastole)

dr_clf_systole = dr_clf.fit(X_train, y=y_train_systole)
dr_clf_diastole = dr_clf.fit(X_train, y=y_train_diastole)

gbr_clf_systole = deepcopy(gbr_clf).fit(X_train, y=y_train_systole)
gbr_clf_diastole = deepcopy(gbr_clf).fit(X_train, y=y_train_diastole)

# ==== validate

metadata_validate = pd.read_csv('data/metadata_validate.csv', usecols=['ImagePath', 'PatientAge', 'PatientSex'])
metadata_validate['Id'] = map(lambda x: x.split('/')[1], metadata_validate['ImagePath'].values)
metadata_validate = metadata_validate.drop('ImagePath', axis=1)
metadata_validate = metadata_validate.drop_duplicates().reset_index(drop=True)
metadata_validate = metadata_validate.sort_values('Id').reset_index()

clf_systole = gbr_clf_systole
clf_diastole = gbr_clf_diastole

X_validate = features.fit_transform(metadata_validate)
systole = clf_systole.predict(X_validate)
diastole = clf_diastole.predict(X_validate)
systole_cdf = real_to_cdf(systole, sigma=1e-10)
diastole_cdf = real_to_cdf(diastole, sigma=1e-10)

submission_df = pd.DataFrame(columns=['Id'] + ['P'+str(i) for i in range(600)])

submission_df_i = 0
for i in range(metadata_validate.shape[0]):
    id_diastole = metadata_validate.loc[i, 'Id'] + '_Diastole'
    id_systole = metadata_validate.loc[i, 'Id'] + '_Systole'
    submission_df.loc[submission_df_i, :] = [id_diastole] + diastole_cdf[i, :].tolist()
    submission_df.loc[submission_df_i+1, :] = [id_systole] + systole_cdf[i, :].tolist()
    submission_df_i += 2
submission_df[submission_df.columns[1:]] = submission_df[submission_df.columns[1:]].astype('float')

# === ensemble

nn_submission = pd.read_csv('../keras/submission.csv')
nn_mid_x = np.argmax(np.diff(nn_submission.iloc[:, 1:]), axis=1)
our_mid_x = np.argmax(np.diff(submission_df.iloc[:, 1:]), axis=1)
diff = our_mid_x - nn_mid_x

gbr_model_weight = 0.05
shift = np.round(diff * gbr_model_weight)

ensemble_df = nn_submission.copy()

for i in range(ensemble_df.shape[0]):
    orig = ensemble_df.iloc[i, 1:].values.astype('float')
    shift_amt = int(shift[i])
    shifted = np.roll(orig, shift_amt)
    if shift_amt < 0:
        shifted[shift_amt:] = np.max(orig)
    elif shift_amt > 0:
        shifted[:shift_amt] = np.min(orig)
    ensemble_df.iloc[i, 1:] = shifted

