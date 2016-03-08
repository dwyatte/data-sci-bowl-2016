import pickle
import pandas as pd
from train import extract_features
from utils import real_to_cdf

if __name__ == '__main__':
    metadata = pd.read_csv('data/metadata_validate.csv')
    features = extract_features(metadata).set_index('Id').sort_index()

    diastole_model = pickle.load(open('diastole.pkl'))
    systole_model = pickle.load(open('systole.pkl'))

    diastole = diastole_model.predict(features)
    systole = systole_model.predict(features)

    systole_cdf = real_to_cdf(systole, sigma=1e-10)
    diastole_cdf = real_to_cdf(diastole, sigma=1e-10)

    submission = pd.DataFrame(columns=['Id'] + ['P%d' % i for i in range(600)])
    i = 0

    for id in range(features.shape[0]):
        diastole_id = '%d_Diastole' % features.index[id]
        systole_id = '%d_Systole' % features.index[id]
        submission.loc[i, :] = [diastole_id] + diastole_cdf[id, :].tolist()
        submission.loc[i+1, :] = [systole_id] + systole_cdf[id, :].tolist()
        i += 2

    submission.to_csv('submission.csv', index=False)
