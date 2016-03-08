"""
Simple ensemble from keras model and metadata. Shifts keras CDFs toward metadata CDFs according to METADATA_WEIGHT
"""

import numpy as np
import pandas as pd

METADATA_WEIGHT = 0.1

keras_submission = pd.read_csv('models/keras/submission.csv')
keras_submission.sort_values('Id').reset_index(drop=True)

metadata_submission = pd.read_csv('models/metadata/submission.csv')
metadata_submission.sort_values('Id').reset_index(drop=True)

# midpoints for cdfs
keras_mid = np.argmax(np.diff(keras_submission.iloc[:, 1:]), axis=1)
metadata_mid = np.argmax(np.diff(metadata_submission.iloc[:, 1:]), axis=1)
shift = np.round((metadata_mid-keras_mid) * METADATA_WEIGHT)

# shift
ensemble_submission = keras_submission.copy()
for i in range(ensemble_submission.shape[0]):
    orig = ensemble_submission.iloc[i, 1:].values.astype('float')
    shift_x = int(shift[i])
    shifted = np.roll(orig, shift_x)
    if shift_x < 0:
        shifted[shift_x:] = np.max(orig)
    elif shift_x > 0:
        shifted[:shift_x] = np.min(orig)
    ensemble_submission.iloc[i, 1:] = shifted

filename = 'ensemble_%.1f_metadata.csv' % METADATA_WEIGHT
ensemble_submission.to_csv(filename, index=False)

