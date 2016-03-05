import os, glob, csv
import pandas as pd
import pydicom


def get_data(train_or_validate):
    data_path = 'data/{}'.format(train_or_validate)

    patient_folders = glob.glob(os.path.join(data_path, '*'))

    # data headers
    data = [['patient', 'age', 'sex']]

    count = 0

    # get the age and sex of each patient, add it to the data list
    for folder in patient_folders:
        image_folders = glob.glob(os.path.join(folder, 'study', '*'))
        img_folder = image_folders[0]
        images = glob.glob(os.path.join(img_folder, '*'))
        image = images[0]
        image_data = pydicom.read_file(image)
        data.append([os.path.split(folder)[1], int(image_data.PatientAge[:-1]), image_data.PatientSex])
        count += 1
        print count

    # sort the data list by patient number, then save to a csv file
    df = pd.DataFrame(data[1:], columns=data[0])
    df['patient'] = df['patient'].astype(int)
    df = df.sort_values(by='patient')
    df = df[['age', 'sex']]
    df.to_csv('age_sex_{}.csv'.format(train_or_validate), index=False)


if __name__ == '__main__':
    #get_data('train')
    get_data('validate')