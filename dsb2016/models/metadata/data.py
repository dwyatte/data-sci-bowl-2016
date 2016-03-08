import os, glob
import pandas as pd
import dicom


METADATA = ['AcquisitionMatrix',
            'AcquisitionNumber',
            'AcquisitionTime',
            'AngioFlag',
            'BitsAllocated',
            'BitsStored',
            'BodyPartExamined',
            'CardiacNumberOfImages',
            'Columns',
            'CommentsOnThePerformedProcedureStep',
            'EchoNumbers',
            'EchoTime',
            'EchoTrainLength',
            'FlipAngle',
            'HighBit',
            'ImageOrientationPatient',
            'ImagePositionPatient',
            'ImageType',
            'ImagedNucleus',
            'ImagingFrequency',
            'InPlanePhaseEncodingDirection',
            'InstanceCreationTime',
            'InstanceNumber',
            'LargestImagePixelValue',
            'MRAcquisitionType',
            'MagneticFieldStrength',
            'Manufacturer',
            'ManufacturerModelName',
            'Modality',
            'NominalInterval',
            'NumberOfAverages',
            'NumberOfPhaseEncodingSteps',
            'PatientAddress',
            'PatientAge',
            'PatientBirthDate',
            'PatientID',
            'PatientName',
            'PatientPosition',
            'PatientSex',
            'PatientTelephoneNumbers',
            'PercentPhaseFieldOfView',
            'PercentSampling',
            'PerformedProcedureStepID',
            'PerformedProcedureStepStartTime',
            'PhotometricInterpretation',
            'PixelBandwidth',
            'PixelRepresentation',
            'PixelSpacing',
            'PositionReferenceIndicator',
            'RefdImageSequence',
            'ReferencedImageSequence',
            'RepetitionTime',
            'Rows',
            'SAR',
            'SOPClassUID',
            'SOPInstanceUID',
            'SamplesPerPixel',
            'ScanOptions',
            'ScanningSequence',
            'SequenceName',
            'SequenceVariant',
            'SeriesDescription',
            'SeriesNumber',
            'SeriesTime',
            'SliceLocation',
            'SliceThickness',
            'SmallestImagePixelValue',
            'SoftwareVersions',
            'SpecificCharacterSet',
            'StudyTime',
            'TransmitCoilName',
            'TriggerTime',
            'VariableFlipAngleFlag',
            'WindowCenter',
            'WindowCenterWidthExplanation',
            'WindowWidth',
            'dBdt']


def extract_metadata(path, filename):
    """
    get the age and sex of each patient, add it to the data list
    """

    i = 0
    patient_folders = glob.glob(os.path.join(path, '*'))

    for folder in patient_folders:
        image_folders = glob.glob(os.path.join(folder, 'study', '*'))
        for image_folder in image_folders:
            image_files = glob.glob(os.path.join(image_folder, '*'))
            for image_file in image_files:
                data = pd.DataFrame(columns=METADATA + ['ImagePath'])
                image = dicom.read_file(image_file)
                for attr in METADATA:
                    if hasattr(image, attr):
                        data.loc[0, attr] = getattr(image, attr)
                data.loc[0, 'ImagePath'] = image_file
                with open(filename, 'a') as f:
                    if i==0:
                        data.to_csv(f, header=True, index=False)
                    else:
                        data.to_csv(f, header=False, index=False)
                i += 1
                if i % 100==0:
                    print i, image_file

if __name__ == '__main__':
    extract_metadata('data/train', 'data/metadata_train.csv')
    extract_metadata('data/validate', 'data/metadata_validate.csv')