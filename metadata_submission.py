import csv
import pandas as pd
from utils import real_to_cdf

data = pd.read_csv('predictions.csv')

pred_systole = data['Sys_predict']
pred_diastole = data['Dias_predict']

cdf_pred_systole = real_to_cdf(pred_systole)
cdf_pred_diastole = real_to_cdf(pred_diastole)

#print cdf_pred_systole[0]
#print cdf_pred_systole.shape


header_column = ['Id']
header_column.extend(['P{}'.format(x) for x in range(600)])
#print header_column

with open('submission_test.csv', 'wb') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerow(header_column)
    for i in range(200):
        row_num = i + 501
        row = [str(row_num)+'_Diastole']
        row.extend(cdf_pred_diastole[i])
        csvwriter.writerow(row)
        row = [str(row_num)+'_Systole']
        row.extend(cdf_pred_diastole[i])
        csvwriter.writerow(row)


# write to submission file
#print('Writing submission to file...')
#fi = csv.reader(open('data/sample_submission_validate.csv'))
#f = open('submission.csv', 'w')
#fo = csv.writer(f, lineterminator='\n')
#fo.writerow(fi.next())
#for line in fi:
#    idx = line[0]
#    key, target = idx.split('_')
#    key = int(key)
#    out = [idx]
#    if key in sub_systole:
#        if target == 'Diastole':
#            out.extend(list(sub_diastole[key][0]))
#        else:
#            out.extend(list(sub_systole[key][0]))
#    else:
#        print('Miss {0}'.format(idx))
#    fo.writerow(out)
#f.close()
#
#print('Done.')