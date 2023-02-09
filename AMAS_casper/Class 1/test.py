#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
#Reading in the data:
data = pd.read_table('FranksNumbers.txt',header=1,sep='\s+',
                    names = ['x','y_measured','to_be_dropped'])

#The last column now contains NaN values, which is why we drop it:
data = data.drop(['to_be_dropped'],axis=1)

#Next, we will check if the first symbol, data['x'][i][0], in each data row [i] is a digit.
check_if_numerical = np.array([data['x'][i][0].isdigit() for i in range(len(data))])

#If the above test was failed and the first symbol is not a digit, it must be a letter (in this case).
#This means that we found a header for the next dataset. Recording the indices of all header rows in this manner:
header_rows = np.where(check_if_numerical == False)[0]

#We will also append -1 from above, since this is technically where the 1st header belongs.
breaking_rows = np.append(-1,header_rows)
breaking_rows = np.append(breaking_rows,len(data))

#We will store the individual datasets in the dictionary below, starting with the row of the previous header
#and ending with the row of the next header.
individual_datasets = {}

for dataset_number in range(len(header_rows)+1):
    #We will also remember to change the format to float (as it used to be string before)
    individual_datasets[dataset_number+1] = \
    data[breaking_rows[dataset_number]+1:breaking_rows[dataset_number+1]].astype('float')
print(individual_datasets[1].values[:,1])

# %%
y_mean = [np.mean(individual_datasets[y].values[:,1]) for y in individual_datasets.keys()]
y_variance = [np.var(individual_datasets[y].values[:,1]) for y in individual_datasets.keys()]
for y in individual_datasets.keys():
    fig = plt.scatter(individual_datasets[y].values[:,0], individual_datasets[y].values[:,1])
    plt.show()

def chi2(y_expected, y_measured, y_error):
    return np.sum((y_expected-y_measured)**2/y_error**2)

y_expected = {y: individual_datasets[y].values[:,0]*0.48 + 3.02 for y in individual_datasets.keys()}
for data_set in individual_datasets.keys():

    chi_2_sqrt = chi2(y_expected[data_set], individual_datasets[data_set].values[:,1],np.sqrt(individual_datasets[data_set].values[:,1]))
    chi_2_fix = chi2(y_expected[data_set], individual_datasets[data_set].values[:,1], 1.22)

# %%
