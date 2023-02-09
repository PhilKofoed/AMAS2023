#%%
import numpy as np
import pandas as pd 

# %%
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

# %%
mean = [np.mean(individual_datasets[i].values[:,1]) for i in individual_datasets.keys()]
print(mean)


# %%
