import scipy.io 
import numpy as np 
import os
os.chdir('/Users/sumanyu/Desktop/RL/Action Recognition/Dataset/')

print(os.getcwd())

data = scipy.io.loadmat("mpii_labels.mat")
for i in data:
	if '__' not in i and 'readme' not in i:
		np.savetxt(("mpii_labels_csv/"+i+".csv"),data[i],fmt = '%s',delimiter=',')