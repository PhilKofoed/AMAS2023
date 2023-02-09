#%%
import numpy as np
import matplotlib.pyplot as plt
import time
# %%
n1 = 10**6
n2 = 10**7
tstart = time.time()

ran1 = np.random.uniform(0,1,n1)
t1 = time.time()
ran2 = np.random.uniform(0,1,n2)
t2 = time.time()

time_1 = t1-tstart
time_2 = t2 - t1

print('Generating',n1,'random numbers took',time_1,'sec')
print('Generating',n2,'random numbers took',time_2,'sec')


# %%
