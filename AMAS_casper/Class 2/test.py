#%%
import numpy as np
import matplotlib.pyplot as plt
import time
plt.style.use(r'C:\Users\caspe\OneDrive\Desktop\AMAS\casper_style.mplstyle')

#%%
x1 = 10**6
x2 = 10**7
t1_list = []
t2_list = []
for i in range(100):
    t0 = time.time()
    rnd_1 = np.random.uniform(0,1, x1)
    t1 = time.time()
    rnd_2 = np.random.uniform(0,1, x2)
    t2 = time.time()
    total_1 = t1-t0
    total_2 = t2-t1
    t1_list.append(total_1)
    t2_list.append(total_2)
plt.hist(t1_list)
plt.hist(t2_list)
#%%
print(np.mean(t1_list))
print(np.mean(t2_list))
# %%
plt.hist(t1_list)
# %%
x1 = 10**6
x2 = 10**7
t1_list = []
t2_list = []
for i in range(100):
    t0 = time.time()
    rnd_1 = np.random.normal(0,1, x1)
    t1 = time.time()
    rnd_2 = np.random.normal(0,1, x2)
    t2 = time.time()
    total_1 = t1-t0
    total_2 = t2-t1
    t1_list.append(total_1)
    t2_list.append(total_2)
plt.hist(t1_list)
plt.hist(t2_list)
# %%
print(np.mean(t1_list))
print(np.mean(t2_list))
# %%
r = 5.2
x_max, y_max = r, r
x_min, y_min = -r, -r
x_list = []
throws = [10,100,1000,10000,100000]
for i in range(len(throws)):
    pi_list = []
    area_list = []
    for _ in range(10):
        num = throws[i]
        x_rnd = np.random.uniform(x_min,x_max, num)
        y_rnd = np.random.uniform(y_min,y_max, num)
        bool_mask = x_rnd**2+y_rnd**2 <= r**2
        x_rnd = x_rnd[bool_mask]
        y_rnd = y_rnd[bool_mask]
        #fig, ax = plt.subplots(figsize=(10,10))
        #plt.scatter(x_rnd, y_rnd)
        area = ((np.sum(bool_mask)/num)*(2*r)**2)
        pi = area/(r**2)
        pi_list.append(pi)
    x_list.append(pi_list)
means = np.mean(x_list, axis=1)
std = np.std(x_list, axis=1, ddof=1)
fig,ax = plt.subplots(figsize=(12,10))
ax.set(xlabel='# Throws', ylabel='Pi estimate', title='Simulating pi')
ax.set_xscale('log')
ax.errorbar(throws, means, yerr=std, fmt='*', capsize=10, capthick=2,markersize=15, elinewidth=4, ecolor='k', label=f'Pi average')
ax.hlines(np.pi,0, 100000, color='purple', linewidth=1.5, label='Supreme pi')
ax.legend(loc='upper right', fontsize=15, frameon=False)
fig.tight_layout()
#%%
r = 5.2
x_max, y_max = r, r
x_min, y_min = -r, -r
pi_list = []
area_list = []
for _ in range(10000):
    num = 1000
    x_rnd = np.random.uniform(x_min,x_max, num)
    y_rnd = np.random.uniform(y_min,y_max, num)
    bool_mask = x_rnd**2+y_rnd**2 <= r**2
    x_rnd = x_rnd[bool_mask]
    y_rnd = y_rnd[bool_mask]
    #fig, ax = plt.subplots(figsize=(10,10))
    #plt.scatter(x_rnd, y_rnd)
    area = ((np.sum(bool_mask)/num)*(2*r)**2)
    area_list.append(area)
    pi = area/(r**2)
    pi_list.append(pi)


bin_width = np.array([3,1,0.1,1.25])
num_bin = np.rint((np.max(area_list)-np.min(area_list))/bin_width).astype(np.int64)
print(num_bin)
true_bin_width = ((np.max(area_list)-np.min(area_list)))/num_bin
fig,ax = plt.subplots(figsize=(10,12))
ax.set_xlabel(f'Area of circle [m$^{2}$]')
ax.set_ylabel(f'Counts')
ax.set_title('Histogram of circle areas')

for i in range(len(num_bin)):
    ax.hist(area_list, num_bin[i], histtype='step', linewidth=5, label=f'{bin_width[i]:.3f} m$^{2}$\t{num_bin[i]}\t{true_bin_width[i]:.3f} m$^{2}$')
ax.legend(title=f'      Bin width    Num bins   True bin width', fontsize='11', loc='upper left')

fig.show()
# %%
r_rnd = np.random.uniform(0,2*(r**2),num)
