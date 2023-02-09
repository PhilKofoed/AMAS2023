#%%
import numpy as np
import matplotlib.pyplot as plt
# %%
def pyth(x, y):
    r2 = x**2 + y**2   # Note that r2 = r**2
    return r2
#%%
radii = 5.2
min_range = 0
max_range = radii + 0.1

x_accept = []
y_accept = []
x_reject = []
y_reject = []

iterations = 1000

for i in range(iterations):
    x = np.random.uniform(min_range,max_range)
    y = np.random.uniform(min_range,max_range)
    if pyth(x,y) <= radii**2:
        x_accept.append(x)
        y_accept.append(y)
    else: 
        x_reject.append(x)
        y_reject.append(y)

plt.scatter(x_accept,y_accept)
plt.scatter(x_reject,y_reject)
plt.xlim(min_range,max_range)
plt.ylim(min_range,max_range)

print('The area = ', (len(x_accept)/iterations)*((max_range**2)*4))
# Note: The area is the number of accepted point 
# divided by the number of points total then 
# multiplied by the area we work in (the sqare)

# %%
def circle(radii, min_range, max_range, iterations):
    x_accept = []
    y_accept = []
    x_reject = []
    y_reject = []
    for i in range(iterations):
        x = np.random.uniform(min_range,max_range)
        y = np.random.uniform(min_range,max_range)
        if pyth(x,y) <= radii**2:
            x_accept.append(x)
            y_accept.append(y)
        else: 
            x_reject.append(x)
            y_reject.append(y)
    
    area = (len(x_accept)/iterations)*((max_range**2)*4)
    pi = area/(radii**2)
    return area, pi


# %%

# EXERCISE 2 
area_list = []
for i in range(1000):
    area_i, pi_i = circle(5.2, 0, 5.3, 100)
    area_list.append(area_i)

bin_width = 3
n_bins = int((max(area_list) - min(area_list))/bin_width)
n_bins2 = n_bins*3
n_bins3 = n_bins*10

fig, ax = plt.subplots(figsize=(7, 9))
ax.hist(area_list, bins = n_bins, label='Bin = 3 meter**2',histtype='step', color = 'purple', linewidth=2)
ax.hist(area_list, bins = n_bins2, label='Bin = 0.3 meter**2',histtype='step', color = 'violet', linewidth=2)
ax.hist(area_list, bins = n_bins3, label='Bin = 0.1 meter**2', color = 'pink', linewidth=2)
ax.legend(loc = 'upper left')
# %%
# EXERCISE 3
area_list = []
pi_list = []
iterations_list = [10, 100, 1000, 10000, 100000]

for i in iterations_list:
    area, pi = circle(5.2, 0, 5.3, i)
    pi_list.append(pi)
    area_list.append(area)

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(iterations_list, pi_list,  label='Estimate of pi', color = 'purple', linewidth=2)
ax.legend(loc = 'best')
ax.set_xscale('log')
ax.hlines(np.pi,0,100000, Alpha = 0.5);



# %%
