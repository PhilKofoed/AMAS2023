# %%
import numpy as np
from iminuit import Minuit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("../KD_style.mplstyle")

def mc_ac_new(func, xmin, xmax, ymin, ymax, N_points, **kwargs):
    # Compute accpet/reject Monte Carlo simulations 
    # Written by Philip Kofoed-Djursner
    
    # Initial random points in 2-dimensions
    xran = np.random.uniform(xmin, xmax, N_points)
    yran = np.random.uniform(ymin, ymax, N_points)
    
    # Compute the assosiated y value with the x randomly generated
    yfunc = func(xran, **kwargs)
    
    # Compare random y with computed y. Random need to be less for equal to the computed for it to be accpeted
    bool_mask = yran <= yfunc
    
    # Calculated how many points are missing. Difference between requested and accepted
    missing = N_points - np.sum(bool_mask)
    
    if missing > 0: # If any points are missing request that amount through recursion. this will go on until N_accept = N_points
        xrest, yrest, bool_rest = mc_ac_new(func, xmin, xmax, ymin, ymax, missing, **kwargs)
    else:
        # if missing = 0 return the arrays created. 
        return xran, yran, bool_mask
    # Merge points accpeted and points created through further along recursion
    finalx = np.append(xran, xrest)
    finaly = np.append(yran, yrest)
    finalbool = np.append(bool_mask, bool_rest)
    return finalx, finaly, finalbool

def ullhfit(x, fitfunc, limits=None, fixed=None, **kwargs):
    # Written by Philip Kofoed-Djursner
    # Unbinned log likelihood fitter
    
    def obt(*args):
        # Obtimization function is taken from Troels Petersen
        # Empty results array
        logf = np.zeros_like(x)
        
        # compute the function value
        f = fitfunc(x, *args)
    
        # find where the PDF is 0 or negative (unphysical)        
        mask_f_positive = f > 0

        # calculate the log of f everyhere where f is positive
        logf[mask_f_positive] = np.log(f[mask_f_positive])
        # set everywhere else to badvalue (value should be so bad the fit is never accepted)
        logf[~mask_f_positive] = -1000000
        
        # compute the sum of the log values: the LLH
        llh = -np.sum(logf)
        return llh

    ullh_Min = Minuit(obt, **kwargs, name = [*kwargs]) # Setup; obtimization function, initial valuable guesses, names of variables. 
    ullh_Min.errordef = 0.5 # needed for likelihood fits. No explaination in the documentation.
    if limits:
        for i, limit in enumerate(limits):
            ullh_Min.limits[i] = limit
    if fixed:
        for i, fix in enumerate(fixed):
            ullh_Min.fixed[i] = fix
    ullh_Min.migrad() # Compute the fit
    valuesfit = np.array(ullh_Min.values, dtype = np.float64) # Convert to numpy
    errorsfit = np.array(ullh_Min.errors, dtype = np.float64) # Convert to numpy
    if not ullh_Min.valid: # Give custom error if the fit did not converge
        print("!!! Fit did not converge !!!\n!!! Give better initial parameters !!!")
    # TO DO: 
    # *** Impliment p-value for ullh fit
    return valuesfit, errorsfit

def likelihood(x, func, **args_func):
    logf = np.zeros_like(x)
    
    # compute the function value
    f = func(x, **args_func)

    # find where the PDF is 0 or negative (unphysical)        
    mask_f_positive = f > 0

    # calculate the log of f everyhere where f is positive
    logf[mask_f_positive] = np.log(f[mask_f_positive])
    # set everywhere else to badvalue
    logf[~mask_f_positive] = -1000000
    
    # compute the sum of the log values: the LLH
    llh = -np.sum(logf)
    return llh

def raster_scan_2D(x, func, variables, ranges, points = [100, 100], vmax=None, **kwargs):
    # Written by Philip Kofoed-Djursner
    v1_range = np.linspace(*ranges[0], points[0])
    v2_range = np.linspace(*ranges[1], points[1])
    mesh = np.meshgrid(v1_range, v2_range)
    arguments = {**{key:value for key, value in zip(variables, mesh)}, **kwargs}
    llh = np.zeros((points[0], points[1]))
    for value in x:
        llh -= 2*np.log(func(value, **arguments))
    # llh = np.sum([np.log(func(value, **arguments)) for value in x], axis = 0)
    
    if vmax:
        ax = sns.heatmap(llh, cmap="Spectral", vmax=np.min(llh)+vmax)

    else:
        ax = sns.heatmap(llh, cmap="Spectral")
    ticks = 21
    ax.set_xticks(np.linspace(0, points[0], ticks))
    ax.set_xticklabels(np.around(np.linspace(*ranges[0], ticks),1))
    ax.set_yticks(np.linspace(0, points[1], ticks))
    ax.set_yticklabels(np.around(np.linspace(*ranges[1], ticks),1))
    
    ax.set_xlabel(variables[0])
    ax.set_ylabel(variables[1])
    ax.set_title('Raster Scan')
    plt.gca().invert_yaxis()
    return v1_range, v2_range, llh

# %%
data1 = pd.read_table("./ParameterEstimation_Ex1.txt", sep = "\s+", names=["x"])
data1_num = data1.values
sns.histplot(data1, bins = 30, stat = "density")
print(np.min(data1_num), np.max(data1_num))

xmin = -0.95
xmax = 0.95
xlin = np.linspace(xmin, xmax, 10000)
def teo_func(x, a, b):
    return (1 + a*x + b*x**2) / ((xmax*(6 + 3*a*xmax + 2*b*xmax**2) - xmin*(6 + 3*a*xmin + 2*b*xmin**2))/6)

# %%
values_fit, errors_fit = ullhfit(data1_num, teo_func, a = 0.5, b = 0.5)
print(values_fit, errors_fit)
# %%
sns.histplot(data1_num, bins = 30, stat = "density")
plt.plot(xlin, teo_func(xlin, *values_fit))

a_values = []
b_values = []
for i in range(500):
    x_all,_,mask = mc_ac_new(teo_func, xmin = xmin, xmax = xmax, ymin = 0, ymax = 0.7, N_points=2000, a = values_fit[0], b = values_fit[1])
    x_pseudo = x_all[mask]
    v_fits,_ = ullhfit(x_pseudo, teo_func, a = values_fit[0], b = values_fit[1])
    a_values.append(v_fits[0])
    b_values.append(v_fits[1])
# %%
plt.scatter(a_values, b_values)
plt.show()
plt.hist(a_values, bins = 30)
plt.show()
plt.hist(b_values, bins = 30)
plt.show()
print(f"A value; ori {values_fit[0]:.3f} +/- {errors_fit[0]:.3f}, MC {np.mean(a_values):.3f} +/- {np.std(a_values, ddof = 1):.3f}")
print(f"B value; ori {values_fit[1]:.3f} +/- {errors_fit[1]:.3f}, MC {np.mean(b_values):.3f} +/- {np.std(b_values, ddof = 1):.3f}")
# Integrate up to get the 1 sigma confidence interval
a_val_sorted = sorted(a_values)
b_val_sorted = sorted(b_values)
print(f"A-:{a_val_sorted[round(0.3173/2*500)]:.3f}, A+:{a_val_sorted[-round(0.3173/2*500)]:.3f}")
print(f"B-:{b_val_sorted[round(0.3173/2*500)]:.3f}, B+:{b_val_sorted[-round(0.3173/2*500)]:.3f}")
# %%
xmin = -0.95
xmax = 0.95
xlin = np.linspace(xmin, xmax, 10000)
def teo_func(x, a, b):
    return (1 + a*x + b*x**2) / ((xmax*(6 + 3*a*xmax + 2*b*xmax**2) - xmin*(6 + 3*a*xmin + 2*b*xmin**2))/6)


values_fit, errors_fit = ullhfit(data1_num, teo_func, fixed="a", a = 0.65, b = 0.5)
print(values_fit, errors_fit)
# %%
sns.histplot(data1_num, bins = 30, stat = "density")
plt.plot(xlin, teo_func(xlin, *values_fit))
plt.show()
a_values = []
b_values = []
for i in range(500):
    x_all,_,mask = mc_ac_new(teo_func, xmin = xmin, xmax = xmax, ymin = 0, ymax = 0.7, N_points=2000, a = values_fit[0], b = values_fit[1])
    x_pseudo = x_all[mask]
    v_fits,_ = ullhfit(x_pseudo, teo_func, fixed = "a", a = values_fit[0], b = values_fit[1])
    a_values.append(v_fits[0])
    b_values.append(v_fits[1])

plt.scatter(a_values, b_values)
plt.show()
plt.hist(a_values, bins = 30)
plt.show()
plt.hist(b_values, bins = 30)
plt.show()
print(f"A value; ori {values_fit[0]:.3f} +/- {errors_fit[0]:.3f}, MC {np.mean(a_values):.3f} +/- {np.std(a_values, ddof = 1):.3f}")
print(f"B value; ori {values_fit[1]:.3f} +/- {errors_fit[1]:.3f}, MC {np.mean(b_values):.3f} +/- {np.std(b_values, ddof = 1):.3f}")
# Integrate up to get the 1 sigma confidence interval
a_val_sorted = sorted(a_values)
b_val_sorted = sorted(b_values)
print(f"A-:{a_val_sorted[round(0.3173/2*500)]:.3f}, A+:{a_val_sorted[-round(0.3173/2*500)]:.3f}")
print(f"B-:{b_val_sorted[round(0.3173/2*500)]:.3f}, B+:{b_val_sorted[-round(0.3173/2*500)]:.3f}")
# %%
xmin = -0.9
xmax = 0.85
xlin = np.linspace(xmin, xmax, 10000)
def teo_func(x, a, b):
    return (1 + a*x + b*x**2) / ((xmax*(6 + 3*a*xmax + 2*b*xmax**2) - xmin*(6 + 3*a*xmin + 2*b*xmin**2))/6)


values_fit, errors_fit = ullhfit(data1_num, teo_func, fixed="a", a = 0.65, b = 0.5)
print(values_fit, errors_fit)
# %%
sns.histplot(data1_num, bins = 30, stat = "density")
plt.plot(xlin, teo_func(xlin, *values_fit))
plt.show()
a_values = []
b_values = []
for i in range(500):
    x_all,_,mask = mc_ac_new(teo_func, xmin = xmin, xmax = xmax, ymin = 0, ymax = 0.7, N_points=2000, a = values_fit[0], b = values_fit[1])
    x_pseudo = x_all[mask]
    v_fits,_ = ullhfit(x_pseudo, teo_func, fixed = "a", a = values_fit[0], b = values_fit[1])
    a_values.append(v_fits[0])
    b_values.append(v_fits[1])

plt.scatter(a_values, b_values)
plt.show()
plt.hist(a_values, bins = 30)
plt.show()
plt.hist(b_values, bins = 30)
plt.show()
print(f"A value; ori {values_fit[0]:.3f} +/- {errors_fit[0]:.3f}, MC {np.mean(a_values):.3f} +/- {np.std(a_values, ddof = 1):.3f}")
print(f"B value; ori {values_fit[1]:.3f} +/- {errors_fit[1]:.3f}, MC {np.mean(b_values):.3f} +/- {np.std(b_values, ddof = 1):.3f}")
# Integrate up to get the 1 sigma confidence interval
a_val_sorted = sorted(a_values)
b_val_sorted = sorted(b_values)
print(f"A-:{a_val_sorted[round(0.3173/2*500)]:.3f}, A+:{a_val_sorted[-round(0.3173/2*500)]:.3f}")
print(f"B-:{b_val_sorted[round(0.3173/2*500)]:.3f}, B+:{b_val_sorted[-round(0.3173/2*500)]:.3f}")
# %%
b_tests = np.linspace(0.7, 1.3, 10000)
llhs = []
for b in b_tests:
    llhs.append(likelihood(data1_num, teo_func, a = 0.65, b = b))
# %%
llhs = np.array(llhs)
plt.plot(b_tests, llhs-np.min(llhs))
print(f"Best b:{b_tests[np.argmin(llhs)]:.3f}")
print(f"sigma1 confidence {b_tests[(llhs-np.min(llhs)) <= 0.5][0]:.3f}, {b_tests[(llhs-np.min(llhs)) <= 0.5][-1]:.3f}")
# %%

a_vals, b_vals, llhs = raster_scan_2D(data1_num, teo_func, ["a", "b"], [[0.4,1], [0.3,1.5]], points=[300,300],vmax = 5.92)
# %%
re_llhs = llhs-np.min(llhs)

llhs_plot1 = np.logical_and(re_llhs <= 1.15+0.05, re_llhs >= 1.15-0.05).astype(int)
llhs_plot2 = np.logical_and(re_llhs <= 3.09+0.05, re_llhs >= 3.09-0.05).astype(int)
llhs_plot3 = np.logical_and(re_llhs <= 5.92+0.05, re_llhs >= 5.92-0.05).astype(int)

sns.heatmap(llhs_plot1+llhs_plot2+llhs_plot3)
plt.gca().invert_yaxis()
plt.show()