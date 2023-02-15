#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(r'..\casper_style.mplstyle')

def generate_gaussian_plot(mu, sigma, size=1000):
    values = np.random.normal(mu, sigma, size)
    fig,ax = plt.subplots()
    sns.distplot(values, ax=ax, hist=False)
    ax.set_xlabel('Values')
    ax.set_title(f'Gaussian centered at {mu} and $\sigma^{2}$ {sigma}')
    fig.show()

# %%
generate_gaussian_plot(1.25, 0.11, 4)
generate_gaussian_plot(1.30, 0.5, 4)

# %%
data = np.array([1.01, 1.30, 1.35, 1.44])
def gaussian_function(x, mu, sigma):
    return (1/np.sqrt(2*np.pi*sigma)) * np.exp(-(x-mu)**2/(2*sigma))

def likelihood(data, function, **kwargs):
    likelihood_value = np.prod(function(data, **kwargs))
    return likelihood_value

def log_likelihood(data, function, **kwargs):
    log_likelihood_value = np.sum(np.log(function(data, **kwargs)))
    return log_likelihood_value

def max_likelihood():
    pass
#%%
print(likelihood(data, function=gaussian_function, mu=1.25,sigma=0.11))
print(likelihood(data, function=gaussian_function, mu=1.30, sigma=0.5))
# %%
print(log_likelihood(data, function=gaussian_function, mu=1.25,sigma=0.11))
print(log_likelihood(data, function=gaussian_function, mu=1.30, sigma=0.5))

# %%
