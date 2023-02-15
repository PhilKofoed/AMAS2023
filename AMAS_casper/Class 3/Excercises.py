#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
from matplotlib.ticker import FormatStrFormatter

plt.style.use(r'..\casper_style.mplstyle')
#%%
def generate_gaussian(mu, sigma, size=50):
    values = np.random.normal(mu, sigma, size)
    return values
def gaussian_function(x, mu, sigma):
    return (1/np.sqrt(2*np.pi*sigma)) * np.exp(-(x-mu)**2/(2*sigma**2))

def likelihood(data, function, **kwargs):
    likelihood_value = np.prod(function(data, **kwargs))
    return likelihood_value

def log_likelihood(data, function, **kwargs):
    log_likelihood_value = np.sum(np.log(function(data, **kwargs)))
    return log_likelihood_value

def max_likelihood():
    pass

def raster_scan(data, mu, sigma):
    fig, ax = plt.subplots()
    sns.heatmap(data, ax=ax)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%0.2f'))
    ax.set_xticks([np.min(mu)+(i/5)*(np.max(mu)-np.min(mu)) for i in range(6)],[np.min(mu)+(i/5)*(np.max(mu)-np.min(mu)) for i in range(6)])
    #f'{int(ticks[i]-(bin_width/2))}-{int(ticks[i]+(bin_width/2))}'
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$\sigma$')
    ax.set_title('Raster Plot')

#%%
np.seed = 42
mu_true = 0.2
sigma_true = 0.1
data = generate_gaussian(mu_true, sigma_true)
mu_raster = np.linspace(0.19,0.21, 50)
sigma_raster = np.linspace(0.09,0.11, 50)
#%%
raster_data = np.array([log_likelihood(data, gaussian_function, mu = mu, sigma = sigma) for mu, sigma in itertools.product(mu_raster, sigma_raster)]).reshape(len(mu_raster),len(sigma_raster))
# %%
raster_scan(raster_data, mu_raster, sigma_raster)

# %%
np.max(mu_raster)-np.min(mu_raster)
# %%
[np.min(mu_raster)+(i/5)*(np.max(mu_raster)-np.min(mu_raster)) for i in range(6)]
# %%
