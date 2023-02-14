
#%%
import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import httpx
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
# %%
# Functions
def get_headers(soup): # Function that gets relevant headers
    headers = []
    results = soup.find(class_='thead2')
    ranks = results.find_all('th')
    for rank in ranks:
        text = rank.get_text()
        headers.append(text)
    headers.remove('AdjEM')
    return headers

def get_data(soup): # Function that gets the relevant data
	results = soup.find("tbody")
	list_of_list = []
	classes_body = ['hard_left', 'next_left', 'conf', 'wl', 'td-left']
	for cl in classes_body:
		list_ranks = []
		ranks = results.find_all("td", class_=cl)
		if cl == 'td-left':
			for rank in ranks:
				text = rank.get_text()
				list_ranks.append(text)
			for i in range(8):
				list_list = [float(i) for i in list_ranks[i::8]]
				list_of_list.append(list_list)
		else:
			for rank in ranks:
				text = rank.get_text()
				list_ranks.append(text)
			list_of_list.append(list_ranks)
	return list_of_list
    
def create_df(headers, data): # Combine headers 
    dictionary = dict(zip(headers, data))
    df = pd.DataFrame(dictionary)
    df['Team'] = df['Team'].str.replace('\d+', '')
    df['Team'] = df['Team'].str.strip()
    return df

def plot_hist(df, header, title, figsize=(8,6), bin_width=10, xlabel=None, ylabel=None, mask=False, mask_name=False):
	if mask:
		data = df[header].iloc[mask].to_numpy(dtype=np.float64)
		num_bin = int(np.ceil((np.max(data)-np.min(data))/bin_width))

		fig, ax = plt.subplots(figsize=figsize)
		bins = np.histogram_bin_edges(df[header].iloc[mask].to_numpy(dtype=np.float64), bins=num_bin)

		sns.histplot(x=header, data=df.iloc[mask], hue=mask_name, hue_order=relevant_conferences, multiple = "dodge", ax=ax, bins=bins)
		
	else:
		data = df[header].to_numpy(dtype=np.float64)
		num_bin = int(np.ceil((np.max(data)-np.min(data))/bin_width))

		fig, ax = plt.subplots(figsize=figsize)
		bins = np.histogram_bin_edges(df[header].to_numpy(dtype=np.float64), bins=num_bin)

		sns.histplot(x=header, data=df, hue=mask_name, multiple = "dodge", ax=ax, bins=bins)

	ticks = [np.min(data)+(i*bin_width/2) for i in np.arange(1,num_bin*2,2)]
	ax.set_xticks(bins)
	ax.tick_params('x',top=True, labeltop=True, bottom=False, labelbottom=False)
	ax.xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))

	tick_names = [f'{int(ticks[i]-(bin_width/2))}-{int(ticks[i]+(bin_width/2))}' for i in range(len(ticks))]
	bin_centers = 0.5 * np.diff(bins) + bins[:-1]
	for tick_name, x in zip(tick_names, bin_centers):
		ax.annotate(tick_name, xy=(x, 0.04), xycoords=('data', 'axes fraction'),
        xytext=(0, -18), textcoords='offset points', va='top', ha='center', fontsize=15)

	ax.set_title(title)
	if xlabel:
		ax.set_xlabel(xlabel)
	if ylabel:
		ax.set_ylabel(ylabel)
	sns.move_legend(ax, loc=(0.01,0.65))
	ax.margins(x=0)
	ax.xaxis.labelpad = 20
	ax.grid(axis='x',linewidth=5, linestyle='-', alpha=1)
	fig.show()

def scrape_html(url, scraper_type='html.parser'):
	scraper = cloudscraper.create_scraper()
	response = scraper.get(url)
	soup = BeautifulSoup(response.content, scraper_type)
	return soup
#%%
url_2014 = 'https://kenpom.com/index.php?y=2014'
url_2009 = 'https://kenpom.com/index.php?y=2009'

soup_2014 = scrape_html(url_2014)
soup_2009 = scrape_html(url_2009)

df_2014 = create_df(get_headers(soup_2014),get_data(soup_2014))
df_2009 = create_df(get_headers(soup_2009),get_data(soup_2009))
#df_2014.to_csv(r'C:\Users\caspe\OneDrive\Desktop\AMAS2023\AMAS_casper\Assignment 1\kenpom_2014', index=False)
#df_2009.to_csv(r'C:\Users\caspe\OneDrive\Desktop\AMAS2023\AMAS_casper\Assignment 1\kenpom_2009', index=False)

#%%
df_2014 = pd.read_csv(r'.\kenpom_2014')
df_2009 = pd.read_csv(r'.\kenpom_2009')

#%%
def sort_as_dict(df, header):
	dictionary = {}
	for i,key in enumerate(df[header]):
		if not key in dictionary:
			dictionary[key] = []
		if key in dictionary:
			dictionary[key].append(i)
	return dictionary
	
conference_2014 = sort_as_dict(df_2014, 'Conf')
conference_2009 = sort_as_dict(df_2009, 'Conf')
relevant_conferences = ['ACC', 'SEC', 'B10', 'BSky', 'A10']
bool_mask_2014 = [conf in np.array(relevant_conferences) for conf in df_2014["Conf"]]
bool_mask_2009 = [conf in np.array(relevant_conferences) for conf in df_2009["Conf"]]

# %%
plt.style.use(r'..\casper_style.mplstyle')

#%%
plot_hist(df=df_2014, header='AdjD', title='College basketball AdjD rating\nSelected conferences 2014', mask=bool_mask_2014, bin_width=7, mask_name='Conf', xlabel='Adjusted Defense Rating')

# %%
plot_hist(df_2009, 'AdjD', 'College basketball AdjD rating\nSelected conferences 2009', mask=bool_mask_2009, bin_width=7,mask_name='Conf', xlabel='Adjusted Defense Rating')
#%%
dict_conf = sort_as_dict(df_2014, 'Conf')

selected_conferences = ['ACC', 'SEC', 'B10', 'BSky', 'A10']
prop_cycle = plt.rcParams['axes.prop_cycle']

xlabel='$X$'; ylabel='$Y$'


Nlayer = 5

fig, axarr = plt.subplots(Nlayer, figsize=(12,20), sharex='col',gridspec_kw={'hspace': 0, 'wspace': 0})

for i,(ax, key) in enumerate(zip(axarr,selected_conferences)):
	histos = df_2014['AdjD'][dict_conf[key]]
	ax.hist(histos, bins=5,linewidth=4.0, color = prop_cycle.by_key()['color'][i], label=f'conference: {selected_conferences[i]}', histtype='step')
	ax.set_ylim(ymin=0, ymax=6.5)
	ax.vlines(x=np.mean(histos), ymin=0, ymax=6.5, color = prop_cycle.by_key()['color'][i], linestyles='-', label=f'mean={np.mean(histos):.1f}')
	ax.vlines(x=np.median(histos), ymin=0, ymax=6.5, color = prop_cycle.by_key()['color'][i], linestyles='-.', label=f'mean={np.median(histos):.1f}')

	ax.legend(loc='upper left', prop={'size':10})

ax.set(xlabel='Adjusted Defense (AdjD)', ylabel='Frequency')
ax.yaxis.set_label_coords(-0.06,Nlayer/2) 
axarr[0].set_title('Adjusted Defense (AdjD) for the five conferences');
# %%
dfmerge = df_2014.merge(df_2009, left_on='Team', right_on='Team',suffixes=('_2014','_2009')).reset_index(drop=True)
dfmerge['AdjO_diff'] = dfmerge['AdjO_2014']-dfmerge['AdjO_2009']
bool_mask_merge = np.array([conf in np.array(relevant_conferences) for conf in dfmerge["Conf_2014"]])
df_merged_selected_conf = dfmerge.iloc[bool_mask_merge,:]
df_merged_rest = dfmerge.iloc[~bool_mask_merge,:]
conference_merged_selected_conf = sort_as_dict(df_merged_selected_conf, 'Conf_2014')

AdjO_diff = np.abs(dfmerge['AdjO_diff'].iloc[bool_mask_merge].to_numpy(dtype=np.float64))

#plt.scatter(dfmerge['AdjO_2009'].iloc[bool_mask_merge],dfmerge['AdjO_2009'].iloc[bool_mask_merge]-dfmerge['AdjO_2014'].iloc[bool_mask_merge])
#%%
mean_rest = np.mean(df_merged_rest['AdjO_diff'])
median_rest = np.median(df_merged_rest['AdjO_diff'])
#%%
conf_mean = {}
conf_median = {}

for i,key in enumerate(df_merged_selected_conf['Conf_2014']):
	if not key in conf_mean:
		conf_mean[key] = []
		conf_median[key] = []

	if key in conf_mean:
		conf_mean[key].append(df_merged_selected_conf['AdjO_diff'].iloc[i])
		conf_median[key].append(df_merged_selected_conf['AdjO_diff'].iloc[i])

for k, v in conf_mean.items():
    conf_mean[k] = np.mean(v)
for k, v in conf_median.items():
    conf_median[k] = np.median(v)

#%%
display(df_merged_selected_conf)
#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=df_merged_selected_conf, x='AdjO_2009', y='AdjO_diff', hue='Conf_2014', hue_order=relevant_conferences, ax=ax, s=200)
ax.set_title('Change in AdjO from 2009-2014 for selected conferences')
ax.set_xlabel('Adjusted Offence Score 2009')
ax.set_ylabel('Adjusted Offence Score change')
fig.show()
#%%
relevant_conferences = ['ACC', 'SEC', 'B10', 'BSky', 'A10', 'BE']
bool_mask_2014 = [conf in np.array(relevant_conferences) for conf in df_2014["Conf"]]
bool_mask_2009 = [conf in np.array(relevant_conferences) for conf in df_2009["Conf"]]

# %%
plt.style.use(r'..\casper_style.mplstyle')

#%%
plot_hist(df_2014, 'AdjD', 'College basketball AdjD rating\nSelected conferences 2014', mask=bool_mask_2014, bin_width=7,mask_name='Conf', xlabel='Adjusted Defense Rating')

# %%
plot_hist(df_2009, 'AdjD', 'College basketball AdjD rating\nSelected conferences 2009', mask=bool_mask_2009, bin_width=7,mask_name='Conf', xlabel='Adjusted Defense Rating')

# %%
dfmerge = df_2014.merge(df_2009, left_on='Team', right_on='Team',suffixes=('_2014','_2009')).reset_index(drop=True)
dfmerge['AdjO_diff'] = dfmerge['AdjO_2014']-dfmerge['AdjO_2009']
bool_mask_merge = np.array([conf in np.array(relevant_conferences) for conf in dfmerge["Conf_2014"]])
df_merged_selected_conf = dfmerge.iloc[bool_mask_merge,:]
df_merged_rest = dfmerge.iloc[~bool_mask_merge,:]
conference_merged_selected_conf = sort_as_dict(df_merged_selected_conf, 'Conf_2014')

AdjO_diff = np.abs(dfmerge['AdjO_diff'].iloc[bool_mask_merge].to_numpy(dtype=np.float64))

#plt.scatter(dfmerge['AdjO_2009'].iloc[bool_mask_merge],dfmerge['AdjO_2009'].iloc[bool_mask_merge]-dfmerge['AdjO_2014'].iloc[bool_mask_merge])
#%%
mean_rest = np.mean(df_merged_rest['AdjO_diff'])
median_rest = np.median(df_merged_rest['AdjO_diff'])
#%%
conf_mean = {}
conf_median = {}

for i,key in enumerate(df_merged_selected_conf['Conf_2014']):
	if not key in conf_mean:
		conf_mean[key] = []
		conf_median[key] = []

	if key in conf_mean:
		conf_mean[key].append(df_merged_selected_conf['AdjO_diff'].iloc[i])
		conf_median[key].append(df_merged_selected_conf['AdjO_diff'].iloc[i])

for k, v in conf_mean.items():
    conf_mean[k] = np.mean(v)
for k, v in conf_median.items():
    conf_median[k] = np.median(v)


#%%
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(data=df_merged_selected_conf, x='AdjO_2009', y='AdjO_diff', hue='Conf_2014', hue_order=relevant_conferences, ax=ax, s=200)
ax.set_title('Change in AdjO from 2009-2014 for selected conferences')
ax.set_xlabel('Adjusted Offence Score 2009')
ax.set_ylabel('Adjusted Offence Score change')
fig.show()


# %%
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    print(dfmerge['AdjO_diff'])
prop_cycle = plt.rcParams['axes.prop_cycle']

fig, ax = plt.subplots(figsize=(8,6))
sns.histplot(dfmerge['AdjO_diff'], alpha=0.7, ax=ax)
for i,mean in enumerate(conf_mean.values()):
	ax.vlines(mean, ymin=0, ymax=70, color = prop_cycle.by_key()['color'][i], label=f'{relevant_conferences[i]} {mean:.1f}',alpha=0.7)
	ax.vlines(list(conf_median.values())[i], ymin=0, ymax=70, color = prop_cycle.by_key()['color'][i], label=f'{relevant_conferences[i]} {list(conf_median.values())[i]:.1f}', linestyles='--',alpha=0.7)

ax.vlines(mean_rest, ymin=0, ymax=70, color ='black', label=f'Remaining {mean_rest:.1f}', alpha=0.7)
ax.vlines(median_rest, ymin=0, ymax=70, color = 'black', label=f'Remaining {median_rest:.1f}',linestyles='--', alpha=0.7)


handles, labels = plt.gca().get_legend_handles_labels()

#specify order of items in legend
order = [0,2,4,6,8,10,1,3,5,7,9,11]

ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol=2, title='Mean                          Median')
# %%
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(list(conf_mean.values())+[mean_rest], list(conf_median.values())+[median_rest], hue=relevant_conferences+['Remaining'], s=500, ax=ax, palette=prop_cycle.by_key()['color']+['k']
)
ax.set_xlabel('Mean')
ax.set_ylabel('Median')
# %%
list(conf_mean.values())+[1]
# %%
prop_cycle = plt.rcParams['axes.prop_cycle'][:5]

# %%
type(plt.rcParams['axes.prop_cycle'])
# %%
prop_cycle.by_key()['color']+['k']
# %%
url_author = 'https://www.nbi.dk/~koskinen/Teaching/AdvancedMethodsInAppliedStatistics2018/data/authors-acknowledgements-v5.pdf'
from bs4 import BeautifulSoup as bs
import requests
import urllib
import pdfkit
# %%
# findAll and find function of BeautifulSoup search for HTML tags
# findAll will return a list with n elements
url_author = 'https://www.nbi.dk/~koskinen/Teaching/AdvancedMethodsInAppliedStatistics2018/data/authors-acknowledgements-v5.pdf'
pd.read_