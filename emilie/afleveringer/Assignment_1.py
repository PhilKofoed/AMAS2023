#%%
import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from matplotlib.ticker import FormatStrFormatter
# %%

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
    return df


#%%
# Code for scraping from the internet 
scraper = cloudscraper.create_scraper()

# DATA FROM 2014
response14 = scraper.get("https://kenpom.com/index.php?y=2014")
soup = BeautifulSoup(response14.content, features="html.parser")  #soup 2012
df_2014 = create_df(get_headers(soup),get_data(soup))

# DATA FROM 2009
response9 = scraper.get("https://kenpom.com/index.php?y=2014")
soup9 = BeautifulSoup(response9.content, features="html.parser")  #soup 2012
df_2009 = create_df(get_headers(soup9),get_data(soup9))


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

conferences = ['ACC', 'SEC', 'B10', 'BSky', 'A10']

# %%
# EXERCISE 1

conferences = ['ACC', 'SEC', 'B10', 'BSky', 'A10']

ACC_arr = df_2014['AdjD'][conference_2014['ACC']]
SEC_arr = df_2014['AdjD'][conference_2014['SEC']]
B10_arr = df_2014['AdjD'][conference_2014['B10']]
BSky_arr = df_2014['AdjD'][conference_2014['BSky']]
A10_arr = df_2014['AdjD'][conference_2014['A10']]



# %%
def plot_hist(df, header, title, figsize=(8,6), bin_width=10, xlabel=None, ylabel=None, mask=False, mask_name=False):
	if mask:
		data = df[header].iloc[mask].to_numpy(dtype=np.float64)
		num_bin = int(np.ceil((np.max(data)-np.min(data))/bin_width))

		fig, ax = plt.subplots(figsize=figsize)
		bins = np.histogram_bin_edges(df[header].iloc[mask].to_numpy(dtype=np.float64), bins=num_bin)

		sns.histplot(x=header, data=df.iloc[mask], hue=mask_name, hue_order=conferences, multiple = "dodge", ax=ax, bins=bins)
		
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
# %%

bool_mask_2014 = [conf in np.array(conferences) for conf in df_2014["Conf"]]
bool_mask_2009 = [conf in np.array(conferences) for conf in df_2009["Conf"]]


plot_hist(df_2014, 'AdjD', 'RANDOM', mask=bool_mask_2014, bin_width=7, mask_name='Conf')
# %%
