
#%%
import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import httpx
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
    return df
#%%
url_2014 = 'https://kenpom.com/index.php?y=2014'
url_2009 = 'https://kenpom.com/index.php?y=2009'
scraper = cloudscraper.create_scraper()
response_2014 = scraper.get(url_2014)
response_2009 = scraper.get(url_2009)

soup_2014 = BeautifulSoup(response_2014.content, 'html.parser')
soup_2009 = BeautifulSoup(response_2009.content, 'html.parser')

df_2014 = create_df(get_headers(soup_2014),get_data(soup_2014))
df_2009 = create_df(get_headers(soup_2009),get_data(soup_2009))
#df_2014.to_csv(r'C:\Users\caspe\OneDrive\Desktop\AMAS\Assignment 1\kenpom_2014', index=False)
#df_2009.to_csv(r'C:\Users\caspe\OneDrive\Desktop\AMAS\Assignment 1\kenpom_2009', index=False)
#%%
display(df_2014)
#%%
df_2014 = pd.read_csv(r'C:\Users\caspe\OneDrive\Desktop\AMAS_Casper\Assignment 1\kenpom_2014')
df_2009 = pd.read_csv(r'C:\Users\caspe\OneDrive\Desktop\AMAS_Casper\Assignment 1\kenpom_2009')

display(df_2014)

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
# %%
relevant_conferences = ['ACC', 'SEC', 'B10', 'BSky', 'A10']

plt.style.use(r'C:\Users\caspe\OneDrive\Desktop\AMAS_Casper\casper_style.mplstyle')
fig, ax = plt.subplots(figsize=(16,16))
alphas = [1,0.9,0.8,0.7,0.6,0.5]
for conf_key, a in zip(relevant_conferences, alphas):
    ax.hist(df_2014['AdjD'][conference_2014[conf_key]], bins=10, label=f'{conf_key}', alpha=a, linewidth=10)
ax.legend(title='Conference')
ax.set_xlabel('Adjusted Defense Score')
ax.set_ylabel('Counts')
ax.set_title('Adjusted defense for college basketball teams in terms of conference')
# %%
x = {'Conf1': [2,5,7,3], 'Conf2': [59,49,28,57]}
#%%
x['Conf1']
# %%
