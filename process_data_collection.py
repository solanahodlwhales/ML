import pandas as pd
import pickle, pprint, pathlib, json
import requests
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def get_howrare():
    url = 'https://api.howrare.is/v0.1/collections'
    request = requests.get(url)
    request = request.json()
    return request

def flatten_data(data):
    for i in range(len(data)):
        item = data[i]
        key = None
        for k, v in item.items():
            if type(v) is dict:
                key = k
                break
        if key is not None:
            item.update(item[key])
            del item[key]
        data[i] = item
    return data

def process_how_rare_data():
    howrare = get_howrare()
    howrare_data = howrare['result']['data']
    howrare_df = pd.DataFrame(howrare_data)
    howrare_df['name'] = howrare_df['name'].apply(lambda x: x.lower().replace(' ', '_'))
    return howrare_df

def get_collection_data(data_dir, fname, collection_name):
    with open(str(data_dir / fname), 'rb') as file:
        tx_data = pickle.load(file)
    dfs = {}
    result = tx_data['transactions']
    for hash_id, value in result.items():
        results = []
        for res in value.get('results', []):
            if 'parsedTransaction' in res:
                results.append(res['parsedTransaction'])
        if len(results) != 0:
            temp = pd.DataFrame(results)
            dfs[hash_id] = temp
    df_all = pd.concat(list(dfs.values()), axis = 0)
    df_all['datetime'] = pd.to_datetime(df_all['blockTime'], unit = 's')
    df_all = df_all.assign(total_amount = lambda x: x['total_amount'] / 10**9)
    df_all['collections'] = collection_name
    return df_all

def get_howrare_collection(url):
    base = f'https://api.howrare.is/v0.1/collections/{url}/only_rarity'
    collection_rarity = requests.get(base)
    rarity = collection_rarity.json()
    rarity_data = rarity['result']['data']['items']
    rarity_data = flatten_data(rarity_data)
    rarity_df = pd.DataFrame(rarity_data)
    return rarity_df


if __name__ == '__main__':
    data_dir = pathlib.Path.cwd() / 'data'
    
    # howrare_data = process_how_rare_data()
    # howrare_data = howrare_data.sort_values(by = ['floor_marketcap'], ascending = False, inplace = False)
    # howrare_data.to_excel(str(data_dir / 'howrare_data_project_level.xlsx'))

    # mappings = ['aurory', 'cets_on_creck', 'ggsg:_galactic_geckos', 'catalina_whale_mixer',
    #             'taiyo_robotics', 'female_hodl_whales', 'degods', 'stoned_ape_crew']
    # mappings = ['solana_hodl_whales']
    howrare_df = pd.read_excel(str(data_dir / 'howrare_data_selected_projects.xlsx'))
    # howrare_data = howrare_df.loc[howrare_df['name'].isin(mappings)]
    howrare_data = howrare_df.loc[~howrare_df['Magic Eden URL'].isna(), :]
    howrare_data['url'] = howrare_data['url'].apply(lambda x: x.replace('/', ''))
    urls = howrare_data['url'].tolist()
    howrare_dfs = []
    for url in urls:
        rarity_df = get_howrare_collection(url)
        howrare_dfs.append(rarity_df)
    howrare_df = pd.concat(howrare_dfs, axis = 0)
    # output_file = 'howrare_df_mint_id.pickle'
    # howrare_df.to_pickle(str(data_dir / output_file))