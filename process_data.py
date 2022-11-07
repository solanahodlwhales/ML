import pandas as pd
import pickle, pprint, pathlib, json
import requests
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import plotly.graph_objects as go
import numpy as np

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
    df_all = df_all.assign(total_amount = lambda x: x['price'] / 10**9)
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
    data_dir = pathlib.Path.home() / 'CandyMachineV2' / 'MLNFT' / 'data'
    files = {
        'aurory' : 'aurory.pickle', 'cets_on_creck' : 'cets_on_creck.pickle',
        'degods' : 'degods.pickle', 'galactic_geckos' : 'galactic_geckos.pickle',
        'female_hodl_whales' : 'female_hodl_whales.pickle', 'taiyo_robotics' : 'taiyo_robotics.pickle',
        'stoned_ape_crew' : 'stoned_ape_crew.pickle', 'the_catalina_whale_mixer' : 'the_catalina_whale_mixer.pickle'
    }

    mappings = ['aurory', 'cets_on_creck', 'ggsg:_galactic_geckos', 'catalina_whale_mixer',
                'taiyo_robotics', 'female_hodl_whales', 'degods', 'stoned_ape_crew']

    howrare = pd.read_pickle(str('data/howrare_df.pickle'))

    ## change this filename to process project by project
    filename = 'galacticgeckos'
    with open(str('data/'+filename+'.pickle'), 'rb') as file:
        all_data = pickle.load(file)

    df_merged = all_data['sales']
    df_merged['datetime'] = pd.to_datetime(df_merged['blockTime'], unit = 's')
    df_merged['price'] = df_merged['price'] #/ (10 ** 9)
    df_merged = df_merged.sort_values(by = 'datetime', ignore_index = True)
    df_merged['last_price'] = df_merged.groupby(['mint'])['price'].apply(lambda x: x.shift(1))
    df_merged['price_change'] = df_merged.groupby(['mint'])['price'].apply(lambda x: x.pct_change(1) - 1)
    df_merged['target'] = df_merged.groupby(['mint'])['price_change'].apply(lambda x: x.shift(-1))

    features = ['collectionSymbol', 'mint', 'price', 'seller', 'buyer',
                'datetime', 'rank', 'howrare.is', 'trait_normalized', 'statistical_rarity',
                'price_change', 'last_price', 'target', 'project']
    df = df_merged[features]
    df = df.assign(year = lambda x: x['datetime'].dt.year, month = lambda x: x['datetime'].dt.month,
                   week = lambda x: x['datetime'].dt.isocalendar().week)
    df_filtered = df.copy()
    # # TODO: explore other options
    # # df_filtered[['last_price', 'price_change']] = df_filtered[['last_price', 'price_change']].fillna(0.0)
    df_filtered['total_weekly_volume'] = df_filtered.groupby(['week'])['last_price'].transform(lambda x: x.sum())
    df_filtered['total_monthly_volume'] = df_filtered.groupby(['month'])['last_price'].transform(lambda x: x.sum())
    # # filter outliers
    change = 0.85
    # # get daily calculations
    projects = df_filtered['collectionSymbol'].unique()
    ts_dir = data_dir / 'ts-data'

    df_filtered.to_feather(str('data/'+filename+'_sales_data.feather'))

    df_list = pd.concat([all_data['listings'],
                         all_data['delists']], axis = 0, ignore_index = True)
    df_list['datetime'] = pd.to_datetime(df_list['blockTime'], unit = 's')
    df_list.to_feather(str('data/'+filename+'_listings_data.feather'))

    exit(0) 

    daily_delist = (df_list.groupby(['project'])
                         .resample('D', on = 'datetime')
                         .apply(lambda x: (x['TxType'] == 'cancelEscrow').sum()))
    daily_delist = daily_delist.reset_index(drop = False).rename({0 : 'Delists'}, axis = 1)
    daily_list = (df_list.groupby(['project'])
                         .resample('D', on = 'datetime')
                         .apply(lambda x: (x['TxType'] == 'initializeEscrow').sum()))
    daily_list = (daily_list.reset_index(drop = False)
                            .rename({0 : 'Listings'}, axis = 1))
                            
    # sales = (df_filtered.groupby(['project', 'rank'])
    #                     .resample('D', on = 'datetime').)
    temp = df_filtered.loc[(df_filtered['project'] == 'degods')]
    df_price = temp.groupby(['rank']).resample('W', on='datetime').apply(lambda x: x['total_amount'].mean())
    df_price = df_price.reset_index(drop = False).rename({0 : 'Avg Price'}, axis = 1)
    df_listings = df_list.copy()
    df_list = df_filtered
    df_list = df_list.assign(week = lambda x: x['datetime'].dt.week)
    # degods_list = df_list.loc[df_list['project'] == 'degods']
    week_max = df_list['week'].max()
    degods_list_last = df_list.loc[df_list['week'] == week_max]
    seller_wallet = degods_list_last['seller_address'].unique().tolist()
    degods_wallet_path = pathlib.Path.home() / 'CandyMachineV2' / 'MLNFT' / 'sol-rayz' / 'seller_wallet.json'
    with open(str(degods_wallet_path), 'w') as file:
        json.dump(seller_wallet, file)
    buyer_wallet = degods_list_last['buyer_address'].unique().tolist()
    buyer_wallet_path = pathlib.Path.home() / 'CandyMachineV2' / 'MLNFT' / 'sol-rayz' / 'buyer_wallet.json'
    with open(str(buyer_wallet_path), 'w') as file:
        json.dump(buyer_wallet, file)

    cur_directory = pathlib.Path.home() / 'CandyMachineV2' / 'MLNFT'
    with open(str(cur_directory / 'sol-rayz' / 'seller_wallet_holdings.json'), 'r') as file:
        seller_wallet_json = json.load(file)

    all_mints = set(howrare['mint'].tolist())
    wallets_data = {}
    for k, v in seller_wallet_json.items():
        if len(v) != 0:
            for i in range(len(v)):
                tmp_data = v[i]['data']
                name, symbol = tmp_data['name'], tmp_data['symbol']
                del v[i]['data']
                v[i]['name'], v[i]['symbol'] = name, symbol
            value_df = pd.DataFrame(v)
            value_df['wallet'] = k
            wallets_data[k] = value_df
    wallets_df = pd.concat(list(wallets_data.values()), axis = 0)
    howrare['project_name'] = howrare['link'].apply(lambda x: x.split('/')[-2])

    project_name = howrare['project_name'].unique()
    wallets_df['project_name'] = 'NaN'
    for project in project_name:
        ids = howrare.loc[(howrare['project_name'] == project), 'mint'].tolist()
        wallets_df.loc[(wallets_df['mint'].isin(ids)), 'project_name'] = project

    seller_last_fp = (degods_list_last.groupby(['week', 'project'])
                                      .apply(lambda x: x['total_amount'].min())
                                      .reset_index(drop = False)
                                      .rename({0 : 'floor_price'}, axis = 1))
    wallets_df = wallets_df.merge(seller_last_fp, how = 'left',
                                  left_on = ['project_name'],
                                  right_on = ['project'])
    wallets_df['floor_price'] = wallets_df['floor_price'].fillna(0.0)