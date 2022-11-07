import pickle, pprint, pathlib, json
import requests
import argparse
import pandas as pd
import pathlib
import numpy as np
from train import main, convert_to_dataframe, get_per_hour_stats

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

def encode_categorical_data(df):
    categorical_features = ['seller_address', 'buyer_address', 'mint', 'project']
    for feature in categorical_features:
        df.loc[:, feature + '_cat'] = df[feature].astype('category').cat.codes
    categorical = [c + '_cat' for c in categorical_features]
    return df, categorical


def arg_parse():
    parser = argparse.ArgumentParser(description = 'Player Model')
    parser.add_argument('-production', default = '0', type = int, help = 'production or train')
    parser.add_argument('-date', type = str, default='2021-02-07', help = 'prediction date')
    parser.add_argument('-model', type = str, default = 'regression')
    parser.add_argument('-threshold', type = int, default = '50', help = 'only used if model = total')
    parser.add_argument('-lin_layer_size', nargs='+', type=int, default = [512, 256,128], help = 'linear layers')
    parser.add_argument('-lin_layer_dropout', nargs = '+', type = float, default = [0.3, 0.3, 0.3], help = 'dropout for each lin layer')
    parser.add_argument('-log_dir_folder', default = 'baseline', type = str, help = 'log_dir_folder')
    parser.add_argument('-early_stopping', default = 50, type = int, help = 'early_stopping')
    parser.add_argument('-valid_criteria', default = 'loss', type = str, help = 'accuracy vs loss')
    parser.add_argument('-swa', default = 0, type = int, help = 'swa')
    parser.add_argument('-lr', default = 1e-3, type = float, help = 'lr')
    parser.add_argument('-weighted_sampler', type = int, default = 0, help = 'weighted sampler')
    parser.add_argument('-scheduler', default = 'stepLR', type = str, help = 'cosine_annealing vs stepLR')
    parser.add_argument('-save_output', default = '0', type = int, help = 'save predictions')
    parser.add_argument('-loss_fn', default = 'regression', type = str, help = 'cross_entropy vs gambler_loss')
    parser.add_argument('-batch_size', default = '128', type = int)
    return parser


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

    howrare = pd.read_pickle(str(data_dir / 'howrare_df.pickle'))

    with open(str(data_dir / 'dfs_data.pickle'), 'rb') as file:
        all_data = pickle.load(file)

    df_merged = all_data['sales']
    df_merged['datetime'] = pd.to_datetime(df_merged['blockTime'], unit = 's')
    df_merged['total_amount'] = df_merged['total_amount'] / (10 ** 9)
    df_merged = df_merged.sort_values(by = 'datetime', ignore_index = True)
    df_merged['last_price'] = df_merged.groupby(['mint'])['total_amount'].apply(lambda x: x.shift(1))
    df_merged['price_change'] = df_merged.groupby(['mint'])['total_amount'].apply(lambda x: x.pct_change(1) - 1)
    df_merged['target'] = df_merged.groupby(['mint'])['price_change'].apply(lambda x: x.shift(-1))
    df_merged['target_price'] = df_merged.groupby(['mint'])['total_amount'].apply(lambda x: x.shift(-1))
    features = ['collection_symbol', 'mint', 'total_amount', 'seller_address', 'buyer_address',
                'datetime', 'rank', 'howrare.is', 'trait_normalized', 'statistical_rarity',
                'price_change', 'last_price', 'target', 'project', 'target_price']
    df = df_merged[features]
    df = df.assign(year = lambda x: x['datetime'].dt.year, month = lambda x: x['datetime'].dt.month,
                   week = lambda x: x['datetime'].dt.week)
    df_filtered = df.copy()
    # # TODO: explore other options
    # # df_filtered[['last_price', 'price_change']] = df_filtered[['last_price', 'price_change']].fillna(0.0)
    df_filtered['total_weekly_volume'] = df_filtered.groupby(['week', 'mint'])['last_price'].transform(lambda x: x.sum())
    df_filtered['total_monthly_volume'] = df_filtered.groupby(['month', 'mint'])['last_price'].transform(lambda x: x.sum())
    # # filter outliers
    change = 0.85
    # # get daily calculations
    projects = df_filtered['collection_symbol'].unique()
    ts_dir = data_dir / 'ts-data'

    # df_filtered.to_feather(str(data_dir / 'sales_data.feather'))
    df_list = pd.concat([all_data['listings'],
                         all_data['delists']], axis = 0, ignore_index = True)
    df_list['datetime'] = pd.to_datetime(df_list['blockTime'], unit = 's')
    # df_list.to_feather(str(data_dir / 'listings_data.feather'))

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

    df_train = df_filtered.loc[~df_filtered['target_price'].isna()]
    categorical_features = ['seller_address', 'buyer_address', 'mint', 'project']
    for feature in categorical_features:
        df_train.loc[:, feature + '_cat'] = df_train[feature].astype('category').cat.codes
        df_train.loc[:, feature + '_numeric'] = df_train[feature].astype('category').cat.codes
    categorical = [c + '_cat' for c in categorical_features]
    features = []

    cat_szs = [(df_train[c].nunique() + 1, min(10, (df_train[c].nunique()) // 2)) for c in categorical]

    features = ['rank', 'trait_normalized', 'statistical_rarity',
                'price_change', 'last_price', 'year', 'month', 'week',
                'total_weekly_volume', 'total_monthly_volume', 'total_amount']

    features += [c + '_numeric' for c in categorical_features]

    for feature in features:
        df_train[feature] = df_train[feature].fillna(0)
    n_models = False
    predictions = {}
    source_data = {}
    args = arg_parse()
    train_date = '2022-02-28'
    valid_date = '2022-03-31'
    arg = arg_parse()
    arg = arg.parse_args()

    seed = 1
    X_train_df = df_train.loc[(df_train['datetime'] <= train_date), :].reset_index(drop = True)
    y_train_df = df_train.loc[(df_train['datetime'] <= train_date), 'target'].reset_index(drop=True)

    X_valid_df = df_train.loc[(train_date < df_train['datetime']) & (df_train['datetime'] <= valid_date), :].reset_index(drop=True)
    y_valid_df = df_train.loc[(train_date < df_train['datetime']) & (df_train['datetime'] <= valid_date), 'target'].reset_index(drop=True)

    X_test_df = df_train.loc[(df_train['datetime'] > valid_date), :].reset_index(drop=True)
    y_test_df = df_train.loc[(df_train['datetime'] > valid_date), 'target'].reset_index(drop=True)
    y_test_pred = main(arg, seed, X_train_df, X_valid_df, X_test_df, y_train_df, y_valid_df, y_test_df,
                       features, cat_szs, categorical)

    yp_test, yp_train_pred, yp_valid_pred = y_test_pred
    df_train['ypred'] = np.concatenate([yp_train_pred, yp_valid_pred, yp_test], axis = 0)
    df_train = df_train.loc[df_train['last_price'] != 0, :]
    df_train['last_price'] = df_train.groupby(['mint'])['total_amount'].apply(lambda x: x.shift(1))
    df_train['price_change'] = df_train.groupby(['mint'])['total_amount'].apply(lambda x: x.pct_change(1) - 1)

    floor_price = (df_train.groupby(['datetime', 'collection_symbol'])['total_amount']
                           .apply(lambda x: x.median()).reset_index(drop = False)
                           .rename({0 : 'floor_price'}, axis = 1))
    floor_price_pred = (df_train.groupby(['datetime', 'collection_symbol'])['predicted_price']
                           .apply(lambda x: x.median()).reset_index(drop = False)
                           .rename({0 : 'predicted_price'}, axis = 1))
    df_floor = floor_price.merge(floor_price_pred, how = 'inner',
                                 left_on = ['datetime', 'collection_symbol'],
                                 right_on = ['datetime', 'collection_symbol'])