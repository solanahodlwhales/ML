import pathlib
import pandas as pd
import numpy as np
import torch
from models import model
from sklearn.preprocessing import StandardScaler

def normalize_data(df_train, df_valid, df_test, feature):
    train_scaler = StandardScaler()
    valid_scaler = StandardScaler()
    df_train.loc[:, feature] = train_scaler.fit_transform(df_train[feature].values)
    df_valid.loc[:, feature] = valid_scaler.fit_transform(df_valid[feature].values)
    df_test.loc[:, feature] = valid_scaler.transform(df_test[feature].values)
    return df_train, df_valid, df_test

def main(arg, seed, X_train_df, X_valid_df, X_test_df,
         y_train_df, y_valid_df, y_test_df, features, cat_szs, categorical_features):
    if arg.model == 'total':
        log_dir = pathlib.Path.cwd().parent / 'output' / f'tb_{arg.log_dir_folder}' / f'{arg.date}_{arg.model}_{arg.threshold}_{seed}'
    else:
        log_dir = pathlib.Path.cwd().parent / 'output' / f'tb_{arg.log_dir_folder}' / f'{arg.date}_{arg.model}_{seed}'

    args = {
        'emb_dims': cat_szs,
        'continuous_features': len(features),
        'lin_layer_sizes': arg.lin_layer_size,
        'output_size': 1,
        'embedding_dropout': 0.2,
        'linear_layer_dropout': arg.lin_layer_dropout,
        'lr': arg.lr,
        'use_embedding': True,
        'seed': seed,
        'log_dir': str(log_dir),
        'objective': arg.model,
        'swa': arg.swa,
        'loss_fn' : arg.loss_fn,
        'scheduler': arg.scheduler,
        'batch_size' : arg.batch_size
    }

    if arg.model != 'regression':
        print(f'Proportion of 1 is for train set is : {y_train_df.sum() / y_train_df.shape[0]}')
        print(f'Proportion of 1 is for valid set is : {y_valid_df.sum() / y_valid_df.shape[0]}')
        print(f'Proportion of 1 is for test set is : {y_test_df.sum() / y_test_df.shape[0]}')

        pos_count = y_train_df.sum()
        neg_count = y_train_df.shape[0] - pos_count
        pos_weight = (neg_count / pos_count) * 1.5

        print(f'Pos weight is {pos_weight}')

    X_train_df, X_valid_df, X_test_df = normalize_data(X_train_df, X_valid_df, X_test_df, features)

    cont_train_tensor = torch.from_numpy(X_train_df[features].values).float()
    cat_train_tensor = torch.from_numpy(X_train_df[categorical_features].values.astype('int')).long()

    cont_valid_tensor = torch.from_numpy(X_valid_df[features].values).float()
    cat_valid_tensor = torch.from_numpy(X_valid_df[categorical_features].values.astype('int')).long()

    if args['loss_fn'] == 'regression':
        train_label_tensor = torch.from_numpy(y_train_df.values.astype('float')).view(-1).float()
        valid_label_tensor = torch.from_numpy(y_valid_df.values.astype('float')).view(-1).float()
    else:
        train_label_tensor = torch.from_numpy(y_train_df.values.astype('long')).long().view(-1)
        valid_label_tensor = torch.from_numpy(y_valid_df.values.astype('long')).long().view(-1)

    print(train_label_tensor.size())

    train_dataset = torch.utils.data.TensorDataset(cat_train_tensor,
                                                   cont_train_tensor,
                                                   train_label_tensor)

    valid_dataset = torch.utils.data.TensorDataset(cat_valid_tensor,
                                                   cont_valid_tensor,
                                                   valid_label_tensor)

    # TODO: Check if we need to add weights
    # if arg.model != 'regression':
    #     label_tr = y_train_df.to_frame('label').copy()
    #     label_tr = label_tr.assign(weights=1)
    #     label_tr.loc[label_tr['label'] == 1, 'weights'] = pos_weight
    #     label_val = y_valid_df.to_frame('label').copy()
    #     label_val = label_val.assign(weights=1)
    #     label_val.loc[label_val['label'] == 1, 'weights'] = 1
    #     all_wts = pd.concat([label_tr, label_val], axis=0, ignore_index=True)
    #     all_wts.loc[all_wts['label'] == 1, 'weights'] = pos_weight

    log_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    if arg.weighted_sampler:
        train_sampler = torch.utils.data.WeightedRandomSampler(label_tr['weights'].values.tolist(), label_tr.shape[0],
                                                               replacement=True)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], sampler=train_sampler, drop_last=True,
                                                   pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=True,
                                                   drop_last=True, pin_memory=True)
    else:

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, drop_last=True,
                                                   pin_memory=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=True,
                                                   drop_last=True, pin_memory=True)

    if arg.weighted_sampler:
        epochs = model.train(args, patience=arg.early_stopping, train_dataset=train_loader, valid_dataset=valid_loader,
                             pos_weights=pos_weight)
    else:
        epochs = model.train(args, patience=arg.early_stopping, train_dataset=train_loader, valid_dataset=valid_loader)

    dataset = torch.utils.data.ConcatDataset([train_dataset, valid_dataset])

    if arg.weighted_sampler:
        all_sampler = torch.utils.data.WeightedRandomSampler(all_wts['weights'].values.tolist(), all_wts.shape[0],
                                                             replacement=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], sampler=all_sampler, drop_last=True,
                                                 pin_memory=True)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=True, drop_last=True,
                                                 pin_memory=True)

    if (arg.model != 'regression'):
        if ((pos_count / y_train_df.shape[0]) < 0.3):
            nn_model, loss = model.train(args, patience=arg.early_stopping, num_epochs=epochs, early_stopping=False,
                                         dataset=dataloader, pos_weights=pos_weight)
        else:
            nn_model, loss = model.train(args, patience=arg.early_stopping, num_epochs=epochs, early_stopping=False,
                                         dataset=dataloader)
    else:
        nn_model, loss = model.train(args, patience=arg.early_stopping, num_epochs=epochs, early_stopping=False,
                                     dataset=dataloader)

    cont_test_tensor = torch.from_numpy(X_test_df[features].values).float()
    cat_test_tensor = torch.from_numpy(X_test_df[categorical_features].values).long()
    # y_test_df = pd.get_dummies(y_test_df)
    test_label_tensor = torch.from_numpy(y_test_df.values).view(-1)

    test_dataset = torch.utils.data.TensorDataset(cat_test_tensor,
                                                  cont_test_tensor,
                                                  test_label_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, drop_last=False,
                                              pin_memory=True)
    train_pred_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=False, drop_last=False,
                                                    pin_memory=True)
    valid_pred_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, drop_last=False,
                                                    pin_memory=True)

    y_test_pred, log_preds_test = model.make_predictions(test_loader, nn_model, args)
    y_train_pred, log_preds_train = model.make_predictions(train_pred_loader, nn_model, args)
    y_valid_pred, log_preds_valid = model.make_predictions(valid_pred_loader, nn_model, args)

    return y_test_pred, y_train_pred, y_valid_pred

def convert_to_dataframe(predictions, source_data):
    years = set([key[0] for key, value in predictions.items()])
    seeds = set([key[1] for key, value in predictions.items()])
    all_predictions, all_label, all_X = {}, {}, {}
    for year in years:
        predictions_year = []
        for seed in seeds:
            df = pd.DataFrame(predictions[(year, seed)], columns = [f'y_test_pred_seed_{seed}'])
            df = df.assign(year = year)
            predictions_year.append(df)
        df_label = source_data[year][-1]
        all_label[year] = df_label
        X_test_df = source_data[year][2]
        all_X[year] = X_test_df
        df_year = pd.concat(predictions_year, axis = 1, ignore_index = False)
        all_predictions[year] = df_year
    df_all = pd.concat(list(all_predictions.values()), axis = 0, ignore_index = True)
    X_test_df = pd.concat(list(all_X.values()), axis = 0, ignore_index = True)
    all_label_df = pd.concat(list(all_label.values()), axis = 0, ignore_index = True)
    return df_all, X_test_df, all_label_df


def get_per_hour_stats(df):
    cols = ['+/-', '3P', '3P%', '3PA', 'AST', 'AST%',
            'BLK', 'BPM', 'DRB', 'DRB%', 'DRtg',
            'FG', 'FG%', 'FGA', 'FT', 'FT%', 'ORB', 'ORB%', 'ORtg',
            'FTA', 'PTS', 'STL', 'STL%', 'TOV', 'TOV%', 'TRB', 'TRB%', 'TS%', 'USG%', 'eFG%']
    initial_cols = df.columns.tolist()
    df['min_played'] = df['MP'].fillna('0:00')
    df['tmp_mp'] = np.NaN
    df.loc[~df['MP'].isna(), 'tmp_mp'] = df.loc[~df['MP'].isna(), 'MP'].astype(str).apply(lambda x: x.replace(':', '.'))
    df['tmp_mp'] = df['tmp_mp'].apply(float)
    df.loc[df['tmp_mp'] >= 60, 'min_played'] = '59:00'
    df = df.assign(tmp_date = lambda x: x['date_str'] + ' ' + x['min_played'])
    df['tmp_date'] = pd.to_datetime(df['tmp_date'], format = '%Y-%m-%d %M:%S')
    df['min'] = df['tmp_date'].dt.minute
    df['min'] = pd.to_timedelta(df['min'], unit = 'm')
    df['min_cumsum'] = df.groupby(['name'])['min'].transform(lambda x: x.cumsum())
    df = df.set_index('min_cumsum')
    col_maps = {k : f'{k}_per60m' for k in cols}
    newcols = list(col_maps.values())
    oldcols = list(col_maps.keys())
    initial_cols += newcols
    for oldcol, newcol in zip(oldcols, newcols):
        df.loc[:, newcol] = df.groupby(['name']).rolling('1h')[oldcol].mean()
    df = df.loc[:, df.columns.isin(initial_cols)].reset_index(drop = True)

    # df = df.assign(last_name_let=lambda x: x['name'].str.split(' ')[0][1][0])
    # df[cols] = df[cols].astype(float)
    # listofdfs = []
    # lets = df['last_name_let'].unique()
    # col_maps = {k : f'{k}_per60m' for k in cols}
    # initial_cols += list(col_maps.values())
    # for letter in lets:
    #     print(f'Processing {letter} ... ')
    #     df_tmp = df.loc[(df['last_name_let'] == letter)].reset_index(drop = True)
    #     grouped_df = (df_tmp.groupby([pd.Grouper(key="name"),
    #                                   pd.Grouper(key='min_cumsum', freq='60min')])[cols].apply(lambda x: x.sum()))
    #     grouped_df = grouped_df.reset_index(drop = False)
    #     grouped_df['min_cumsum_end'] = (grouped_df.groupby(['name'])['min_cumsum']
    #                                               .transform(lambda x: x.shift(-1).fillna(method = 'ffill')))
    #     grouped_df.rename(col_maps, axis = 1)
    #     df_tmp['min_cumsum'] = pd.to_timedelta(df_tmp['min_cumsum']) / np.timedelta64(1, 'h')
    #     grouped_df = grouped_df.assign(min_cumsum_end = lambda x: pd.to_timedelta(x['min_cumsum_end']) / np.timedelta64(1, 'h'),
    #                                    min_cumsum = lambda x: pd.to_timedelta(x['min_cumsum']) / np.timedelta64(1, 'h'))
    #     df_tmp = df_tmp.merge(grouped_df, left_on = ['name'], right_on = ['name'])
    #     df_tmp = df_tmp.loc[(df_tmp['min_cumsum_x'] >= df_tmp['min_cumsum_y']) &
    #                         (df_tmp['min_cumsum_x'] <= df_tmp['min_cumsum_end'])].reset_index(drop = True)
    #     listofdfs.append(df_tmp)
    # df_all = pd.concat(listofdfs, axis = 0, ignore_index = True)
    return df
