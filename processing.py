import numpy as np
import pandas as pd


def preprocessing(path = './data/data_regression_for_task.csv', encoding_categorical=None):
    """ Data preprocessing, which includes type conversion, missing value processing,
    features engineering, and  deleting uninformative data.
    Args:
        path: path to the file with data 
        encoding_categorical: list of categorical data need to be encoded
    Returns:
        data: preprocessed data
    """    
    data = pd.read_csv(path)
    # type conversion
    data['SALES'] = data['SALES'].astype('int64')
    data['STORE_SALES'] = data['STORE_SALES'].astype('int64')
    # fill in the only missing value in ARTICLE_GROUP
    data.loc[data['ARTICLE_GROUP'].isnull(),'ARTICLE_GROUP'] = 'WINE'
    # delete the rows with missing values in CONTRAGENT
    data.drop(data[data['CONTRAGENT'].isnull()].index, inplace=True)
    # bring the codes and names into correspondence
    count_names = data.groupby('ARTICLE_CODE')['ARTICLE_NAME'].unique()
    count_names = dict(count_names.apply(lambda x: x[0]))
    data['ARTICLE_NAME'] = data['ARTICLE_CODE'].map(count_names)
    data = data.groupby(['YEAR', 'MONTH', 'CONTRAGENT', 'ARTICLE_NAME',
             'ARTICLE_GROUP'])[['SALES', 'STORE_SALES']].sum().reset_index()
    # new time based feature
    data['YEAR_MONTH'] = data['YEAR'].astype('str') + '_' \
                         + data['MONTH'].astype('str').apply(lambda x: x.zfill(2))
    ym_to_int = dict(zip(sorted(np.append(data['YEAR_MONTH'].unique(), '2017_07')), np.arange(11)))
    data['YEAR_MONTH'] = data['YEAR_MONTH'].map(ym_to_int)
    # features engineering
    for timestep in range(1,4):
        for gb in ['CONTRAGENT', 'ARTICLE_NAME', 'ARTICLE_GROUP']:
            for col in ['SALES','STORE_SALES']:
                data = add_time_features(data, groupby=gb, col=col, timestep=timestep)
    # remove the uninformative category
    data.drop(data[data['ARTICLE_GROUP'] == 'DUNNAGE'].index, inplace=True)
    # delete the first month with only zero sales
    data.drop(data[data['YEAR_MONTH'] == 0].index, inplace=True)
    # encoding categorical features
    if encoding_categorical:
        data = pd.get_dummies(data, columns=encoding_categorical)
    # updating indexes
    data.reset_index(inplace=True)
    del data['index']
    
    return data
    
    
def add_time_features(data, groupby, col, timestep=1):
    """ Create lag features for timeseries dataset.
    
    Args:
        data: input dataframe with columns for grouping and lagging
        groupby: a column by which we perform grouping (product name, category, etc.)
        col: a taregt column for creating lags
        timestep: lag shift expressed in time units (int, default = 1)
    Returns:
        data: data with lag features
    """
    data['NEXT_YEAR_MONTH'] = data['YEAR_MONTH'] + timestep 
    agent_sales = data.groupby([groupby,'NEXT_YEAR_MONTH']) \
                       [col].agg([np.mean, np.sum]).reset_index()
    agent_sales.rename(columns={'NEXT_YEAR_MONTH':'YEAR_MONTH',
                                    'mean':'GB_' + groupby + '_PREV_MEAN_' + col + '_' + str(timestep),
                                    'sum':'GB_' + groupby + '_PREV_SUM_' + col + '_' + str(timestep)
                                   }, inplace=True)
    data = pd.merge(data, agent_sales, how='left',
             on=[groupby,'YEAR_MONTH']).sort_values([groupby,'NEXT_YEAR_MONTH'])
    data['GB_' + groupby + '_PREV_MEAN_' + col + '_' + str(timestep)].fillna(0, inplace=True)
    data['GB_' + groupby + '_PREV_SUM_' + col + '_' + str(timestep)].fillna(0, inplace=True)
    del data['NEXT_YEAR_MONTH']
    return data


 