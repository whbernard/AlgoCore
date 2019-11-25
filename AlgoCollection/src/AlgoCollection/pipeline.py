#%%
import json
import pandas as pd
import numpy as np

from .algorithms import *


def embedded2dict(embeded, df_category):
    x = embeded[:, 0].tolist()
    y = embeded[:, 1].tolist()
    if embeded.shape[1] == 2:
        z = [0 for _ in range(len(x))] # 2-D
    elif embeded.shape[1] == 3:
        z = embeded[:, 2].tolist() # 3-D
    
    output = {'x': x, 'y': y, 'z': z, 'category': {}}
    for col in df_category:
        output['category'][col] = df_category[col].tolist()

    return output


def dict2json(d):
    # convert to string for now
    # might change to `json.dump` for a file/socket
    return json.dumps(d)


def pick_category(df, percentage=0.3):
    # only pick columns where unique values are less than `percentage` of total number of data
    nrows = df.shape[0]
    criterion = nrows * percentage
    potential_category = []

    for column in df:
        if pd.unique(df[column]).size < criterion:
            potential_category.append(column)

    return df[potential_category]


def process_data_file(filename, **kwargs):
    # read dataset
    df = pd.read_csv(filename)
    # extract algorithm to use
    algorithm = kwargs.pop('algorithm', 'pca')
    # extract to what percentage a column is viewed as a category
    category_percentage = kwargs.pop('category_percentage', 0.3)
    # caculate embeded
    embeded = call_algorithm(df, algorithm, **kwargs)
    # extract category dataframe
    df_category = pick_category(df, category_percentage)
    # construct data struct for visulization
    d = embedded2dict(embeded, df_category)
    # convert to JSON
    return dict2json(d)


def call_algorithm(df, algorithm, **kwargs):
    # only feed `np.number` type of data to `algo`
    X = df.select_dtypes(include=[np.number]).to_numpy()
    if algorithm == 'pca':
        embeded = pca(X, **kwargs)
    elif algorithm == 'tsne':
        embeded = tsne(X, **kwargs)
    elif algorithm == 'umap':
        pass
    
    return embeded
