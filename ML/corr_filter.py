import numpy as np
import pandas as pd

def filter(x_train,y_train,threshold):
    corr_matrix = np.abs(x_train.corr())
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1,len(corr_matrix.columns)):
            if corr_matrix.iloc[i,j] > threshold:
                high_corr_pairs.append((corr_matrix.columns[i],corr_matrix.columns[j],corr_matrix.iloc[i,j]))

    high_corr_pairs.sort(key=lambda x:x[1],reverse=True)

    features_drop = []

    for feat1,feat2, val in high_corr_pairs:
        if abs(x_train[feat1].corr(y_train)) > abs(x_train[feat2].corr(y_train)):
            features_drop.append(feat2)
        else:
            features_drop.append(feat1)

    x_filtered = x_train.drop(features_drop)

    return x_filtered