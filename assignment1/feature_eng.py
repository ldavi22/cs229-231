import numpy as np
import pandas as pd

def preprocess(x_train, x_test):
    x_train = x_train.copy()
    x_test = x_test.copy()

    # --- Separate target ---
    y_train = np.log1p(x_train['SalePrice'])
    y_test = np.log1p(x_test['SalePrice'])
    x_train.drop('SalePrice', axis=1, inplace=True)
    x_test.drop('SalePrice', axis=1, inplace=True)

    # --- Drop Id ---
    x_train.drop('Id', axis=1, inplace=True)
    x_test.drop('Id', axis=1, inplace=True)

    # --- MasVnrType / MasVnrArea ---
    for df in [x_train, x_test]:
        df.loc[df['MasVnrType'].isna() & (df['MasVnrArea'] == 0), 'MasVnrType'] = 'None'
        df.loc[df['MasVnrType'].isna() & df['MasVnrArea'].isna(), 'MasVnrArea'] = 0
        df.loc[df['MasVnrType'].isna() & df['MasVnrArea'].isna(), 'MasVnrType'] = 'None'
        df.loc[df['MasVnrType'].isna() & (df['MasVnrArea'] > 0), 'MasVnrType'] = 'BrkFace'
        df.loc[(df['MasVnrType'] == 'None') & df['MasVnrArea'].isna(), 'MasVnrArea'] = 0

    # --- Basement NAs → 'None' where no basement ---
    bsmt_cat_cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
    for df in [x_train, x_test]:
        for col in bsmt_cat_cols:
            df.loc[(df['TotalBsmtSF'] == 0) & df[col].isna(), col] = 'None'

    # --- Garage NAs → 'None' where no garage ---
    garage_cat_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
    for df in [x_train, x_test]:
        for col in garage_cat_cols:
            df.loc[(df['GarageArea'] == 0) & df[col].isna(), col] = 'None'

    # --- Drop GarageYrBlt (0.82 corr with YearBuilt) ---
    x_train.drop('GarageYrBlt', axis=1, inplace=True)
    x_test.drop('GarageYrBlt', axis=1, inplace=True)

    # --- Drop PoolQC and PoolArea (insufficient variance) ---
    x_train.drop(['PoolQC', 'PoolArea'], axis=1, inplace=True)
    x_test.drop(['PoolQC', 'PoolArea'], axis=1, inplace=True)

    # --- FireplaceQu: NA → 'None' where no fireplace ---
    for df in [x_train, x_test]:
        df.loc[(df['Fireplaces'] == 0) & df['FireplaceQu'].isna(), 'FireplaceQu'] = 'None'

    # --- MiscFeature, Alley, Fence: NA → 'None' ---
    for df in [x_train, x_test]:
        for col in ['MiscFeature', 'Alley', 'Fence']:
            df.loc[df[col].isna(), col] = 'None'

    # --- LotFrontage: median imputation (fit on train) ---
    frontage_median = x_train['LotFrontage'].median()
    x_train['LotFrontage'].fillna(frontage_median, inplace=True)
    x_test['LotFrontage'].fillna(frontage_median, inplace=True)

    # --- Electrical: mode imputation (fit on train) ---
    electrical_mode = x_train['Electrical'].mode()[0]
    x_train['Electrical'].fillna(electrical_mode, inplace=True)
    x_test['Electrical'].fillna(electrical_mode, inplace=True)

    # --- Remaining NAs in basement (genuinely missing) → mode from train ---
    for col in bsmt_cat_cols:
        if x_test[col].isna().sum() > 0:
            mode_val = x_train[col].mode()[0]
            x_test[col].fillna(mode_val, inplace=True)
        if x_train[col].isna().sum() > 0:
            mode_val = x_train[col].mode()[0]
            x_train[col].fillna(mode_val, inplace=True)

    # --- Quality ordinal encoding (shared scale) ---
    quality_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    quality_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                    'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageCond', 'GarageQual']
    for col in quality_cols:
        x_train[col] = x_train[col].map(quality_map)
        x_test[col] = x_test[col].map(quality_map)

    # --- Other ordinal encodings ---
    ordinal_maps = {
        'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
        'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
        'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
        'LotShape':     {'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4},
        'LandSlope':    {'Sev': 1, 'Mod': 2, 'Gtl': 3},
        'PavedDrive':   {'N': 1, 'P': 2, 'Y': 3},
        'Utilities':    {'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4},
        'Functional':   {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8},
        'Electrical':   {'Mix': 1, 'FuseP': 2, 'FuseF': 3, 'FuseA': 4, 'SBrkr': 5},
        'Fence':        {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4},
    }
    for col, mapping in ordinal_maps.items():
        x_train[col] = x_train[col].map(mapping)
        x_test[col] = x_test[col].map(mapping)

    # --- Engineer HasMultipleExterior, drop Exterior2nd ---
    x_train['HasMultipleExterior'] = (x_train['Exterior1st'] != x_train['Exterior2nd']).astype(int)
    x_test['HasMultipleExterior'] = (x_test['Exterior1st'] != x_test['Exterior2nd']).astype(int)
    x_train.drop('Exterior2nd', axis=1, inplace=True)
    x_test.drop('Exterior2nd', axis=1, inplace=True)

    # --- Engineer HasMultipleCondition, drop Condition2 ---
    x_train['HasMultipleCondition'] = (x_train['Condition1'] != x_train['Condition2']).astype(int)
    x_test['HasMultipleCondition'] = (x_test['Condition1'] != x_test['Condition2']).astype(int)
    x_train.drop('Condition2', axis=1, inplace=True)
    x_test.drop('Condition2', axis=1, inplace=True)

    # --- Binary encoding ---
    x_train['Street'] = x_train['Street'].map({'Grvl': 0, 'Pave': 1})
    x_test['Street'] = x_test['Street'].map({'Grvl': 0, 'Pave': 1})
    x_train['CentralAir'] = x_train['CentralAir'].map({'N': 0, 'Y': 1})
    x_test['CentralAir'] = x_test['CentralAir'].map({'N': 0, 'Y': 1})

    # --- One-hot encode nominal features ---
    # --- One-hot encode nominal features (combined to ensure consistent encoding) ---
    nominal_cols = ['MSSubClass', 'MSZoning', 'Alley', 'LandContour', 'LotConfig',
                    'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle',
                    'RoofStyle', 'RoofMatl', 'Exterior1st', 'MasVnrType',
                    'Foundation', 'Heating', 'GarageType', 'MiscFeature',
                    'SaleType', 'SaleCondition']

    # Mark sets, combine, encode together, then split back
    combined = pd.concat([x_train.assign(_set='train'),
                          x_test.assign(_set='test')])
    combined = pd.get_dummies(combined, columns=nominal_cols, drop_first=True, dtype=int)

    x_train = combined[combined['_set'] == 'train'].drop('_set', axis=1)
    x_test = combined[combined['_set'] == 'test'].drop('_set', axis=1)

    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)


    return x_train, x_test, y_train, y_test

def correlation_filter(x_train, y_train, threshold):
        corr_matrix = np.abs(x_train.corr())
        high_corr_pairs = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

        high_corr_pairs.sort(key=lambda x: x[2], reverse=True)

        features_drop = []

        for feat1, feat2, val in high_corr_pairs:
            if abs(x_train[feat1].corr(y_train)) > abs(x_train[feat2].corr(y_train)):
                features_drop.append(feat2)
            else:
                features_drop.append(feat1)

        features_drop = list(set(features_drop))

        x_filtered = x_train.drop(columns = features_drop)

        return x_filtered