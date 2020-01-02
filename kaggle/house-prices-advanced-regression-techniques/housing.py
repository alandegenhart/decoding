# Setup environment and import standard modules
import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt

# Define function to perform one-hot encoding
# TODO: add option to convert to n-1 classes if the list of possible values is
# fully-determined (e.g., T/F)
def one_hot_enc(vals, key_str):
    """
    Transform categorical data to one-hot encoding.

    This function transforms the input data in vals to a one-hot encoding
    scheme, where the possible values the input data can take become new
    columns in the transformed data set.  This function can accept either
    a Pandas Series or DataFrame as an input.

    This function currently assumes that the input data is encoded in a
    boolean manner (T/F) for multi-category inputs.  In other words, if
    the data is 2D and has the same value for multiple columns, this
    function will not count these values multiple times.
    """
    # Initialize dataframe
    df = pd.DataFrame()

    # Determine key string prefix.  If 'key_str' is a string, use the
    # provided value.  If it is an array of strings, use the first value.
    # This allows the function to be used in an iterable manner, where a
    # list of fields/lists can be iterated over -- in this case it can be
    # easier just to pass in the list of fields used to select the subest
    # of columns to use.
    if type(key_str) == list:
        key_str = key_str[0]

    # Get set of keys
    vals = vals.values
    data_shape = vals.shape
    if len(data_shape) == 1:
        data_shape = (data_shape[0], 1)
        vals = np.expand_dims(vals, axis=1)

    n_col = data_shape[1]
    key_list = set()
    for col in range(n_col):
        key_list.update(vals[:, col])

    for key in key_list:
        # Define name for the value to be encoded
        new_key_name = '{}_{}'.format(key_str, key)

        # Iterate over columns and transform
        trans_vals = vals[:, 0] == key
        for col in range(1, n_col):
            trans_vals = trans_vals | (vals[:, col] == key)

        # Add transformed data for current key to output dataframe
        df[new_key_name] = trans_vals

    return df


def load_proc_datasets():
    """
    Load and process full dataset.

    This function loads the train and test datasets and processes them for
    subsequent anaysis.
    """
    # Define list of default values
    default_vals = {
        'MSZoning': 'NONE',
        'LotFrontage': 0.0,
        'Alley': 'NONE',
        'Utilities': 'NONE',
        'Exterior1st': 'NONE',
        'Exterior2nd': 'NONE',
        'MasVnrType': 'NONE',
        'MasVnrArea': 0.0,
        'BsmtQual': 'NONE',
        'BsmtCond': 'NONE',
        'BsmtExposure': 'NONE',
        'BsmtFinType1': 'NONE',
        'BsmtFinSF1': 0.0,
        'BsmtFinType2': 'NONE',
        'BsmtFinSF2': 0.0,
        'BsmtUnfSF': 0.0,
        'TotalBsmtSF': 0.0,
        'Electrical': 'NONE',
        'BsmtFullBath': 0,
        'BsmtHalfBath': 0,
        'KitchenQual': 'NONE',
        'Functional': 'Typ',
        'FireplaceQu': 'NONE',
        'GarageType': 'NONE',
        'GarageFinish': 'NONE',
        'GarageCars': 0,
        'GarageArea': 0.0,
        'GarageQual': 'NONE',
        'GarageCond': 'NONE',
        'PoolQC': 'NONE',
        'Fence': 'NONE',
        'MiscFeature': 'NONE',
        'SaleType': 'Oth'
    }

    # Load data
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')

    # Merge datasets so that they can be processed together
    df_train_2 = df_train.copy()
    sale_price = df_train_2.pop('SalePrice')
    df_merged = pd.concat([df_train_2, df_test])

    # Merging the dataset means that there are now duplicate indices.  To fix
    # this, re-set the index.
    df_merged.set_index(np.arange(df_merged.shape[0]), inplace=True)

    # Replace any empty entries for 'GarageYrBlt' with 'YearBuilt'.
    mask = df_merged['GarageYrBlt'].isna()
    df_merged['GarageYrBlt'][mask] = df_merged['YearBuilt'][mask]

    # Iterate over fields and replace all NaNs with the default value
    for key, value in df_merged.iteritems():
        if sum(value.isna()>0):
            value[value.isna()] = default_vals[key]
            df_merged[key] = value

    # Check to see which fields have nan values
    n_na = df_merged.isna().sum()  # Provides the sum across columns
    print(n_na[n_na > 0])

    # Specify fields that can be converted to numerical
    enum_fields = {
        'ExterQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtQual': ['NONE', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtCond': ['NONE', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtExposure': ['NONE', 'No', 'Mn', 'Av', 'Gd'],
        'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'KitchenQual': ['NONE', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'Functional': ['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
        'FireplaceQu': ['NONE', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'GarageFinish': ['NONE', 'Unf', 'RFn', 'Fin'],
        'GarageQual': ['NONE', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'GarageCond': ['NONE', 'Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'PoolQC': ['NONE', 'Fa', 'TA', 'Gd', 'Ex'],
        'Fence': ['NONE', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv']
    }
    # Now iterate over keys and update
    for key in enum_fields:
        # Create dictionary
        num_vals = np.arange(len(enum_fields[key]))
        key_dict = {'{}'.format(code):val for code, val in zip(enum_fields[key], num_vals)}

        # Apply dictionary to elements in the dataframe
        df_merged.replace({key: key_dict}, inplace=True)

    # Merge square footage for different basement values
    uni_vals = set(df_merged['BsmtFinType1'])
    n_col = df_merged.shape[0]
    bsmt_added_cols = []
    for val in uni_vals:
        col_name = 'BsmtFin_{}'.format(val)
        bsmt_added_cols.append(col_name)

        # Get mask for type 1
        col_mask_1 = df_merged['BsmtFinType1'] == val
        col_mask_2 = df_merged['BsmtFinType2'] == val

        # Get associated square footage
        col_sf_1 = np.zeros(n_col)
        col_sf_1[col_mask_1] = df_merged['BsmtFinSF1'][col_mask_1]
        col_sf_2 = np.zeros(n_col)
        col_sf_2[col_mask_2] = df_merged['BsmtFinSF2'][col_mask_2]
        col_sf_merged = col_sf_1 + col_sf_2

        # Add field to dataframe
        df_merged[col_name] = col_sf_merged

    # List of numerical fields -- these can be copied into the processed
    # dataset directly. It might be helpful to z-score these -- this would help
    # when visualizing the weights (?)
    numerical_fields = [
        'LotFrontage',
        'LotArea',
        'OverallQual',
        'OverallCond',
        'YearBuilt',
        'YearRemodAdd',
        'MasVnrArea',
        'BsmtUnfSF',
        'TotalBsmtSF',
        '1stFlrSF',
        '2ndFlrSF',
        'LowQualFinSF',
        'GrLivArea',
        'BsmtFullBath',
        'BsmtHalfBath',
        'FullBath',
        'HalfBath',
        'BedroomAbvGr',
        'KitchenAbvGr',
        'TotRmsAbvGrd',
        'Fireplaces',
        'GarageYrBlt',
        'GarageCars',
        'GarageArea',
        'WoodDeckSF',
        'OpenPorchSF',
        'EnclosedPorch',
        '3SsnPorch',
        'ScreenPorch',
        'PoolArea',
        'MiscVal',
        'MoSold',
        'YrSold'
    ]
    # Add enumerated fields
    numerical_fields.extend(enum_fields.keys())
    numerical_fields.extend(bsmt_added_cols)
    for key in numerical_fields:
        df_merged[key] = stats.zscore(df_merged[key].astype(float))

    # Specify values to convert to one-hot encoding
    one_hot_fields = [
        'MSZoning',
        'MSSubClass',
        'Street',
        'Alley',
        'LotShape',
        'LandContour',
        'Utilities',
        'LotConfig',
        'LandSlope',
        'Neighborhood',
        'BldgType',
        'HouseStyle',
        'RoofStyle',
        'RoofMatl',
        'MasVnrType',
        'Foundation',
        'Heating',
        'CentralAir',
        'Electrical',
        'GarageType',
        'PavedDrive',
        'SaleType',
        'SaleCondition',
        'MiscFeature'
    ]
    df_enc = [one_hot_enc(df_merged[key], key) for key in one_hot_fields]

    # Create merged processed dataframe
    numerical_fields = ['Id'] + numerical_fields
    df_list = [df_merged[numerical_fields]] + df_enc
    df_proc = pd.concat(df_list, axis=1, sort=False)

    # Finally, split into training and testing sets
    train_id = df_train['Id']
    test_id = df_test['Id']
    train_mask = df_proc['Id'].isin(df_train['Id'])
    test_mask = df_proc['Id'].isin(df_test['Id'])
    df_proc_train = df_proc.iloc[train_mask.values, :]
    df_proc_test = df_proc.iloc[test_mask.values, :]

    # Add sale price back to the training set
    df_proc_train['SalePrice'] = sale_price

    return df_proc_train, df_proc_test


# Define functions -- these will eventually be off-loaded to a separate module

# Function to plot the sale price as a function of a specific feature
def plot_price(df, field_name):

    plt.plot(df[field_name], df['SalePrice'], 'k.')
    plt.xlabel(field_name)
    plt.ylabel('Sale price')

    return None


# Function to check to see if data has NaN/empty values
def check_nan(df):
    num_nan = []
    nan_list = []
    for key, value in df.iteritems():
        # Check to see if any of the values are NaN
        if sum(np.isnan(value)) > 0:
            num_nan.append(sum(np.isnan(value)))
            nan_list.append(key)

    print('Fields with NaN values: {}{}'.format(nan_list, num_nan))

    return None


# Define function to z-score features
def calc_norm_params(df):
    n_features = len(df.keys())
    feat_mean = {}
    feat_std = {}
    for key, value in df.iteritems():
        # Calculate mean
        feat_value = df[key]
        feat_mean[key] = np.mean(feat_value)
        feat_value = feat_value - feat_mean[key]

        # Calcualte standard deviation
        feat_std[key] = np.std(feat_value)

    return feat_mean, feat_std


# Define function to apply normalization parameters
def apply_norm_params(df, feat_mean, feat_std):
    # Iterate over features and apply normalization
    for key, value in df.iteritems():
        df[key] = (df[key] - feat_mean[key])/feat_std[key]

    return df


# Define function to plot results
def plot_fit(y, y_predict, title):
    # Plot actual vs predicted
    plt.plot(y, y_predict, 'k.')

    # Plot unity line and set axis limits
    x_lim = plt.gca().get_xlim()
    y_lim = plt.gca().get_ylim()
    ax_lim = [np.min([x_lim, y_lim]), np.max([x_lim, y_lim])]
    plt.plot(ax_lim, ax_lim, 'r--')
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    # Format plot
    plt.xlabel('Actual sale price')
    plt.ylabel('Predicted sale price')
    plt.title(title)
    plt.show()


# Define function to clean data
def clean_data_subset(df):
    # Get a subset of fields
    col_subset = [
        'OverallQual',
        'OverallCond',
        'YearBuilt',
        'MiscVal',
        'Functional',
        '1stFlrSF',
        '2ndFlrSF',
        'LotArea',
        'LotFrontage',
        'TotalBsmtSF'
    ]
    df_subset = df[col_subset].copy()
    # This suppress pandas warnings related to modifying a copy of a dataframe

    # Convert 'Functional' field to a numerical value
    func_val = {
        'Sal': 0,
        'Sev': 1,
        'Maj2': 2,
        'Maj1': 3,
        'Mod': 4,
        'Min2': 5,
        'Min1': 6,
        'Typ': 7
    }
    # Iterate over rows and convert string to a numerical value.  This has to
    # be done before checking for other NaN values b/c the 'isnan' function
    # returns an error if the input array contains non-numerical data.
    for idx, row in df_subset.iterrows():
        # If info is not provided, assume that it is NaN
        if row.Functional not in func_val.keys():
            row.Functional = 'Typ'
        # Replace string with numerical value
        df_subset.loc[idx, 'Functional'] = func_val[row.Functional]

    # Replace any NaN values with 0
    for key, value in df_subset.iteritems():
        mask = np.isnan(value)
        df_subset[key][df_subset[key].isna()] = 0

    # Add 'TotalSF' as a field
    df_subset['TotalSF'] = df_subset['1stFlrSF'] + df_subset['2ndFlrSF']

    return df_subset


# Define function to format/safe data for submission
def save_submission(test_id, y_predict, filename):
    # Create dataframe from imput data
    y_predict = y_predict.flatten()
    id_vals = test_id.values
    id_out = pd.Series(id_vals, name='Id')
    sale_price = pd.Series(y_predict, name='SalePrice')
    df_out = pd.concat({'Id': id_out, 'SalePrice': sale_price}, axis=1)

    # Save dataframe as *.csv file
    df_out.to_csv('housing_results_{}.csv'.format(filename), index=False)

    return df_out


# Define function to plot history
def plot_history(history):
    # Convert the TF history object to a Pandas data frame
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch  # Add training epoch

    # Create figure -- absolute error
    plt.figure(figsize=(10,5))
    plt.subplot(1, 1, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Loss')
    plt.plot(hist['epoch'], hist['val_loss'], label='Val Loss')
    plt.ylim([0, 1])
    plt.legend()

    # Show figures
    plt.show()
