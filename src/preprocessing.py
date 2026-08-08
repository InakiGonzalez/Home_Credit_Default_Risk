from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder




def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """

      
    # Print shape of input data
    print("Input train data shape: ", train_df.shape)
    print("Input val data shape: ", val_df.shape)
    print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # 1. Correct outliers/anomalous values in numerical
    # columns (`DAYS_EMPLOYED` column).
    working_train_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_val_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
    working_test_df["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)

    # 2. Encode string categorical features (dtype `object`):
    #     - 2 categories -> binary encoding via OrdinalEncoder (in-place, 1 col in/out)
    #     - >2 categories -> one-hot encoding via OneHotEncoder (expands columns,
    #       so original columns must be dropped and replaced, not assigned in-place)
    object_cols = working_train_df.select_dtypes(include="object").columns
    binary_cols = [c for c in object_cols if working_train_df[c].nunique(dropna=True) == 2]
    onehot_cols = [c for c in object_cols if working_train_df[c].nunique(dropna=True) > 2]

    ord_enc = OrdinalEncoder()
    ord_enc.fit(working_train_df[binary_cols])

    oh_enc = OneHotEncoder(sparse_output=False)
    oh_enc.fit(working_train_df[onehot_cols])
    onehot_feature_names = oh_enc.get_feature_names_out(onehot_cols)

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        df[binary_cols] = ord_enc.transform(df[binary_cols])

        onehot_encoded = pd.DataFrame(
            oh_enc.transform(df[onehot_cols]),
            columns=onehot_feature_names,
            index=df.index,
        )
        df = pd.concat([df.drop(columns=onehot_cols), onehot_encoded], axis=1)
        return df


    working_train_df_enc = encode(working_train_df)
    working_val_df_enc = encode(working_val_df)
    working_test_df_enc = encode(working_test_df)


    # 3. TODO Impute values for all columns with missing data or, just all the columns.
    # Use median as imputing value. Please use sklearn.impute.SimpleImputer().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the SimpleImputer and then use the fitted
    #     model to transform all the datasets.
    simple_imp = SimpleImputer(strategy="median")
    simple_imp.fit(working_train_df_enc)
    
    working_train_df_imp = simple_imp.transform(working_train_df_enc)
    working_val_df_imp = simple_imp.transform(working_val_df_enc)
    working_test_df_imp = simple_imp.transform(working_test_df_enc)

    # 4. TODO Feature scaling with Min-Max scaler. Apply this to all the columns.
    # Please use sklearn.preprocessing.MinMaxScaler().
    # Again, take into account that:
    #   - You must apply this to the 3 DataFrames (working_train_df, working_val_df,
    #     working_test_df).
    #   - In order to prevent overfitting and avoid Data Leakage you must use only
    #     working_train_df DataFrame to fit the MinMaxScaler and then use the fitted
    #     model to transform all the datasets.

    mm_scaler = MinMaxScaler()
    mm_scaler.fit(working_train_df_imp)
    
    working_train_df_engineered = mm_scaler.transform(working_train_df_imp)
    working_val_df_engineered= mm_scaler.transform(working_val_df_imp)
    working_test_df_engineered= mm_scaler.transform(working_test_df_imp)
    
    return working_train_df_engineered, working_val_df_engineered, working_test_df_engineered
