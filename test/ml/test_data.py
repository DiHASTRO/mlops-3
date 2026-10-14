'''
Testing ML data
'''
import os
import pandas as pd


def test_raw_data_exists(raw_data_path):
    '''Subj'''
    assert os.path.exists(raw_data_path), f"File {raw_data_path} not found!"


def test_raw_data_columns(raw_data_df):
    '''Subj'''
    expected = ["SalePrice", "Overall Qual", "Gr Liv Area", "Garage Cars", "Total Bsmt SF", "Year Built"]
    for col in expected:
        assert col in raw_data_df.columns, f"Missing column: {col}"


def test_raw_data_no_nulls_target(raw_data_df):
    '''Subj'''
    assert raw_data_df["SalePrice"].notna().all(), "Nulls in SalePrice"


def test_raw_data_positive_prices(raw_data_df):
    '''Subj'''
    assert (raw_data_df["SalePrice"] > 0).all(), "Non-positive SalePrice"


def test_processed_data_no_nulls(processed_train_path):
    '''Subj'''
    df = pd.read_csv(processed_train_path)
    assert df.notna().all().all(), "Nulls in processed data"
