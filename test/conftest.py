'''
All stuffs that are required for tests (fixtures, extra functions etc.)
'''
import pandas as pd
import pytest

from fastapi.testclient import TestClient

from src import main as entry_point


@pytest.fixture
def client():
    '''
    Client fixture to use in tests
    '''
    with TestClient(entry_point.app) as test_client:
        yield test_client


@pytest.fixture
def raw_data_path():
    '''
    Raw data path
    '''
    return "src/ml/dataset/raw/housing.csv"


@pytest.fixture
def processed_train_path():
    '''
    Processed train path
    '''
    return "src/ml/dataset/processed/train.csv"


@pytest.fixture
def raw_data_df(raw_data_path):  # pylint: disable=redefined-outer-name
    '''
    Raw data df
    '''
    return pd.read_csv(raw_data_path, sep='\t')
