import pytest
import pandas as pd
import os

def pytest_addoption(parser):
    parser.addoption("--csv", action="store", help="Path to the csv file")
    parser.addoption("--ref", action="store", help="Path to the reference data")
    parser.addoption("--kl_threshold", action="store", help="KL divergence threshold")
    parser.addoption("--min_price", action="store", help="Minimum price")
    parser.addoption("--max_price", action="store", help="Maximum price")

@pytest.fixture(scope="session")
def data(request):
    csv_path = request.config.option.csv
    if not os.path.exists(csv_path):
        pytest.fail(f"File not found: {csv_path}")
    return pd.read_csv(csv_path)

@pytest.fixture(scope="session")
def ref_data(request):
    csv_path = request.config.option.ref
    if not os.path.exists(csv_path):
        pytest.fail(f"File not found: {csv_path}")
    return pd.read_csv(csv_path)

@pytest.fixture(scope="session")
def kl_threshold(request):
    return float(request.config.option.kl_threshold)

@pytest.fixture(scope="session")
def min_price(request):
    return float(request.config.option.min_price)

@pytest.fixture(scope="session")
def max_price(request):
    return float(request.config.option.max_price)