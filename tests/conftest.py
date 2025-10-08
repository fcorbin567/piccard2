import pytest
import geopandas as gpd
import numpy as np
import networkx as nx
import random
import pandas as pd
import copy

@pytest.fixture
def create_datasets():
    '''
    Set up tests by reading and preparing datasets to make a network from.
    '''
    housing_data_06 = gpd.read_file("tests/testing_data/housing_data_06.geojson")
    housing_data_06.rename(columns={'v_CA06_2054: Average value of dwelling $': 'avg_value',
                                'v_CA06_2059: Average gross rent $': 'avg_rent',
                                'v_CA06_1584: Average income $': 'avg_income'
                                }, inplace=True)
    housing_data_11 = gpd.read_file("tests/testing_data/housing_data_11.geojson")
    housing_data_11.rename(columns={'v_CA11N_2287: Average value of dwellings ($)': 'avg_value',
                                'v_CA11N_2292: Average monthly shelter costs for rented dwellings ($)': 'avg_rent',
                                'v_CA11N_2344: Average income $': 'avg_income'
                                }, inplace=True)
    housing_data_16 = gpd.read_file("tests/testing_data/housing_data_16.geojson")
    housing_data_16.rename(columns={'v_CA16_4896: Average value of dwellings ($)': 'avg_value',
                                'v_CA16_4901: Average monthly shelter costs for rented dwellings ($)': 'avg_rent',
                                'v_CA16_4957: Average total income in 2015 among recipients ($)': 'avg_income'
                                }, inplace=True)
    housing_data_21 = gpd.read_file("tests/testing_data/housing_data_21.geojson")
    housing_data_21.rename(columns={'v_CA21_4312: Average value of dwellings ($) (60)': 'avg_value',
                                'v_CA21_4318: Average monthly shelter costs for rented dwellings ($) (59)': 'avg_rent',
                                'v_CA21_605: Average total income in 2020 among recipients ($)': 'avg_income'
                                }, inplace=True)
    
    census_dfs = [housing_data_06, housing_data_11, housing_data_16, housing_data_21]
    return census_dfs

@pytest.fixture
def create_table(create_datasets):
    '''
    Set up tests by creating a network graph and table to use for clustering.
    '''
    import piccard as pc
    census_dfs = create_datasets
    years = ['2006', '2011', '2016', '2021']
    G = pc.create_network(census_dfs, years, 'GeoUID')
    network_table = pc.create_network_table(census_dfs, years, 'GeoUID')
    return (G, network_table)

@pytest.fixture
def create_clustered_table(create_table):
    '''
    Set up tests by creating a clustered network table, another dataset to compare, and adding columns that allow
    for comparison based on bins of values. Ensures deterministic results by setting random seeds and avoiding in-place modification.
    '''
    import piccard as pc
    def _create_clustered_table(mismatch_networks=False, mismatch_years=False):
        np.random.seed(42)
        random.seed(42)
        gpd = __import__('geopandas')
        pd = __import__('pandas')
        # create dataframe with core housing need data (work on a copy)
        housing_data_coreneed_21 = gpd.read_file("tests/testing_data/housing_data_coreneed_21.geojson").copy()
        housing_data_coreneed_21.rename(columns={'v_CA21_4312: Average value of dwellings ($) (60)': 'avg_value',
                                        'v_CA21_4318: Average monthly shelter costs for rented dwellings ($) (59)': 'avg_rent',
                                        'v_CA21_605: Average total income in 2020 among recipients ($)': 'avg_income'
                                        }, inplace=True)
        housing_data_coreneed_21['pct_coreneed'] = housing_data_coreneed_21[
            'v_CA21_4303: In core need'] / housing_data_coreneed_21['v_CA21_4302: Total - Owner and tenant households with household ' \
            'total income greater than zero and shelter-cost-to-income ratio less than 100%, in non-farm, non-reserve private dwellings']
        housing_data_coreneed_21 = housing_data_coreneed_21.drop(['v_CA21_4303: In core need', 'v_CA21_4302: Total - Owner and tenant households with household ' \
            'total income greater than zero and shelter-cost-to-income ratio less than 100%, in non-farm, non-reserve private dwellings'], axis=1)
        G, network_table = create_table
        years = ['2006', '2011', '2016', '2021']
        clustering_cols = []
        vars = ['avg_income', 'avg_value', 'avg_rent']
        for var in vars:
            for year in years:
                clustering_cols.append(f'{var}_{year}')
        arr, label_dict, network_table = pc.clustering_prep(network_table, clustering_cols)
        clustered_table = pc.cluster(network_table, G, 4, arr=arr, label_dict=label_dict)
        # sort variable values into bins for prob_reasoning functions (work on a copy)
        table = clustered_table.table.copy()
        vars = ['avg_income', 'avg_value', 'avg_rent', 'pct_coreneed']
        for var in vars:
            for year in years:
                if var != 'pct_coreneed' and not (mismatch_years==True and year == '2016'):
                    table[f'{var}_binned_{year}'] = pd.qcut(table[f'{var}_{year}'], q=4, labels=['d1_Q1', 'd1_Q2', 'd1_Q3', 'd1_Q4'], duplicates='drop')
                elif var != 'pct_coreneed' and mismatch_years==True and year == '2016':
                    table[f'{var}_binned_{year}'] = pd.qcut(table[f'{var}_{year}'], q=3, labels=['d2_Q1', 'd2_Q2', 'd2_Q3'], duplicates='drop')
                if year == '2021' and mismatch_networks==False:
                    housing_data_coreneed_21[f'{var}_binned_{year}'] = pd.qcut(housing_data_coreneed_21[var], q=4, labels=['d1_Q1', 'd1_Q2', 'd1_Q3', 'd1_Q4'], duplicates='drop')
                elif year == '2021' and mismatch_networks==True:
                    housing_data_coreneed_21[f'{var}_binned_{year}'] = pd.qcut(housing_data_coreneed_21[var], q=3, labels=['d2_Q1', 'd2_Q2', 'd2_Q3'], duplicates='drop')
        clustered_table.modify_table(table)
        # Always return deep copies to ensure test isolation
        return (copy.deepcopy(clustered_table), copy.deepcopy(clustering_cols), copy.deepcopy(housing_data_coreneed_21))
    return _create_clustered_table
