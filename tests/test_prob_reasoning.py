import pytest
import geopandas as gpd
import pandas as pd

import sys
import os
from test_network_creation import create_datasets
from test_clustering import create_table
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/piccard")))

import piccard as pc

@pytest.fixture
def create_clustered_table(create_table):
    '''
    Set up tests by creating a clustered network table, another dataset to compare, and adding columns that allow
    for comparison based on bins of values
    '''
    # create dataframe with core housing need data
    housing_data_coreneed_21 = gpd.read_file("tests/testing_data/housing_data_coreneed_21.geojson")
    housing_data_coreneed_21.rename(columns={'v_CA21_4312: Average value of dwellings ($) (60)': 'avg_value',
                                    'v_CA21_4318: Average monthly shelter costs for rented dwellings ($) (59)': 'avg_rent',
                                    'v_CA21_605: Average total income in 2020 among recipients ($)': 'avg_income'
                                    }, inplace=True)
    housing_data_coreneed_21['pct_coreneed'] = housing_data_coreneed_21[
        'v_CA21_4303: In core need'] / housing_data_coreneed_21['v_CA21_4302: Total - Owner and tenant households with household ' \
        'total income greater than zero and shelter-cost-to-income ratio less than 100%, in non-farm, non-reserve private dwellings']
    housing_data_coreneed_21 = housing_data_coreneed_21.drop(['v_CA21_4303: In core need', 'v_CA21_4302: Total - Owner and tenant households with household ' \
        'total income greater than zero and shelter-cost-to-income ratio less than 100%, in non-farm, non-reserve private dwellings'], axis=1)
    
    # go through clustering process
    G, network_table = create_table
    years = ['2006', '2011', '2016', '2021']
    clustering_cols = []
    vars = ['avg_income', 'avg_value', 'avg_rent']
    for var in vars:
        for year in years:
            clustering_cols.append(f'{var}_{year}')
    arr, label_dict, network_table = pc.clustering_prep(network_table, clustering_cols)
    clustered_table = pc.cluster(network_table, G, 4, arr=arr, label_dict=label_dict)

    # sort variable values into bins for prob_reasoning functions
    table = clustered_table.table
    vars = ['avg_income', 'avg_value', 'avg_rent', 'pct_coreneed']
    for var in vars:
        for year in years:
            if var != 'pct_coreneed':
                table[f'{var}_binned_{year}'] = pd.qcut(table[f'{var}_{year}'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            if year == '2021':
                housing_data_coreneed_21[f'{var}_binned_{year}'] = pd.qcut(housing_data_coreneed_21[var], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # modify and return clustered_table with new bins
    clustered_table.modify_table(table)
    return (clustered_table, clustering_cols, housing_data_coreneed_21)


# TODO: test_prob_reasoning_networks with mismatches and with modifying tables


def test_prob_reasoning_networks_default_inputs(create_clustered_table):
    '''
    Test the prob_reasoning_networks function with no mismatches and without modifying tables.
    '''
    clustered_table, clustering_cols, housing_data_coreneed_21 = create_clustered_table
    indep_vars = [f'{col[:-4]}binned_{col[-4:]}' for col in clustering_cols]
    indep_vars_2 = [col for col in indep_vars if '2021' in col]
    dep_vars = []
    dep_vars_2 = ['pct_coreneed_binned_2021']
    joined_pdf_networks = pc.prob_reasoning_networks(clustered_table, housing_data_coreneed_21, indep_vars, indep_vars_2, dep_vars, dep_vars_2)
    for i in range(4):
        print(f'Probability of each pct_coreneed bin given that income is in Q{i + 1}')
        print(joined_pdf_networks.query(['pct_coreneed_binned_2021'], evidence_vars={'avg_income_binned_2021':f'Q{i + 1}'}))


def test_prob_reasoning_years_default_inputs(create_clustered_table):
    '''
    Test the prob_reasoning_years function with no mismatches and without modifying tables.
    '''
    clustered_table, clustering_cols, housing_data_coreneed_21 = create_clustered_table
    indep_vars = [f'{col[:-4]}binned_{col[-4:]}' for col in clustering_cols if '2016' in col]
    indep_vars_2 = [f'{col[:-4]}binned_{col[-4:]}' for col in clustering_cols if '2021' in col]
    dep_vars = ['cluster_assignment_2016']
    dep_vars_2 = ['cluster_assignment_2021']
    joined_pdf_years = pc.prob_reasoning_years(clustered_table, '2016', '2021', indep_vars, indep_vars_2, dep_vars, dep_vars_2)
    for i in range(4):
        print(f'Probability of each cluster assignment given that income is in Q{i + 1}')
        print(joined_pdf_years.query(['cluster_assignment'], evidence_vars={'avg_income_binned':f'Q{i + 1}'}))


# TODO: test_prob_reasoning_networks with mismatches and with modifying tables