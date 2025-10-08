import pytest
import geopandas as gpd
import pandas as pd
import copy

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
    for comparison based on bins of values. Ensures deterministic results by setting random seeds and avoiding in-place modification.
    '''
    import numpy as np
    import random
    def _create_clustered_table(mismatch_networks=False, mismatch_years=False):
        # Set random seeds for deterministic binning and clustering
        np.random.seed(42)
        random.seed(42)

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

        # go through clustering process (work on a copy)
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


def test_prob_reasoning_networks_default_inputs(create_clustered_table):
    '''
    Test the prob_reasoning_networks function with no mismatches.
    '''
    clustered_table, clustering_cols, housing_data_coreneed_21 = create_clustered_table()

    indep_vars = [f'{col[:-4]}binned_{col[-4:]}' for col in clustering_cols]
    indep_vars_2 = [col for col in indep_vars if '2021' in col]
    dep_vars = []
    dep_vars_2 = ['pct_coreneed_binned_2021']
    joined_pdf_networks = pc.prob_reasoning_networks(clustered_table, housing_data_coreneed_21, indep_vars, indep_vars_2, dep_vars, dep_vars_2)
    for i in range(4):
        print(f'Probability of each pct_coreneed bin given that income is in Q{i + 1}')
        print(joined_pdf_networks.query(['pct_coreneed_binned_2021'], evidence_vars={'avg_income_binned_2021':f'd1_Q{i + 1}'}))


def test_prob_reasoning_years_default_inputs(create_clustered_table):
    '''
    Test the prob_reasoning_years function with no mismatches.
    '''
    clustered_table, clustering_cols, housing_data_coreneed_21 = create_clustered_table()

    indep_vars = [f'{col[:-4]}binned_{col[-4:]}' for col in clustering_cols if '2016' in col]
    indep_vars_2 = [f'{col[:-4]}binned_{col[-4:]}' for col in clustering_cols if '2021' in col]
    dep_vars = ['cluster_assignment_2016']
    dep_vars_2 = ['cluster_assignment_2021']
    joined_pdf_years = pc.prob_reasoning_years(clustered_table, '2016', '2021', indep_vars, indep_vars_2, dep_vars, dep_vars_2)
    for i in range(4):
        print(f'Probability of each cluster assignment given that income is in Q{i + 1}')
        print(joined_pdf_years.query(['cluster_assignment'], evidence_vars={'avg_income_binned':f'd1_Q{i + 1}'}))


def test_prob_reasoning_networks_mismatches(create_clustered_table):
    '''
    Test the prob_reasoning_networks function with mismatches between number of quantiles across datasets.
    WARNING: Queries may fail due to lack of data for some parent state combinations.
    '''
    clustered_table, clustering_cols, housing_data_coreneed_21 = create_clustered_table(mismatch_networks=True)

    indep_vars = [f'{col[:-4]}binned_{col[-4:]}' for col in clustering_cols]
    indep_vars_2 = [col for col in indep_vars if '2021' in col]
    dep_vars = []
    dep_vars_2 = ['pct_coreneed_binned_2021']

    joined_pdf_networks = pc.prob_reasoning_networks(
        clustered_table, housing_data_coreneed_21,
        indep_vars, indep_vars_2, dep_vars, dep_vars_2,
        mismatches = {
            'avg_income_binned_2021': 'categorical',
            'avg_value_binned_2021': 'categorical',
            'avg_rent_binned_2021': 'categorical'
        }
    )

    # Print a sample of present parent state combinations and query each
    cpd = joined_pdf_networks.bayes_net.get_cpds('pct_coreneed_binned_2021')
    if cpd is not None:
        parent_vars = cpd.get_evidence()
        if parent_vars:
            parent_states = [cpd.state_names[p] for p in parent_vars]
            parent_combos = list(pd.MultiIndex.from_product(parent_states, names=parent_vars))
            # Get all unique combinations present in the data
            data = clustered_table.table.copy()
            # Rename columns to year-agnostic
            data = data.rename(columns={col: col[:-5] for col in data.columns if col.endswith('_2021')})
            data = data.loc[:, ~data.columns.duplicated()]
            # Use year-agnostic parent_vars for indexing
            year_agnostic_parent_vars = [p[:-5] if p.endswith('_2021') else p for p in parent_vars]
            present_combos = set(tuple(row) for row in data[year_agnostic_parent_vars].dropna().values)
            print("Sample of present parent state combinations in data:")
            for combo in list(present_combos)[:10]:
                evidence = dict(zip(parent_vars, combo))
                print(f"Evidence: {evidence}")
                try:
                    result = joined_pdf_networks.query(['pct_coreneed_binned_2021'], evidence_vars=evidence)
                    print("Query result:")
                    print(result)
                except Exception as e:
                    print(f"Query failed: {e}")


def test_prob_reasoning_years_mismatches(create_clustered_table):
    '''
    Test the prob_reasoning_years function with mismatches between number of quantiles across years in the same dataset.
    WARNING: Queries may fail due to lack of data for some parent state combinations.
    '''
    clustered_table, clustering_cols, housing_data_coreneed_21 = create_clustered_table(mismatch_years=True)

    indep_vars_2016 = [col for col in clustered_table.table.columns if col.endswith('_2016') and 'binned' in col and col != 'pct_coreneed_binned_2016']
    indep_vars_2021 = [col for col in clustered_table.table.columns if col.endswith('_2021') and 'binned' in col and col != 'pct_coreneed_binned_2021']
    dep_vars_2016 = ['cluster_assignment_2016']
    dep_vars_2021 = ['cluster_assignment_2021']

    joined_pdf_years = pc.prob_reasoning_years(
        clustered_table, '2016', '2021',
        indep_vars_2016, indep_vars_2021, dep_vars_2016, dep_vars_2021,
        mismatches={
            'avg_income_binned': 'categorical',
            'avg_value_binned': 'categorical',
            'avg_rent_binned': 'categorical'
        }
    )

    cpd = joined_pdf_years.bayes_net.get_cpds('cluster_assignment')
    if cpd is not None:
        parent_vars = cpd.get_evidence()
        if parent_vars:
            parent_states = [cpd.state_names[p] for p in parent_vars]
            parent_combos = list(pd.MultiIndex.from_product(parent_states, names=parent_vars))
            data = clustered_table.table.copy()
            data = data.rename(columns={col: col[:-5] for col in data.columns if col.endswith('_2016') or col.endswith('_2021')})
            data = data.loc[:, ~data.columns.duplicated()]
            # Use year-agnostic parent_vars for indexing
            year_agnostic_parent_vars = [p[:-5] if p.endswith('_2016') or p.endswith('_2021') else p for p in parent_vars]
            present_combos = set(tuple(row) for row in data[year_agnostic_parent_vars].dropna().values)
            print("Sample of present parent state combinations in data:")
            for combo in list(present_combos)[:10]:
                evidence = dict(zip(parent_vars, combo))
                print(f"Evidence: {evidence}")
                try:
                    result = joined_pdf_years.query(['cluster_assignment'], evidence_vars=evidence)
                    print("Query result:")
                    print(result)
                except Exception as e:
                    print(f"Query failed: {e}")