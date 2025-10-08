import pytest
import geopandas as gpd
import pandas as pd
import copy

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/piccard")))

import piccard as pc

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