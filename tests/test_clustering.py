import pytest
import numpy as np

import sys
import os
from test_network_creation import create_datasets
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/piccard")))

import piccard as pc

@pytest.fixture
def create_table(create_datasets):
    census_dfs = create_datasets
    years = ['2006', '2011', '2016', '2021']
    G = pc.create_network(census_dfs, years, 'GeoUID')
    network_table = pc.create_network_table(census_dfs, years, 'GeoUID')
    return (G, network_table)


def test_clustering_prep_cols_specified(create_table):
    years = ['2006', '2011', '2016', '2021']
    network_table = create_table[1]
    clustering_cols = []
    vars = ['avg_income', 'avg_value', 'avg_rent']
    for var in vars:
        for year in years:
            clustering_cols.append(f'{var}_{year}')
    old_num_rows = len(network_table.table)
    arr, label_dict, network_table = pc.clustering_prep(network_table, clustering_cols)
    new_num_rows = len(network_table.table)
    # check that all rows with entirely NaN values were filtered out
    assert old_num_rows > new_num_rows
    # check if all numerical columns are in the features array
    assert arr.shape[2] == 3
    # check the names of the labels
    assert label_dict['F'] == vars


def test_clustering_prep_no_cols_specified(create_table):
    network_table = create_table[1]
    old_num_rows = len(network_table.table)
    arr, label_dict, network_table = pc.clustering_prep(network_table)
    new_num_rows = len(network_table.table)
    # check that all rows with entirely NaN values were filtered out
    assert old_num_rows > new_num_rows
    # check if all numerical columns are in the features array
    assert arr.shape[2] == 12
    # check the names of the labels
    assert label_dict['F'] == ['shape area', 'households', 'dwellings', 'population', 
                               'cma_uid', 'csd_uid', 'cd_uid', 'area (sq km)', 'avg_income', 
                               'avg_value', 'avg_rent', 'network_level'] 


def test_cluster_default_inputs(create_table):
    network_table = create_table[1]
    years = ['2006', '2011', '2016', '2021']
    clustering_cols = []
    vars = ['avg_income', 'avg_value', 'avg_rent']
    for var in vars:
        for year in years:
            clustering_cols.append(f'{var}_{year}')
    arr, label_dict, network_table = pc.clustering_prep(network_table, clustering_cols)

    G = create_table[0]
    clustered_table = pc.cluster(network_table, G, 4, arr=arr, label_dict=label_dict)
    for year in years:
        # check all years have a cluster assignment column in the network table
        assert f"cluster_assignment_{year}" in clustered_table.table.columns
        # check paths are only assigned to 0, 1, 2, or 3
        clusters_year = clustered_table.table[f'cluster_assignment_{year}']
        assert all((type(entry) == int) and (entry <= 3) for entry in clusters_year)
    for node in list(G.nodes(data=True)):
        # check all nodes have a cluster assignment in the graph
        assert f'cluster_assignment' in node[1]
        # check nodes are only assigned to 0, 1, 2, 3, or nan
        assert (type(node[1]['cluster_assignment']) == int and node[1]['cluster_assignment'] <= 3) or np.isnan(node[1]['cluster_assignment'])


def test_cluster_different_num_clusters(create_table):
    network_table = create_table[1]
    years = ['2006', '2011', '2016', '2021']
    clustering_cols = []
    vars = ['avg_income', 'avg_value', 'avg_rent']
    for var in vars:
        for year in years:
            clustering_cols.append(f'{var}_{year}')
    arr, label_dict, network_table = pc.clustering_prep(network_table, clustering_cols)

    G = create_table[0]
    clustered_table = pc.cluster(network_table, G, 6, arr=arr, label_dict=label_dict)
    for year in years:
        # check all years have a cluster assignment column in the network table
        assert f"cluster_assignment_{year}" in clustered_table.table.columns
        # check paths are only assigned to 0, 1, 2, 3, 4, or 5
        clusters_year = clustered_table.table[f'cluster_assignment_{year}']
        assert all((type(entry) == int) and (entry <= 5) for entry in clusters_year)
    for node in list(G.nodes(data=True)):
        # check all nodes have a cluster assignment in the graph
        assert f'cluster_assignment' in node[1]
        # check nodes are only assigned to 0, 1, 2, 3, 4, 5, or nan
        assert (type(node[1]['cluster_assignment']) == int and node[1]['cluster_assignment'] <= 5) or np.isnan(node[1]['cluster_assignment'])


def test_cluster_different_algo(create_table):
    network_table = create_table[1]
    years = ['2006', '2011', '2016', '2021']
    clustering_cols = []
    vars = ['avg_income', 'avg_value', 'avg_rent']
    for var in vars:
        for year in years:
            clustering_cols.append(f'{var}_{year}')
    arr, label_dict, network_table = pc.clustering_prep(network_table, clustering_cols)

    G = create_table[0]
    clustered_table = pc.cluster(network_table, G, 4, arr=arr, label_dict=label_dict, algo="opt")
    for year in years:
        # check all years have a cluster assignment column in the network table
        assert f"cluster_assignment_{year}" in clustered_table.table.columns
        # check paths are only assigned to 0, 1, 2, or 3
        clusters_year = clustered_table.table[f'cluster_assignment_{year}']
        assert all((type(entry) == int) and (entry <= 3) for entry in clusters_year)
    for node in list(G.nodes(data=True)):
        # check all nodes have a cluster assignment in the graph
        assert f'cluster_assignment' in node[1]
        # check nodes are only assigned to 0, 1, 2, 3, or nan
        assert (type(node[1]['cluster_assignment']) == int and node[1]['cluster_assignment'] <= 3) or np.isnan(node[1]['cluster_assignment'])


def test_cluster_different_scheme(create_table):
    network_table = create_table[1]
    years = ['2006', '2011', '2016', '2021']
    clustering_cols = []
    vars = ['avg_income', 'avg_value', 'avg_rent']
    for var in vars:
        for year in years:
            clustering_cols.append(f'{var}_{year}')
    arr, label_dict, network_table = pc.clustering_prep(network_table, clustering_cols)

    G = create_table[0]
    clustered_table = pc.cluster(network_table, G, 4, arr=arr, label_dict=label_dict, scheme="z1c0") # changing centre, fixed assignment
    for year in years:
        # check all years have a cluster assignment column in the network table
        assert f"cluster_assignment_{year}" in clustered_table.table.columns
        # check paths are only assigned to 0, 1, 2, or 3
        clusters_year = clustered_table.table[f'cluster_assignment_{year}']
        assert all((type(entry) == int) and (entry <= 3) for entry in clusters_year)
    for node in list(G.nodes(data=True)):
        # check all nodes have a cluster assignment in the graph
        assert f'cluster_assignment' in node[1]
        # check nodes are only assigned to 0, 1, 2, 3, or nan
        assert (type(node[1]['cluster_assignment']) == int and node[1]['cluster_assignment'] <= 3) or np.isnan(node[1]['cluster_assignment'])