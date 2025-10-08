import pytest
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/piccard")))

import piccard as pc

def test_clustering_prep_cols_specified(create_table):
    '''
    Test the clustering_prep function with a subset of specified columns, as recommended.
    '''
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
    '''
    Test the clustering_prep function with all possible columns, 
    which is not recommended but still should not throw any errors.
    '''
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
    '''
    Test the cluster function with a subset of specified columns and all default inputs.
    '''
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
    '''
    Test the cluster function with 6 clusters instead of the 4 recommended by the Elbow Method.
    '''
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


# def test_cluster_different_algo(create_table):
#     '''
#     Test the cluster function with the OptTSCluster algorithm. 
#     This test takes forever. Go outside and touch grass or something while it runs.
#     '''
#     network_table = create_table[1]
#     years = ['2006', '2011', '2016', '2021']
#     clustering_cols = []
#     vars = ['avg_income', 'avg_value', 'avg_rent']
#     for var in vars:
#         for year in years:
#             clustering_cols.append(f'{var}_{year}')
#     arr, label_dict, network_table = pc.clustering_prep(network_table, clustering_cols)

#     G = create_table[0]
#     clustered_table = pc.cluster(network_table, G, 4, arr=arr, label_dict=label_dict, algo="opt")
#     for year in years:
#         # check all years have a cluster assignment column in the network table
#         assert f"cluster_assignment_{year}" in clustered_table.table.columns
#         # check paths are only assigned to 0, 1, 2, or 3
#         clusters_year = clustered_table.table[f'cluster_assignment_{year}']
#         assert all((type(entry) == int) and (entry <= 3) for entry in clusters_year)
#     for node in list(G.nodes(data=True)):
#         # check all nodes have a cluster assignment in the graph
#         assert f'cluster_assignment' in node[1]
#         # check nodes are only assigned to 0, 1, 2, 3, or nan
#         assert (type(node[1]['cluster_assignment']) == int and node[1]['cluster_assignment'] <= 3) or np.isnan(node[1]['cluster_assignment'])


def test_cluster_different_scheme(create_table):
    '''
    Test the cluster function with a different tscluster clustering scheme (changing centres, fixed assignment).
    '''
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
        # since the scheme is now fixed assignment, all cluster assignment columns should be the same
        assert all(clusters_year == clustered_table.table['cluster_assignment_2006'])
    for node in list(G.nodes(data=True)):
        # check all nodes have a cluster assignment in the graph
        assert f'cluster_assignment' in node[1]
        # check nodes are only assigned to 0, 1, 2, 3, or nan
        assert (type(node[1]['cluster_assignment']) == int and node[1]['cluster_assignment'] <= 3) or np.isnan(node[1]['cluster_assignment'])
