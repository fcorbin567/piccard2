import pytest
import geopandas as gpd
import networkx as nx
import random
from pyproj.exceptions import CRSError

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/piccard")))

import piccard as pc

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


def test_preprocessing_default_inputs(create_datasets):
    '''
    Test the preprocessing function with the default CRS.
    '''
    census_dfs = create_datasets
    preprocessed_df = pc.preprocessing(census_dfs[0], '2006', "GeoUID")
    # check that area was successfully added
    assert "area_2006" in preprocessed_df.columns


def test_preprocessing_different_crs(create_datasets):
    '''
    Test the preprocessing function with a different valid CRS.
    '''
    census_dfs = create_datasets
    preprocessed_df = pc.preprocessing(census_dfs[0], '2006', "GeoUID", crs="EPSG:4326")
    # check that area was successfully added
    assert "area_2006" in preprocessed_df.columns


def test_preprocessing_invalid_crs(create_datasets):
    '''
    Test that a CRSError is thrown when an invalid CRS is input.
    '''
    census_dfs = create_datasets
    # check that CRSError successfully thrown
    with pytest.raises(CRSError):
        pc.preprocessing(census_dfs[0], '2006', "GeoUID", crs="NotACRS")


def test_create_network_default_inputs(create_datasets):
    '''
    Test the create_network function with all years in the dataset and the default crs and threshold.
    '''
    census_dfs = create_datasets
    years = ['2006', '2011', '2016', '2021']
    G = pc.create_network(census_dfs, years, 'GeoUID')
    
    # take a sample subgraph from 4 random nodes in 2006 and all nodes with edges incident to those nodes
    sample_nodes = []
    sample_nodes_iteration = []
    year_nodes = [node for node in list(G.nodes(data=True)) if node[0][:4] == years[0]]
    for _ in range(4):
        rand = random.randrange(len(year_nodes))
        sample_nodes.append(year_nodes[rand][0])
        sample_nodes_iteration.append(year_nodes[rand])
    for node in list(G.nodes(data=True)):
        if any([G.has_edge(node[0], sample_node[0]) for sample_node in sample_nodes_iteration]):
            sample_nodes.append(node[0])
            sample_nodes_iteration.append(node)
    subgraph = G.subgraph(sample_nodes)

    # check that the subgraph contains at most 4 connected components, all of which are trees
    # in the case of shrinking population, it's possible there could be less than 4 connected components
    assert nx.number_connected_components(subgraph) <= 4
    for node in list(subgraph.nodes(data=True)):
        node_year = node[0][:4]
        if node_year == years[0]:
            connected_nodes = nx.node_connected_component(subgraph, node[0])
            assert nx.is_tree(subgraph.subgraph(list(connected_nodes)))
        # check that every connection corresponds to consecutive census years (every 5 years with this example bc canada)
        for neighbour in subgraph.neighbors(node[0]):
            neighbour_year = neighbour[:4]
            assert abs(int(node_year) - int(neighbour_year)) == 5


def test_create_network_years_subset(create_datasets):
    '''
    Test the create_network function with a subset of years in the dataset and the default crs and threshold.
    '''
    census_dfs = create_datasets[:-1]
    years = ['2006', '2011', '2016']
    G = pc.create_network(census_dfs, years, 'GeoUID')
    
    # take a sample subgraph from 4 random nodes in 2006 and all nodes with edges incident to those nodes
    sample_nodes = []
    sample_nodes_iteration = []
    year_nodes = [node for node in list(G.nodes(data=True)) if node[0][:4] == years[0]]
    for _ in range(4):
        rand = random.randrange(len(year_nodes))
        sample_nodes.append(year_nodes[rand][0])
        sample_nodes_iteration.append(year_nodes[rand])
    for node in list(G.nodes(data=True)):
        if any([G.has_edge(node[0], sample_node[0]) for sample_node in sample_nodes_iteration]):
            sample_nodes.append(node[0])
            sample_nodes_iteration.append(node)
    subgraph = G.subgraph(sample_nodes)

    # check that the subgraph contains at most 4 connected components, all of which are trees
    # in the case of shrinking population, it's possible there could be less than 4 connected components
    assert nx.number_connected_components(subgraph) <= 4
    for node in list(subgraph.nodes(data=True)):
        node_year = node[0][:4]
        if node_year == years[0]:
            connected_nodes = nx.node_connected_component(subgraph, node[0])
            assert nx.is_tree(subgraph.subgraph(list(connected_nodes)))
        # check that every connection corresponds to consecutive census years (every 5 years with this example bc canada)
        for neighbour in subgraph.neighbors(node[0]):
            neighbour_year = neighbour[:4]
            assert abs(int(node_year) - int(neighbour_year)) == 5


def test_create_network_different_threshold(create_datasets):
    '''
    Test the create_network function with all years in the dataset and the default crs, but a different threshold.
    '''
    census_dfs = create_datasets
    years = ['2006', '2011', '2016', '2021']
    G = pc.create_network(census_dfs, years, 'GeoUID', threshold=0.1)
    
    # take a sample subgraph from 4 random nodes in 2006 and all nodes with edges incident to those nodes
    sample_nodes = []
    sample_nodes_iteration = []
    year_nodes = [node for node in list(G.nodes(data=True)) if node[0][:4] == years[0]]
    for _ in range(4):
        rand = random.randrange(len(year_nodes))
        sample_nodes.append(year_nodes[rand][0])
        sample_nodes_iteration.append(year_nodes[rand])
    for node in list(G.nodes(data=True)):
        if any([G.has_edge(node[0], sample_node[0]) for sample_node in sample_nodes_iteration]):
            sample_nodes.append(node[0])
            sample_nodes_iteration.append(node)
    subgraph = G.subgraph(sample_nodes)

    # check that the subgraph contains at most 4 connected components, all of which are trees
    # in the case of shrinking population, it's possible there could be less than 4 connected components
    assert nx.number_connected_components(subgraph) <= 4
    for node in list(subgraph.nodes(data=True)):
        node_year = node[0][:4]
        if node_year == years[0]:
            connected_nodes = nx.node_connected_component(subgraph, node[0])
            assert nx.is_tree(subgraph.subgraph(list(connected_nodes)))
        # check that every connection corresponds to consecutive census years (every 5 years with this example bc canada)
        for neighbour in subgraph.neighbors(node[0]):
            neighbour_year = neighbour[:4]
            assert abs(int(node_year) - int(neighbour_year)) == 5


def test_create_network_table_default_inputs(create_datasets):
    '''
    Test the create_network_table function with all years in the dataset and the default crs and threshold.
    '''
    census_dfs = create_datasets
    years = ['2006', '2011', '2016', '2021']
    G = pc.create_network(census_dfs, years, 'GeoUID')
    network_table = pc.create_network_table(census_dfs, years, 'GeoUID')

    # check that the created network table object has the correct parameters
    assert network_table.years == years
    assert network_table.id == 'GeoUID'

    # for a random sample of 4 rows in the network table, make sure each row corresponds to a valid path through G
    table = network_table.table
    for _ in range(4):
        rand = random.randrange(len(table))
        sample_row = table.iloc[rand]
        sample_row_ids = [sample_row[f'geouid_{year}'] for year in years]
        for i in range(len(sample_row_ids) - 1):
            assert G.has_edge(sample_row_ids[i], sample_row_ids[i+1])


def test_create_network_table_years_subset(create_datasets):
    '''
    Test the create_network_table function with a subset of years in the dataset and the default crs and threshold.
    '''
    census_dfs = create_datasets[:-1]
    years = ['2006', '2011', '2016']
    G = pc.create_network(census_dfs, years, 'GeoUID')
    network_table = pc.create_network_table(census_dfs, years, 'GeoUID')

    # check that the created network table object has the correct parameters
    assert network_table.years == years
    assert network_table.id == 'GeoUID'

    # for a random sample of 4 rows in the network table, make sure each row corresponds to a valid path through G
    table = network_table.table
    for _ in range(4):
        rand = random.randrange(len(table))
        sample_row = table.iloc[rand]
        sample_row_ids = [sample_row[f'geouid_{year}'] for year in years]
        for i in range(len(sample_row_ids) - 1):
            assert G.has_edge(sample_row_ids[i], sample_row_ids[i+1])


def test_create_network_table_different_threshold(create_datasets):
    '''
    Test the create_network_table function with all years in the dataset and the default crs, but a different threshold.
    '''
    census_dfs = create_datasets
    years = ['2006', '2011', '2016', '2021']
    G = pc.create_network(census_dfs, years, 'GeoUID')
    network_table = pc.create_network_table(census_dfs, years, 'GeoUID', threshold=0.1)

    # check that the created network table object has the correct parameters
    assert network_table.years == years
    assert network_table.id == 'GeoUID'

    # for a random sample of 4 rows in the network table, make sure each row corresponds to a valid path through G
    table = network_table.table
    for _ in range(4):
        rand = random.randrange(len(table))
        sample_row = table.iloc[rand]
        sample_row_ids = [sample_row[f'geouid_{year}'] for year in years]
        for i in range(len(sample_row_ids) - 1):
            assert G.has_edge(sample_row_ids[i], sample_row_ids[i+1])
