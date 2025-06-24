import warnings
import pandas as pd
import geopandas as gpd
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import math
import random
from typing import Optional, List

def preprocessing(
    data: gpd.GeoDataFrame, 
    year: str, 
    id: str
) -> gpd.GeoDataFrame:
  '''
    Returns a cleaned geopandas df of the input data.
    Note: Input data is assumed to have been passed through gpd.read_file() beforehand.

    Parameters:
        data (GeoDataFrame):
            The census data to be analyzed with piccard.

        year (str):
            The year that the census data was collected.

        id (str):
            The name of the unique identifier that will be used to distinguish geographical areas.
    
    Returns:
        GeoDataFrame: the cleaned data
  '''
  process_data = data.copy()

  #Suppressing CRS warning associated with .buffer()
  with warnings.catch_warnings():
      warnings.simplefilter(action='ignore', category=UserWarning)
      process_data['geometry'] = (process_data.to_crs('EPSG:4246').geometry
                                  .buffer(-0.000001))
      process_data['area' + '_' + year] = process_data.area
  process_data[id] = year + '_' + process_data[id]

  return process_data


def create_network(
    census_dfs: List[gpd.GeoDataFrame], 
    years: List[str], 
    id: str, 
    threshold: Optional[float] = 0.05
) -> nx.Graph:
  '''
  Creates a network representation of the temporal connections present in `census_dfs` over `years` 
  when each yearly geographic area has at most `threshold` percentage of overlap with its 
  corresponding area(s) in the next year. Represents geographical areas as nodes, and temporal connections
  as edges.

  Parameters:
      census_dfs (List[gpd.GeoDataFrame]):
          A list of GeoDataFrames containing the census data to be turned into a network.

      years (List[str]):
          A list of years present in census_dfs over which the network representation will be created.
          Data from years not present in years will be ignored.
      
      id (str):
          The name of the unique identifier that will be used to distinguish geographical areas.

      threshold (float | None):
          The percentage of overlap (divided by 100)
          that geographic areas must meet or exceed in order to have a connection.
          Default is 0.05, or 5 percent.    

  Returns:
      nx.Graph: The networkx graph containing the nodes (geographical areas) and edges (geographical overlap)
          created in the new network representation.

  '''
  preprocessed_dfs = [preprocessing(census_dfs[i], years[i], id) for i in range(len(census_dfs))]
  contained_cts = ct_containment(preprocessed_dfs, years)

  nodes = get_nodes(contained_cts, id, threshold)
  attributes = get_attributes(nodes, census_dfs, years, id)

  G = nx.from_pandas_edgelist(nodes, f'{id}_1', f'{id}_2')
  nx.set_node_attributes(G, attributes.set_index(id).to_dict('index'))

  return G


def create_network_table(
    census_dfs: List[gpd.GeoDataFrame], 
    years: List[str], 
    id: str, 
    threshold: Optional[float] = 0.05
) -> pd.DataFrame:
  '''
  Creates a pandas DataFrame showing the network representation of the census data in census_dfs. 
  Each feature present in the data is a column, and each possible path through the network is a row.

  Parameters:
      census_dfs (List[gpd.GeoDataFrame]):
          A list of GeoDataFrames containing the census data to be turned into a network.

      years (List[str]):
          A list of years present in census_dfs over which the network representation will be created.
          Data from years not present in years will be ignored.
      
      id (str):
          The name of the unique identifier that will be used to distinguish geographical areas.

      threshold (float | None):
          The percentage of overlap (divided by 100)
          that geographic areas must meet or exceed in order to have a connection.
          Default is 0.05, or 5 percent.    

  Returns:
      pd.DataFrame: the table.
  '''
  num_years = len(years)
  num_joins = math.ceil(num_years/2)
  final_cols = [id + '_' + col_name for col_name in years]
  network_table = pd.DataFrame()
  drop_cols = final_cols[1:]

  preprocessed_dfs = [preprocessing(census_dfs[i], years[i], id) for i in range(len(census_dfs))]
  contained_cts = ct_containment(preprocessed_dfs, years)
  nodes = get_nodes(contained_cts, id, threshold)

  #all_paths returns a three item tuple
  all_paths = find_all_paths(nodes, num_joins, id)
  all_paths_df = all_paths[0]
  left_cols = all_paths[1]
  # right_cols = all_paths[2]

  #Dividing all network paths into full paths and partial paths
  na_df = all_paths_df[all_paths_df.isnull().any(axis=1)]
  no_na_df = all_paths_df[~all_paths_df.isnull().any(axis=1)]

  full_paths = find_full_paths(no_na_df, final_cols)
  full_paths_list = full_paths.to_numpy().flatten()

  partial_paths = find_partial_paths(na_df, years, left_cols, final_cols, full_paths_list)

  network_table = pd.concat([full_paths, partial_paths])
  network_table = network_table[final_cols]
  network_table = network_table.T.drop_duplicates().T
  network_table = network_table.drop_duplicates(subset=drop_cols, keep='last')
  network_table.sort_values(by=final_cols[0], ignore_index=True)

  attributes = get_attributes(nodes, census_dfs, years, id)
  final_table = attach_attributes(network_table, attributes, years, final_cols, id)

  #Formatting final table columns
  for i in range(len(final_cols)):
      col = str(final_cols[i])
      popped = final_table.pop(col)
      final_table.insert(i, popped.name, popped)
  final_table.columns= final_table.columns.str.lower()

  return final_table


def plot_subnetwork(
    network_table: pd.DataFrame, 
    G: nx.Graph, 
    years: Optional[List[str]] = None,
    paths_to_show: Optional[List[int]] = None,
    ids_to_show: Optional[List[str]] = None,
    num_to_sample: Optional[int] = 4
) -> go.Figure:
    """
    Draws a subgraph of the network representation. If neither a specific list of ids to show nor a specific
    list of paths to show are given, picks num_to_sample random nodes from the first census year in the data
    and plots a subnetwork of their paths.
    Hovering over each node shows the paths the node is part of.

    Parameters:
        network_table (pd.DataFrame):
            The result of pc.create_network_table().
        
        G (nx.Graph):
            The result of pc.create_network().

        years (List[str] | None):
            A list of years to show in the subnetwork. Default is all census years present in the data.

        paths_to_show (List[int] | None):
            A list of paths (numbered according to their position in network_table) whose points 
            will be plotted in the subnetwork.

        ids_to_show (List[str] | None):
            A list of ids (use the same id you used when creating the graph and network table) that
            will be plotted in the subnetwork. If both paths_to_show and ids_to_show are given, the function
            will only consider ids_to_show.

        num_to_sample (int | None):
            The number of random nodes to plot the paths of in the subnetwork. Default is 4. 
            Note: A large num_to_sample value may result in an unorganized and hard-to-read visualization.
    
    Returns:
        go.Figure: 
            The interactive subnetwork plot.
    """
    # create valid list of years
    all_years = sorted(list({int(year[0][:4]) for year in G.nodes(data=True)}))
    if years is None:
        years = all_years
    else:
        years = [year for year in years if int(year) in all_years]
        if len(years) == 0:
            years = all_years
    all_years = [str(year) for year in all_years]
    years = [str(year) for year in years]

    # organize node names by year and network table path number
    paths_for_each_year = [list(network_table[network_table.columns[i]]) for i in range(len(years))]

    # prepare nodes to be graphed
    sample_nodes = []
    sample_nodes_iteration = []
    # get nodes by id
    if ids_to_show is not None:
        for year in years:
            for id in ids_to_show:
                sample_nodes.append(f'{year}_{id}')
    # get nodes by network table path
    elif paths_to_show is not None:
        for year in years:
            year_index = all_years.index(year)
            for i in paths_to_show:
                sample_nodes.append(paths_for_each_year[year_index][i])
    # get nodes by random sample
    else:
        year_nodes = [node for node in list(G.nodes(data=True)) if node[0][:4] == years[0]]
        for _ in range(num_to_sample):
            rand = random.randrange(len(year_nodes))
            sample_nodes.append(year_nodes[rand][0])
            sample_nodes_iteration.append(year_nodes[rand])
        for node in list(G.nodes(data=True)):
            if any([G.has_edge(node[0], sample_node[0]) for sample_node in sample_nodes_iteration]):
                sample_nodes.append(node[0])
                sample_nodes_iteration.append(node)

    # create the graph
    subgraph = G.subgraph(sample_nodes)
    pos = nx.multipartite_layout(subgraph, subset_key='network_level')

    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    # customize node information
    node_x = []
    node_y = []
    text = []
    title_text = []
    for node in subgraph.nodes():
        paths = ''
        for year in years:
            year_index = all_years.index(year)
            for i in range(len(paths_for_each_year[year_index])):
                if paths_for_each_year[year_index][i] == node:
                    paths = paths + f'Path {i}, '
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(f'ID: {node}    Paths: {paths[:-2]}')
        title_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=title_text,
        textposition='top center',
        hoverinfo='text',
        hovertext=text,
        marker=dict(
            showscale=False,
            color='orange',
            size=10,
            line=dict(width=2)
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Subnetwork',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=30,l=30,r=30,t=80),
                        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
                    ))
    return fig


def plot_num_areas(
    network_table: pd.DataFrame, 
    years: Optional[List[str]] = None,
) -> go.Figure:
    '''
    Plots the number of geographical areas across a subset of census years in the data.

    Parameters:
        network_table (pd.DataFrame):
            The result of pc.create_network_table().

        years (List[str] | None):
            A list of years to show in the subnetwork. Default is all census years present in the data.

    Returns:
        go.Figure:
            The plot of the number of geographical areas.
    '''
    id_label = network_table.columns[0][:-5]
    if years is None:
        year_cols = [col for col in network_table.columns.to_list() if id_label in col]
        years = sorted(list({col[-4:] for col in year_cols}))

    ct_per_year = []
    num_years = len(years)
    for i in range(num_years):
        ids_list = list(network_table[network_table.columns[i]])
        ct_per_year.append(len({id for id in ids_list}))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=years,
        y=ct_per_year,
        mode='lines+markers',
        line=dict(color='royalblue'),
        marker=dict(size=8),
        name=f'Number of {id_label}s'
    ))

    fig.update_layout(
        title=f'Number of {id_label}s from {years[0]} to {years[num_years - 1]}',
        xaxis_title='Year',
        yaxis_title=f'Number of {id_label}s',
        width=700,
        height=500,
    )

    return fig


def ct_containment(preprocessed_dfs, years):
  '''
  Returns a GeoDataFrame with census tracts which are contained
  within a census tract from the following census
  '''
  num_years = len(years)
  contained_tracts = []

  for i in range(num_years-1):
      #Getting CTs which are contained within a previous year's CT
      contained_df = gpd.overlay(preprocessed_dfs[i], preprocessed_dfs[i+1],
                                  how='intersection')
      with warnings.catch_warnings():
          warnings.simplefilter(action='ignore', category=UserWarning)

          contained_df['area_intersection'] = contained_df.area
          #Calculating the percentage of the overlapping area between the 2 years
          pct_col = 'pct_' + years[i+1] + '_of_' + years[i]
          contained_df[pct_col] = (contained_df['area_intersection'] /
                                    contained_df[['area_'+years[i],
                                                  'area_'+years[i+1]]].min(axis=1))
      contained_tracts.append(contained_df)
  return contained_tracts


def get_nodes(contained_tracts_df, id, threshold=0.05):
  '''
  Returns a GeoDataFrame with the graph connections between two census tracts
  of different years. Each row corresponds to one edge in the final network.
  '''
  nodes = gpd.GeoDataFrame()
  id_cols = [f'{id}_1', f'{id}_2']

  #Aggregating overlapped percentage area for all unique CTs
  for i in range(len(contained_tracts_df)):
      pct_col = contained_tracts_df[i].iloc[:, -1].name
      year_pct = (contained_tracts_df[i]
                  .groupby(id_cols)
                  .agg({f'{pct_col}': 'sum'})
                  .reset_index()
                  )

      #Selecting CTs with an overlapped area above user's threshold
      connected_cts = year_pct[year_pct[pct_col] >= threshold][id_cols]
      nodes = pd.concat([nodes, connected_cts], axis=0, ignore_index=True)

  return nodes


def assign_node_level(row, years, id):
  """
  Assigns the level of a node in the network based on its relative year in the
  network
  Example: All 2021 nodes are in level 3 in a graph with years 2011, 2016, 2021
  """
  for i in range(len(years)):
    if row[id].startswith(str(years[i])):
      return i+1
    

def get_attributes(nodes, census_dfs, years, id):
  '''
  Returns all the attributes in the original data corresponding to the network
  nodes
  '''
  #Condensing nodes into single column df
  single_nodes = pd.concat([nodes[col] for col in nodes]).reset_index(drop=True)
  single_nodes_df = pd.DataFrame({id: single_nodes})
  attr = []

  for i in range(len(census_dfs)):
      #Adding year as a prefix for the merge
      curr_df_id = census_dfs[i].loc[:, id]
      curr_df_id = years[i] + '_' + curr_df_id

      #Removing geometry column in attributes for the final table
      year_attr = census_dfs[i].loc[:, (census_dfs[i].columns != 'geometry')].copy()
      year_attr[id] = curr_df_id
      year_attr = pd.merge(single_nodes_df, year_attr, on=id, how='right')

      attr.append(year_attr)
  all_attr = (pd.concat(attr)).drop_duplicates(subset=id)
  all_attr = all_attr[all_attr[id].notna()]

  #Assigning each node its level in the network (used for mainly drawing)
  all_attr['network_level'] = all_attr.apply(lambda x: assign_node_level(x, years, id), axis=1)
  return all_attr


def find_all_paths(nodes_df, num_joins, id):
  '''
  Return all possible paths present in the input data.
  Note: The resulting dataframe is not organized and does contain
        duplicate entries in both the rows and columns.
  '''
  left_cols = [f'{id}_1_x', f'{id}_2_x']
  right_cols = [f'{id}_1_y', f'{id}_1_x']

  #Merging network nodes num_joins amount of times to ensure all paths are found
  curr_join = nodes_df.merge(nodes_df, how='left', left_on=f'{id}_1', right_on=f'{id}_2')
  curr_join = curr_join.sort_values(by=[f'{id}_1_y', f'{id}_2_y'], ignore_index=True)

  if num_joins > 1:
      for i in range(num_joins - 1):
          curr_join = curr_join.merge(curr_join, how='left', left_on=left_cols, right_on=right_cols, suffixes=['x', 'y'])
          #Accounting for the new column names after the merge
          left_cols = [col_name + 'x' for col_name in left_cols]
          right_cols = [col_name + 'x' for col_name in right_cols]
  return (curr_join, left_cols, right_cols)


def find_full_paths(full_paths_df, final_cols):
  '''
  Return all full paths present in input data.
  Note: Define a full path as a path in the network where the starting node is
        from the first input year and the ending node is from the last input year.
  '''
  full_paths = pd.DataFrame()

  if (not full_paths_df.empty):
      full_paths = full_paths_df.T.drop_duplicates().sort_values(by=0).T
      full_paths.columns = final_cols
  return full_paths


def first_year_partial_paths(all_partial_paths, years, final_cols):
  '''
  Return all partial paths only for the first input year.
  Note: Define a partial path as a path in the network where the starting and
        ending nodes are of any year (i.e., not a full path).
  '''
  num_years = len(years)
  drop_cols = final_cols[1:]

  #Selecting paths with the starting node as the first year
  mask = all_partial_paths.iloc[:, 0].str.startswith(years[0] + '_')
  first_year_partials = all_partial_paths[mask]

  #Checking if df empty or not
  if len(first_year_partials.index) != 0:
    #Calculating which year contains the ending node
    max_partial_year = max(all_partial_paths.T.stack().values)[:4]

    #Appending NaN columns to the end for each year as they don't exist in data
    if ((max_partial_year >= years[1]) & (max_partial_year != years[-1])):
        for i in reversed(range((num_years - 1) - max_partial_year)):
            last_col = len(first_year_partials.columns)
            first_year_partials.insert(last_col, final_cols[-i], np.nan)
        first_year_partials.columns = final_cols
    first_year_partials = first_year_partials.T.drop_duplicates().dropna().T
    first_year_partials.columns = final_cols
    return first_year_partials
  else:
    empty_df = pd.DataFrame(columns = final_cols)
    return empty_df
  

def unique_partial_paths(all_partial_paths, years, left_cols, final_cols):
  '''
  Return all unique partial paths between two consecutive input years.
  Note: Define a partial path as a path in the network where the starting and
        ending nodes are of any year (i.e., not a full path).
  '''
  num_years = len(years)
  unique_partials = pd.DataFrame()

  for i in range(1, num_years):
      curr_year = years[i] + '_'
      prev_year = years[i-1] + '_'

      curr_year_mask = all_partial_paths.iloc[:, 0].str.startswith(curr_year)
      prev_year_mask = all_partial_paths.iloc[:, 0].str.startswith(prev_year)

      curr_year_partials = all_partial_paths[curr_year_mask]
      prev_year_partials = all_partial_paths[prev_year_mask]

      curr_year_mask = ~curr_year_partials[left_cols[0]].isin(prev_year_partials)
      curr_year_unique = curr_year_partials[curr_year_mask]
      curr_year_unique = curr_year_partials.dropna(axis=1).T.drop_duplicates().T

  #Appending NaN column to the front to account for missing first year
      for k in range(i):
          curr_year_unique.insert(0, final_cols[k], np.nan)

  #Appending NaN column to the end to account for missing last year
      if(not curr_year_unique.empty):
          curr_year_val = max(curr_year_unique.T.stack().values)[:4]
          curr_year_index = years.index(curr_year_val)

          if (curr_year_index != years[-1]):
              for j in range((num_years - 1) - curr_year_index):
                  last_col = len(curr_year_unique.columns)
                  curr_year_unique.insert(last_col, final_cols[-j], np.nan)

          curr_year_unique.columns = final_cols
      unique_partials = pd.concat([unique_partials, curr_year_unique])
  return unique_partials


def find_partial_paths(partial_paths_df, years, left_cols, final_cols, exclude_nodes):
  '''
  Return all partial paths present in input data.
  Note: Define a partial path as a path in the network where the starting and
        ending nodes are of any year (i.e., not a full path).
  '''

  all_partial_paths = partial_paths_df.T.drop_duplicates().T
  all_partial_paths = all_partial_paths[~all_partial_paths[left_cols[0]].isin(exclude_nodes)]

  first_year_partials = first_year_partial_paths(all_partial_paths, years, final_cols)
  unique_partials = unique_partial_paths(all_partial_paths, years, left_cols, final_cols)
  all_partials = pd.concat([unique_partials, first_year_partials])

  return all_partials


def attach_attributes(network_table, attributes, years, final_cols, id):
  '''
  Return network table with attached attributes corresponding to the nodes
  involved.
  '''
  years_df_list = []

  for i in range(len(final_cols)):
      col = str(final_cols[i])

      #Getting attributes for each year
      table_col = network_table[col].to_frame().astype(object)
      curr_year = table_col.merge(attributes, how='left', left_on=col, right_on=id)
      curr_year = curr_year.drop([id], axis=1)

      #Suppressing warning for str.replace
      with warnings.catch_warnings():
          warnings.simplefilter(action='ignore', category=FutureWarning)
          curr_year = curr_year.apply(lambda x: x.str.replace(r'[0-9]+_', '') if x.dtypes==object else x).reset_index(drop=True)

          #Formatting all columns as 'colname_year'
          curr_year_cols = [f'{col}_{years[i]}' if col != final_cols[i] and col != f'area_{years[i]}' else col for col in curr_year.columns]
          curr_year.columns = curr_year_cols
          years_df_list.append(curr_year)

  #Combining all years dfs into one
  network_table = (pd.concat(years_df_list, axis=1)).dropna(how='all', axis=1)
  return network_table
