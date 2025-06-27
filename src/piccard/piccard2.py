import math
import random
import numpy as np
import geopandas as gpd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import shapely
from itertools import cycle, islice
from tscluster.opttscluster import OptTSCluster
from tscluster.greedytscluster import GreedyTSCluster
from tscluster.preprocessing.utils import load_data, tnf_to_ntf, ntf_to_tnf
import pandas as pd # for type annotations
import networkx as nx # for type annotations
from typing import Union, Any, List, Tuple, Optional # for type annotations
from plotly.subplots import make_subplots

import time

import warnings
warnings.filterwarnings('ignore')

# Network Creation

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

  # Suppressing CRS warning associated with .buffer()
  with warnings.catch_warnings():
      warnings.simplefilter(action='ignore', category=UserWarning)

      if process_data.crs != 'EPSG:4246':
          process_data = process_data.to_crs('EPSG:4246')

      # Only buffer rows where geometry complexity is high
      def is_complex(g):
        try:
            return len(g.exterior.coords) > 500 if g.geom_type == 'Polygon' else False
        except:
            return False

      mask = process_data.geometry.apply(is_complex)
      process_data.loc[mask, 'geometry'] = shapely.buffer(process_data.loc[mask, 'geometry'], -0.000001)

      process_data['area' + '_' + year] = process_data.area
  
  process_data[id] = year + '_' + process_data[id]

  return process_data


def create_network(
    census_dfs: List[gpd.GeoDataFrame], 
    years: List[str], 
    id: str, 
    threshold: Optional[float] = 0.05,
    verbose: Optional[bool] = True
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
      
      verbose (bool | None):
          Whether to issue print statements about the progress of network creation. Default is true.

  Returns:
      nx.Graph: The networkx graph containing the nodes (geographical areas) and edges (geographical overlap)
          created in the new network representation.

  '''
  preprocessed_dfs = [preprocessing(census_dfs[i], years[i], id) for i in range(len(census_dfs))]
  if verbose:
      print('Preprocessing complete')
  contained_cts = ct_containment(preprocessed_dfs, years)

  nodes = get_nodes(contained_cts, id, threshold)
  if verbose:
      print('All nodes found')
  attributes = get_attributes(nodes, census_dfs, years, id)
  if verbose:
      print('All attributes found')

  G = nx.from_pandas_edgelist(nodes, f'{id}_1', f'{id}_2')
  nx.set_node_attributes(G, attributes.set_index(id).to_dict('index'))
  if verbose:
      print('Graph created')

  return G


def create_network_table(
    census_dfs: List[gpd.GeoDataFrame], 
    years: List[str], 
    id: str, 
    threshold: Optional[float] = 0.05,
    verbose: Optional[bool] = True
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

      verbose (bool | None):
          Whether to issue print statements about the progress of network creation. Default is true.

  Returns:
      pd.DataFrame: the table.
  '''
  num_years = len(years)
  num_joins = math.ceil(num_years/2)
  final_cols = [id + '_' + col_name for col_name in years]
  network_table = pd.DataFrame()
  drop_cols = final_cols[1:]

  preprocessed_dfs = [preprocessing(census_dfs[i], years[i], id) for i in range(len(census_dfs))]
  if verbose:
      print('Preprocessing complete')
  contained_cts = ct_containment(preprocessed_dfs, years)
  nodes = get_nodes(contained_cts, id, threshold)
  if verbose:
      print('All nodes found')

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
  if verbose:
      print('All possible paths through the graph found')

  network_table = pd.concat([full_paths, partial_paths])
  network_table = network_table[final_cols]
  network_table = network_table.T.drop_duplicates().T
  network_table = network_table.drop_duplicates(subset=drop_cols, keep='last')
  network_table.sort_values(by=final_cols[0], ignore_index=True)

  attributes = get_attributes(nodes, census_dfs, years, id)
  final_table = attach_attributes(network_table, attributes, years, final_cols, id)
  if verbose:
      print('All attributes found')

  #Formatting final table columns
  for i in range(len(final_cols)):
      col = str(final_cols[i])
      popped = final_table.pop(col)
      final_table.insert(i, popped.name, popped)
  final_table.columns= final_table.columns.str.lower()
  if verbose:
      print('Table created')

  return final_table


# Clustering

def clustering_prep(
    network_table: pd.DataFrame, 
    id: str, 
    cols: Optional[list[str]]=[]
) -> tuple[np.ndarray[np.float64], dict[str, Any]]:
    '''
    Converts a piccard network table into a 3d numpy array of all possible paths and their corresponding
    features. This will be used for clustering with tscluster.
    The user can (optionally) input a list of columns that they want to be considered in the clustering algorithm, 
    and the function will check that these columns are valid.

    Note that you must run pc.create_network_table() before this function.

    Parameters:
        network_table (pd.DataFrame): 
            The result of pc.create_network_table().
        
        id (str): 
            The same id inputted into pc.create_network_table().
 
        cols (list[str]): A list of the names of network table columns that should be considered in
            the clustering algorithm. If none, every numerical feature will be considered. Leaving it none is
            not recommended as many numerical features, such as network level, have little bearing on the data.

    Returns:
        (tuple[np.ndarray[np.float64], dict[str, Any]]):
            a tuple of a 3d numpy array and a corresponding dictionary of labels showing
            the shape of the array.
    '''
    # default to considering all features
    if cols == []:
        cols = network_table.columns.to_list()

    # Find all years present in the data. These will be used as timesteps for tscluster.
    year_cols = [col for col in network_table.columns.to_list() if id in col]
    years = sorted(list({col[-4:] for col in year_cols}))

    # Filter columns
    filtered_cols = filter_columns(network_table, years, cols)

    # Extract features for each year and add them to a 2D array representing that year. 
    # Then add that array to a list of arrays representing the 3D array used for tscluster.
    list_of_arrays = []
    for year in years:
        year_statistics = network_table[[col for col in filtered_cols[0] if year in col]].to_numpy()
        list_of_arrays.append(year_statistics)
    
    # Filter out entities whose features are entirely NaN
    # Run load_data now so we get access to variables necessary for tnf_to_ntf
    list_of_arrays = load_data(list_of_arrays)[0]
    ntf_list_of_arrays = tnf_to_ntf(list_of_arrays)
    count = -1
    for entity in ntf_list_of_arrays:
        count += 1
        number_in_entity = False
        for i in entity.flat:
            if not np.isnan(i):
                number_in_entity = True
                break
        if not number_in_entity:
            np.delete(ntf_list_of_arrays, count, 0)

    # Interpolate remaining nan values for clustering
    for entity in ntf_list_of_arrays:
        transposed_entity = entity.T
        for row in transposed_entity:
            nans = np.isnan(row)
            x = np.arange(len(row))
            row[nans] = np.interp(x[nans], x[~nans], row[~nans])

    list_of_arrays = ntf_to_tnf(ntf_list_of_arrays)
                
    # Return the final numpy array and create a corresponding label dictionary.
    # This can then be preprocessed using tscluster's scalers.
    label_dict = {'T': years, 'N': [i for i in range(count + 1)], 'F': filtered_cols[1]}
    return (list_of_arrays, label_dict)


def cluster(
    network_table: pd.DataFrame, 
    G: nx.Graph, 
    id: str, 
    num_clusters: int, 
    algo: Optional[str]='greedy', 
    scheme: Optional[str]='z1c1', 
    arr: Optional[np.ndarray[np.float64]]=None, 
    label_dict: Optional[dict[str, Any]]=None
) -> Union[OptTSCluster, GreedyTSCluster]:
    '''
    Runs one of tscluster's clustering algorithms (default is fully dynamic clustering or 'z1c1')
    and adds the resulting cluster assignments to the network table and nodes as an additional feature.
    Information about the different clustering algorithms is available here: https://tscluster.readthedocs.io/en/latest/introduction.html
    We recommend either Sequential Label Analysis ('z1c0') or the default 'z1c1'.

    Users can choose to only input the network table, in which case clustering_prep will be run for them with the default columns,
    or they can choose to run clustering_prep on their own and then have the option to apply one or both of the
    normalization methods available in tscluster.preprocessing.utils.

    Parameters:
        network_table (pd.DataFrame): 
            The result of pc.create_network_table().

        G (nx.Graph): 
            The result of pc.create_network().

        id (str): 
            The same id inputted into pc.create_network_table().

        num_clusters (int): 
            The number of clusters that the algorithm will find.

        algo (str | None): 
            The algorithm that tscluster will use, either 'greedy' (default) or 'opt'.
            'greedy' runs GreedyTSCluster, which is a faster and easier, but less accurate, method than OptTSCluster. 
            Since it doesn't require a special academic licence, we recommend 'greedy' for any non-academic users.
            'opt' runs OptTSCluster, which is guaranteed to find the optimal clustering but requires a Gurobi academic
            licence to run the clustering algorithm. More information about obtaining an academic licence can be found
            here: https://www.gurobi.com/academia/academic-program-and-licenses/
        
        scheme (str | None): 
            the clustering scheme. See the first paragraph for more information. Default is 'z1c1'.

        arr (np.ndarray[np.float64] | None): 
            the array of data to be clustered. If none, arr and label_dict will be generated by running
            pc.clustering_prep() with the default columns. See the pc.clustering_prep() documentation for why we DO NOT
            recommend leaving this blank.
        
        label_dict (dict[str, Any] | None): 
            the label dictionary corresponding to the data array. See 'arr'.

    Returns:
        (OptTSCluster | GreedyTSCluster): 
            an OptTSCluster or GreedyTSCluster object with useful labels, cluster assignments, etc 
            for future visualizations.
    '''
    # Get the data into the correct format. See the documentation for clustering_prep
    if arr is None and label_dict is None:
        arr, label_dict = clustering_prep(network_table, id)
    
    # Ensure valid scheme
    if scheme.lower() != 'z0c0' and scheme.lower() != 'z0c1' and scheme.lower() != 'z1c0' and scheme.lower() != 'z1c1':
        raise ValueError("Please ensure scheme is either z0c0, z0c1, z1c0, or z1c1. See tscluster documentation.")

    # Initialize the model
    if algo.lower() == 'opt':
        tsc = OptTSCluster(
            n_clusters=num_clusters,
            scheme=scheme,
            n_allow_assignment_change=None, # Allow as many changes as possible
            random_state=3
        )
    elif algo.lower() == 'greedy':
        tsc = GreedyTSCluster(
            n_clusters=num_clusters,
            scheme=scheme,
            n_allow_assignment_change=None, # Allow as many changes as possible
            random_state=3
        )
    else:
        raise ValueError("Please ensure algo is either greedy or opt.")
    
    # Assign clusters
    tsc.fit(arr, label_dict=label_dict)

    # Add cluster assignments to network table
    cluster_assignments_table = tsc.get_named_labels(label_dict=label_dict)
    years = label_dict['T']
    for year in years:
        network_table[f'cluster_assignment_{year}'] = list(cluster_assignments_table[year])

    # Add cluster assignments to graph nodes
    nodes_list = list(G.nodes(data=True))
    for node in nodes_list:
            year = node[0][:4]
            cluster = network_table.loc[network_table[f'geouid_{year}'] == node[0]]
            if len(cluster) != 0:
                cluster = int(cluster.iloc[0][f'cluster_assignment_{year}'])
                dict = tsc.get_named_cluster_centers(label_dict=label_dict)[cluster].loc[year]
                # figure out which cluster to assign a node to if it's already been assigned to a different cluster
                if 'cluster_assignment' in node[1] and node[1]['cluster_assignment'] != cluster:
                    old_dict = tsc.get_named_cluster_centers(label_dict=label_dict)[node[1]['cluster_assignment']].loc[year]
                    # comparing distances between clusters
                    old_cluster_distance = 0
                    new_cluster_distance = 0
                    for i in range(len(dict)):
                        old_cluster_distance += (math.abs(int(node[1][label_dict['F'][i]]) - int(old_dict[i])))
                        new_cluster_distance += (math.abs(int(node[1][label_dict['F'][i]]) - int(dict[i])))
                    if old_cluster_distance < new_cluster_distance:
                        cluster = node[1]['cluster_assignment']
                node[1]['cluster_assignment'] = cluster
            elif 'cluster_assignment' not in node[1]:
                node[1]['cluster_assignment'] = np.nan
    
    return tsc

# Plot & Visuals

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


def plot_clusters_scatter(
    network_table: pd.DataFrame,
    arr: np.ndarray[np.float64],
    label_dict: dict[str, Any],
    tsc: Union[OptTSCluster, GreedyTSCluster],
    years: Optional[List[str]] = None,
    cluster_colours: Optional[dict[int, str]] = None,
    dynamic_paths_only: Optional[bool] = True,
    paths_to_show: Optional[List[int]] = None,
    ids_to_show: Optional[List[str]] = None,
    clusters_to_show: Optional[List[int]] = None, 
    clusters_to_exclude: Optional[List[int]] = [],
    figsize: Optional[Tuple[float, float]] = (700, 500),
    cluster_labels: Optional[List[str]] = None,
) -> List[go.Figure]:
    '''
    Creates a plotly scatterplot for each variable used in clustering with each timestep 
    on the x axis and values on the y axis. The colours of data points correspond to their assigned cluster,
    and there is a legend showing which colour goes with which cluster. (Cluster numbers start at 0.)
    Since cluster assignment often changes along the same path (or within the same area) over the years,
    plotting all the data points in one cluster often involves considering other clusters as well. Therefore,
    when you select a cluster to plot, you will see every path that contains a point in that cluster, and some
    of these paths will also contain paths in different clusters.
    Add any clusters you don't want to see (e.g. a cluster composed of NaN values) to exclude_clusters. This
    will exclude all paths containing these clusters, even paths that also have paths specified in the
    clusters list. In addition, you can curate the specific paths you want to see with paths_to_show; just
    make sure the paths are numbered according to their position in network_table.

    Parameters:
        network_table (pd.DataFrame):
            The result of pc.create_network_table().

        arr (np.ndarray[np.float64]):
            The numpy array from pc.clustering_prep() that you used in pc.cluster().

        label_dict (dict[str, Any]):
            The label dictionary from pc.clustering_prep() that you used in pc.cluster(). 
            label_dict could also be a custom label dictionary.
        
        tsc (Union[OptTSCluster, GreedyTSCluster]): 
            The result of pc.cluster().

        years (List[str] | None): 
            The years displayed on the map. Default is all years in the network table.

        cluster_colours (dict[int, str] | None):
            A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
            colour map will be used. If a cluster number is not part of the dict, plotly's default
            colour map will be used for that cluster.

        dynamic_paths_only (bool | None): 
            A boolean indicating whether to only plot dynamic entities (entities whose cluster
            assignment has changed over time). Default is true.

        paths_to_show (List[int] | None): 
            A list of paths (numbered according to their position in network_table) whose points 
            will be displayed on the map. Default is every path.

        ids_to_show (List[str] | None):
            A list of ids (use the same id you used when creating the graph and network table) whose points
            will be displayed on the map. Default is every id.
        
        clusters_to_show (List[int] | None): 
            A list of the clusters whose points will be displayed on the map. Default is every cluster.

        clusters_to_exclude (List[int] | None): 
            A list of the clusters whose points will NOT be displayed on the map. Default is
            an empty list.
        
        figsize (Tuple[float, float] | None): 
            A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).
        
        cluster_labels (List[str] | None): 
            A custom list of cluster names. Default is Cluster 0, ..., Cluster n.

    Returns:
        (List[go.Figure]):
            a list of plotly.graph_objects.Figure (you cannot show the whole list; rather, iterate through 
            the list and show each figure)
    '''

    # get necessary data from tsc
    cluster_centres = tsc.cluster_centers_
    labels = tsc.labels_

    # prepare the variables we will use to iterate through features and cluster centres
    F = arr.shape[2]
    K = cluster_centres.shape[1] if cluster_centres is not None else np.unique(labels).size

    # verify years exist in network table
    if years is None:
        years = label_dict['T']
    for year in years:
        column = f'cluster_assignment_{year}'
        if column not in network_table.columns:
            raise ValueError(f"Expected column '{column}' not found in DataFrame.")

    # set default values and colours      
    if paths_to_show is None:
        paths_to_show = list(range(arr.shape[1])) 
    if clusters_to_show is None:
        clusters_to_show = list(range(K))
    if cluster_labels is None:
        cluster_labels = [str(i) for i in range(K)]

    colors = []
    if cluster_colours:
        for i in range(K):
            if i in cluster_colours:
                colors.append(cluster_colours[i])
            else:
                colors.append(plotly.colors.qualitative.Plotly[i])
    else:
        colors = plotly.colors.qualitative.Plotly
        if K > len(colors):
            colors = list(islice(cycle(colors), K))

    # make sure clusters_to_show and clusters_to_exclude only look at cluster assignments in years timeframe
    new_network_table = network_table.copy(deep=True)
    for year in label_dict['T']:
        if year not in years:
            new_network_table[f'cluster_assignment_{year}'] = [
                np.nan for _ in range(len(network_table[f'cluster_assignment_{year}']))]

    # filter entities using paths_to_show, clusters_to_show, clusters_to_exclude, dynamic_paths_only
    paths_to_show = [
        i for i in paths_to_show
        if any(int(c) in clusters_to_show for c in new_network_table.iloc[i][-len(label_dict['T']):])
        and all(int(c) not in clusters_to_exclude for c in new_network_table.iloc[i][-len(label_dict['T']):])
    ]
    if ids_to_show is not None:
        paths_to_show = [
        i for i in paths_to_show
        if any(c in ids_to_show for c in [network_table.iloc[i][label_dict['T'].index(j)] for j in years])
        ]
    if dynamic_paths_only:
        dynamic_entities = set(tsc.get_dynamic_entities()[0])
        paths_to_show = [i for i in paths_to_show if i in dynamic_entities]

    # create list of figures and iterate through features
    figures = []
    for f in range(F):
        fig = go.Figure()
        used_clusters = set()
        used_paths = {}
        # iterate through each path for the given feature
        for i in paths_to_show:
            x = years
            y = arr[:, i, f]
            
            # Create hover data
            path_ids = [network_table.iloc[i][label_dict['T'].index(j)] for j in years]
            for id in path_ids:
                if id not in used_paths:
                    used_paths[id] = [i]
                else:
                    used_paths[id].append(i)
            path_ids = [f'ID: {id}  Paths: {[path for path in used_paths[id]]}' for id in path_ids]

            # plot lines indicating values
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='lines+markers',
                    line=dict(color='black', dash='dot'),
                    showlegend=False
                )
            )
            # plot coloured dots indicating cluster
            label_i = labels[i] if labels.ndim == 1 else labels[i, 0]
            used_clusters.add(int(label_i))
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    marker=dict(color=colors[int(label_i)], size=6),
                    name=f"Path {i}",
                    hoverinfo='text',
                    hovertext=path_ids,
                    showlegend=False
                )
            )
        # plot cluster centres
        for j in range(K):
            if j in used_clusters:
                mode = 'lines+markers'
                fig.add_trace(
                    go.Scatter(
                        x=years,
                        y=cluster_centres[:, j, f],
                        mode=mode,
                        line=dict(color=colors[j]),
                        name=f"Cluster {cluster_labels[j]}"
                    )
                )
                
        # create layout and add figure to return list
        fig.update_layout(
            width=figsize[0],
            height=figsize[1],
            title=label_dict['F'][f],
            xaxis_title="Year",
            yaxis_title="Value",
            legend_title="Legend",
        )
        figures.append(fig)

    return figures


def plot_clusters_parallelcats(
    network_table: pd.DataFrame, 
    years: Optional[List[str]] = None,
    cluster_colours: Optional[dict[int, str]] = None,
    colour_index_year: Optional[str] = None,
    cluster_labels: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = (700, 500),
) -> go.Figure:
    """
    Creates an interactive parallel categories (parallel sets) plot to visualize how cluster 
    assignments evolve over time.

    Each column in the plot corresponds to a time point (e.g., a census year), and each
    path across the columns represents a "temporal path" of a tract or unit as it transitions
    across categories.

    Parameters:
        network_table (pd.DataFrame):
            A DataFrame containing the data. Expected to include one column per year
            with names in the format cluster_assignment_(year), e.g., 'cluster_assignment_2016'. (This
            will automatically be done if you have already run pc.cluster())

        years (List[str] | None):
            A list of strings representing the time points to include, such as ['2011', '2016', '2021'].
            Default is every year in the network table.

        cluster_colours (dict[int, str] | None):
            A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
            colour map will be used. If a cluster number is not part of the dict, plotly's default
            colour map will be used for that cluster.
        
        colour_index_year (str | None):
            The year that will be used to determine the colours of the parallel plot. For example, if you chose
            2011 as the colour index year, every cluster in the 2011 dimension would have a colour assigned to it,
            and then the paths into and out of these clusters would be shown in those colours. Default is the
            first year in the network table, and if an invalid input is given, the default will be used.
        
        cluster_labels (List[str] | None): 
            A custom list of cluster names. Default is Cluster 0, ..., Cluster n.

        figsize (Tuple[float, float] | None): 
            A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).

    Returns:
        plotly.graph_objects.Figure: 
            The interactive map
    """
    # create a list of valid columns across years
    columns = []
    if years is None:
        years = [col[-4:] for col in network_table.columns if "cluster_assignment" in col]
    for year in years:
        column = f'cluster_assignment_{year}'
        if column not in network_table.columns:
            raise ValueError(f"Expected column '{column}' not found in DataFrame.")
        else:
            columns.append(column)
    
    # create a list of dimensions (labelled vertical bars)
    dimensions = []
    for col in columns:
        values = network_table[col]
        if cluster_labels:
            value_map = {i: label for i, label in enumerate(cluster_labels)}
            values = values.map(value_map)
        dimensions.append(go.parcats.Dimension(
            values=values,
            categoryorder='category ascending',
            label=col[-4:]
        ))

    # set colour_index_year, dimension that will determine colours, and colour values list
    if colour_index_year is None or colour_index_year not in years:
        colour_index_year = years[0]
        
    color_col = f"cluster_assignment_{colour_index_year}"
    if color_col not in network_table.columns:
        raise ValueError(f"Coloring year '{colour_index_year}' not found in the table.")
    else:
        color_values = network_table[color_col]
    
    num_clusters = max([max(network_table[f'cluster_assignment_{year}']) for year in years]) + 1
    colors = []
    if cluster_colours:
        for i in range(num_clusters):
            if i in cluster_colours:
                colors.append(cluster_colours[i])
            else:
                colors.append(plotly.colors.qualitative.Plotly[i])
    else:
        colors = plotly.colors.qualitative.Plotly
        if num_clusters > len(colors):
            colors = list(islice(cycle(colors), num_clusters))
    
    colorscale = [[i / (len(cluster_labels) - 1), colors[i]] for i in range(len(cluster_labels))]

    # make the figure
    fig = go.Figure(data = [go.Parcats(dimensions=dimensions,
        line={'color': color_values,
        'colorscale': colorscale},
        hoveron='category', hoverinfo='count+probability',
        )])
    fig.update_layout(
        title="Parallel Categories Plot of Cluster Assignments Over Time",
        width=figsize[0],
        height=figsize[1],
    )

    return fig


def plot_clusters_area(
    network_table: pd.DataFrame,
    years: Optional[List[str]] = None,
    cluster_colours: Optional[dict[int, str]] = None,
    cluster_labels: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = (700, 500),
    stacked: Optional[bool] = True,
) -> go.Figure:
    """
    Creates an interactive area chart to visualize how cluster assignments evolve over time.

    Each column in the plot corresponds to a time point (e.g., a census year), and each
    path across the columns represents a "temporal path" of a tract or unit as it transitions
    across categories.

    Parameters:
        network_table (pd.DataFrame):
            A DataFrame containing the data. Expected to include one column per year
            with names in the format <feature_name>_<year>, e.g., 'cluster_assignment_2016'. (This
            will automatically be done if you have already run pc.cluster())

        years (List[str] | None):
            A list of strings representing the time points to include, such as ['2011', '2016', '2021'].
            Default is every year in the network table.

        cluster_colours (dict[int, str] | None):
            A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
            colour map will be used. If a cluster number is not part of the dict, plotly's default
            colour map will be used for that cluster.
        
        cluster_labels (List[str] | None): 
            A custom list of cluster names. Default is Cluster 0, ..., Cluster n.

        figsize (Tuple[float, float] | None): 
            A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).

        stacked (bool | None):
            Whether to show the area plot as a stacked plot, with all the areas on top of each other. If False,
            shows the area plot as a regular line graph. Default is True.

    Returns:
        plotly.graph_objects.Figure: 
            The interactive map
    """

    # auto-detect year columns if not provided
    if years is None:
        years = [col[-4:] for col in network_table.columns if "cluster_assignment" in col]

    # build count table: rows = years, columns = clusters
    cluster_counts = pd.DataFrame()
    for year in years:
        col = f"cluster_assignment_{year}"
        if col not in network_table.columns:
            raise ValueError(f"Expected column '{col}' not found in DataFrame.")
        counts = network_table[col].value_counts().sort_index()
        cluster_counts[year] = counts
    cluster_counts = cluster_counts.fillna(0).astype(int).T

    num_clusters = max([max(network_table[f'cluster_assignment_{year}']) for year in years]) + 1
    colors = []
    if cluster_colours:
        for i in range(num_clusters):
            if i in cluster_colours:
                colors.append(cluster_colours[i])
            else:
                colors.append(plotly.colors.qualitative.Plotly[i])
    else:
        colors = plotly.colors.qualitative.Plotly
        if num_clusters > len(colors):
            colors = list(islice(cycle(colors), num_clusters))

    # create traces
    fig = go.Figure()
    x_vals = cluster_counts.index.tolist()
    cumulative = pd.DataFrame(0, index=cluster_counts.index, columns=cluster_counts.columns)

    for i, cluster in enumerate(cluster_counts.columns):
        y_vals = cluster_counts[cluster]

        if stacked:
            y_base = cumulative.sum(axis=1)
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_base + y_vals,
                mode='lines',
                line=dict(color=colors[cluster]),
                name=f'Cluster {cluster}' if cluster_labels is None else cluster_labels[i],
                stackgroup='one',
            ))
            cumulative[cluster] = y_vals
        else:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines+markers',
                line=dict(color=colors[cluster]),
                name=f'Cluster {cluster}' if cluster_labels is None else cluster_labels[i]
            ))

    # final layout
    fig.update_layout(
        title="Area Plot of Cluster Assignments Over Time",
        xaxis_title="Year",
        yaxis_title="Number of Geographical Units",
        width=figsize[0],
        height=figsize[1],
        legend_title="Clusters",
    )

    return fig


def plot_clusters_map(
    year: str,
    id: Optional[str] = 'geouid',
    cluster_colours: Optional[dict[int, str]] = None,
    label_dict: Optional[dict[str, Any]] = None,
    cluster_labels: Optional[List[str]] = None,
    geofile_path: Optional[str] = None,
    network_table: Optional[pd.DataFrame] = None,
    gdf: Optional[gpd.GeoDataFrame] = None,
    figsize: Optional[Tuple[float, float]] = (700, 500),
) -> px.choropleth:
    """
    Plots cluster assignments in their associated geographical regions for a specific year using a GeoDataFrame.
    To properly load the geographical data, the user must provide AT LEAST ONE of the following:
    1. geofile_path AND network_table
    2. gdf

    Parameters:
        year (str):
            Year to visualize (used in column name)

        id (str):
            Unique identifier for the geographical region and year (used in hover data). Default is 'geouid'.

        cluster_colours (dict[int, str] | None):
            A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
            colour map will be used. If a cluster number is not part of the dict, plotly's default
            colour map will be used for that cluster.

        label_dict(dict[str, Any] | None):
            The label dictionary from pc.clustering_prep() that you used in pc.cluster() or a custom 
            label dictionary. Used to determine what data will be shown when you hover over each geographical
            region. If None, only the index (path number) will be shown.

        cluster_labels (List[str] | None): 
            A custom list of cluster names. Default is Cluster 0, ..., Cluster n.

        geofile_path (str | None):
            Path to geographical data file if gdf is not passed

        network_table (pd.DataFrame | None):
            Network table to be merged with GeoJSON

        gdf (GeoDataFrame | None):
            Pre-joined GeoDataFrame (recommended for advanced users)

        figsize (Tuple[float, float] | None):
            A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).

    Returns:
        plotly.express.choropleth: 
            The interactive choropleth map
    """
    # Load and merge geodata if not provided
    if gdf is None:
        if geofile_path is None or network_table is None:
            raise ValueError("You must provide either `gdf` or both `geofile_path` and `network_table`.")
        gdf = join_geometries(geofile_path, network_table, year)

    # Ensure column is valid
    cluster_col = f"cluster_assignment_{year}"
    if cluster_col not in gdf.columns:
        raise ValueError(f"Column '{cluster_col}' not found in the GeoDataFrame.")

    # Ensure data is categorical for consistent colouring
    if cluster_labels:
        cluster_labels = {i: cluster_labels[i] for i in range(len(cluster_labels))}
        gdf["cluster_name"] = gdf[cluster_col].map(cluster_labels)
        color_col = "cluster_name"
    else:
        gdf[cluster_col] = gdf[cluster_col].astype(str)
        color_col = cluster_col

    # create hover data
    hover_data = {}
    hover_data[f'{id}_{year}'] = True
    if cluster_labels:
        hover_data['cluster_name'] = False
    if label_dict:
        for feature in label_dict['F']:
            hover_data[f'{feature}_{year}'] = True

    # create colours
    if network_table is not None:
        years = [col[-4:] for col in network_table.columns if "cluster_assignment" in col]
        num_clusters = max([max(network_table[f'cluster_assignment_{year}']) for year in years]) + 1
    else:
        years = [col[-4:] for col in gdf.columns if "cluster_assignment" in col]
        num_clusters = max([max(gdf[f'cluster_assignment_{year}']) for year in years]) + 1
    if cluster_colours:
        for i in range(num_clusters):
            if i not in cluster_colours:
                cluster_colours[i] = plotly.colors.qualitative.Plotly[i]
        if cluster_labels:
            # map cluster_colours to labels
            cluster_colours = {
                cluster_labels[i]: cluster_colours[i] for i in cluster_colours if i < len(cluster_labels)
            }
        else:
            # ensure string keys
            cluster_colours = {str(k): v for k, v in cluster_colours.items()}
    

    # Convert geometry to GeoJSON
    gdf = gdf.to_crs(epsg=4326)  # Ensure proper projection for web mapping
    geojson = gdf.__geo_interface__

    fig = px.choropleth(
        gdf,
        geojson=geojson,
        locations=gdf.index,  # use index to map geometries
        color=color_col,
        hover_name=color_col,
        hover_data=hover_data,
        color_discrete_map=cluster_colours if cluster_colours else None,
        title=f"Cluster Assignments in {year}"
    )

    fig.update_geos( # if we don't do this it will show the whole world
        fitbounds="locations",
        visible=False
    )

    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        legend_title_text="Cluster"
    )

    return fig


def plot_line_means(
        cluster_feature_means: pd.DataFrame,
        years: List[int],
        selected_features: List[str],
        cluster_colours: Optional[dict[int, str]] = None,
        title: str = "Mean Variables by Cluster Over Time",
        figsize: Tuple[float, float] = (1200, 600),
) -> go.Figure:
    """
    Creates an interactive line‐chart with one subplot per feature, showing how
    cluster‐mean values evolve over the selected years.

    Each subplot corresponds to a feature (variable), and each line within it
    tracks a single cluster across time, using a consistent colour per cluster.

    Parameters:
        cluster_feature_means (pd.DataFrame):
            The resulted dataframe from cluster_means_by_year function

        years (List[int]):
            Time points to include (e.g. ['2011','2016','2021']). If None,
            auto-detected from the second level of the DataFrame’s columns.
        selected_features (List[str]):
            Which features (base_cols) to plot

        title (str):
            Figure title.

        figsize (Tuple[float, float]):
            Width and height of the overall figure in pixels.


    Returns:
        plotly.graph_objects.Figure:
            The composed line‐chart with subplots.
    """

    # 1) Pick a distinct color palette & map clusters → colors
    clusters = list(cluster_feature_means.index)

    num_clusters = len(years)
    colors = []
    if cluster_colours:
        for i in range(num_clusters):
            if i in cluster_colours:
                colors.append(cluster_colours[i])
            else:
                colors.append(plotly.colors.qualitative.Plotly[i])
    else:
        colors = plotly.colors.qualitative.Plotly
        if num_clusters > len(colors):
            colors = list(islice(cycle(colors), num_clusters))

    # 2) Make subplots
    fig = make_subplots(
        rows=1,
        cols=len(selected_features),
        shared_yaxes=False,
        subplot_titles=[v.replace('_', ' ').title() for v in selected_features],
    )

    # 3) Add one trace per cluster per subplot, forcing line+marker colors
    for col_idx, var in enumerate(selected_features, start=1):
        df_var = cluster_feature_means[var]
        x_vals = df_var.columns.astype(int)

        for cluster in clusters:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=df_var.loc[cluster],
                    mode='lines+markers',
                    name=f'Cluster {cluster}',
                    line=dict(color=colors[cluster]),
                    marker=dict(color=colors[cluster]),
                    legendgroup=str(cluster),
                    showlegend=(col_idx == 1)
                ),
                row=1,
                col=col_idx
            )

        # update axes
        fig.update_xaxes(
            title_text=f"Mean {var.replace('_', ' ')}",
            tickmode="array",
            tickvals=years,
            ticktext=[str(y) for y in years],
            row=1,
            col=col_idx
        )

    # 4) Final layout
    fig.update_layout(
        title=title,
        width=figsize[0],
        height=figsize[1],
        legend_title_text="Cluster",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(t=80, b=40, l=60, r=20)
    )
    return fig


def plot_bar_means(
        cluster_feature_means: pd.DataFrame,
        years: List[int],
        selected_features: List[str],
        cluster_colours: Optional[dict[int, str]] = None,
        figsize: Tuple[float, float] = (900, 600),
) -> go.Figure:
    """
    Create grouped bar-chart subplots of cluster means for each year.

    For each year in `years`, plots the mean value of each feature in
    `selected_features` for every cluster. Subplots are arranged in a
    grid with two columns; colors are assigned per cluster via the
    provided `cluster_colours` mapping or default Plotly palette.

    Parameters:

        cluster_feature_means : pd.DataFrame
            The resulted DataFrame from the cluster_means_by_year function

        years : List[int]
            Years to include as separate subplots

        selected_features : List[str]
            List of feature names to plot on the x-axis

        cluster_colours : dict[int, str], optional
            Mapping from cluster label (row index) to a Plotly color string
            If omitted, uses `plotly.colors.qualitative.Plotly` and cycles if needed

        figsize : Tuple[int, float]
            Width and height of the full figure in pixels


    Returns:
        go.Figure
        A Plotly Figure with one subplot per year, each showing grouped
        bars for clusters across the selected features

    """
    # 1) determine grid
    rows = math.ceil(len(years) / 2)

    # 2) prepare color map for clusters
    clusters = list(cluster_feature_means.index)

    num_clusters = len(years)
    colors = []
    if cluster_colours:
        for i in range(num_clusters):
            if i in cluster_colours:
                colors.append(cluster_colours[i])
            else:
                colors.append(plotly.colors.qualitative.Plotly[i])
    else:
        colors = plotly.colors.qualitative.Plotly
        if num_clusters > len(colors):
            colors = list(islice(cycle(colors), num_clusters))

    # 3) build subplot figure
    subplot_titles = [f"Means by Cluster in {year}" for year in years]
    fig = make_subplots(
        rows=rows,
        cols=2,
        subplot_titles=subplot_titles,
        shared_yaxes=False,

    )

    # 4) grab the second level (years) of the MultiIndex
    lvl1 = cluster_feature_means.columns.get_level_values(1)

    # 5) for each year, slice and add a bar trace per cluster
    for idx, year in enumerate(years):
        # compute row/col
        r = idx // 2 + 1
        c = idx % 2 + 1

        # build a boolean mask (int or str)
        if year in lvl1:
            mask = lvl1 == year
        else:
            str_lvl1 = [str(y) for y in lvl1]
            if str(year) in str_lvl1:
                mask = [s == str(year) for s in str_lvl1]
            else:
                raise KeyError(f"Year {year!r} not found in columns: {sorted(set(lvl1))}")

        # slice out only this year's columns, rename them to selected_features
        df_year = cluster_feature_means.loc[:, mask].copy()
        df_year.columns = cluster_feature_means.columns.get_level_values(0)[mask]
        df_year = df_year[selected_features]

        # plot each cluster as a bar trace
        for cluster in clusters:
            fig.add_trace(
                go.Bar(
                    x=[v.replace('_', ' ') for v in selected_features],
                    y=df_year.loc[cluster],
                    name=f"Cluster {cluster}",
                    marker_color=colors[cluster],
                    showlegend=(idx == 0)  # legend only on first subplot
                ),
                row=r,
                col=c
            )

    # 6) final layout tweaks
    fig.update_layout(
        title="Cluster Means by Variable and Year",
        width=figsize[0],
        height=figsize[1],
        bargap=0.2,
        legend_title_text="Cluster",
        template="plotly_white"
    )
    # tighten subplot margins
    fig.update_layout(margin=dict(t=80, b=50, l=50, r=20))
    return fig


def radar_chart_multiple_years(
        cluster_feature_means: pd.DataFrame,
        years: List[int],
        selected_cluster: int,
        selected_features: list,
        figsize: Tuple[int, int] = (900, 600)
) -> go.Figure:
    """
    Create a radar (polar) chart of selected variables for a given cluster across years

    Parameters:
    cluster_feature_means : pd.DataFrame
        The resulted dataframe from cluster_means_by_year function

    years : List[int]
        Which years to plot

    selected_cluster : int
        Which clusters to plot

    selected_features : List[str]
        Which features to include, should be at lease 3

    figsize : (width, height),
        Size of the figure in pixels
    """

    # Get the year level of the MultiIndex
    lvl1 = cluster_feature_means.columns.get_level_values(1)
    fig = go.Figure()

    for idx, year in enumerate(years):
        # build mask robustly
        if year in lvl1:
            mask = lvl1 == year
        else:
            str_lvl1 = [str(y) for y in lvl1]
            if str(year) in str_lvl1:
                mask = [s == str(year) for s in str_lvl1]
            else:
                raise KeyError(f"Year {year!r} not in columns: {sorted(set(lvl1))}")

        # Slice out just this year's columns and rename to variable names
        df_year = cluster_feature_means.loc[:, mask].copy()
        df_year.columns = cluster_feature_means.columns.get_level_values(0)[mask]

        # Reorder and pick only the requested variables
        df_year = df_year[selected_features]

        # Extract the row for the given cluster
        result = df_year.iloc[selected_cluster]

        # Add as a polar trace
        fig.add_trace(go.Scatterpolar(
            r=result.values,
            theta=result.index,
            fill='toself',
            name=str(year),
        ))

        fig.update_layout(
            title=dict(
                text=f"Cluster {selected_cluster} Yearly Profile",
                x=0.5, xanchor="center"
            ),
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            width=figsize[0],
            height=figsize[1],
            template="plotly_white",
        )

    return fig


def radar_chart_multiple_clusters(
        cluster_feature_means: pd.DataFrame,
        selected_year: int,
        selected_features: list,
        figsize: Tuple[int, int] = (900, 600),
) -> go.Figure:
    """
    Draw a radar chart of the given variables for every cluster in `cluster_feature_means`,
    all on the same figure, for the specified year

    Parameters:
    -----------
    cluster_feature_means : pd.DataFrame
        The resulted dataframe from cluster_means_by_year function

    selected_year : int
        Selected year to show its features

    selected_features : list of str
        Which variables to plot

    figsize : (width, height), default=(900,600)
        Size of the figure in pixels

    Returns:

    fig : go.Figure
        The Plotly figure containing one polar trace per cluster.
    """
    # Extract the year-level values
    lvl1 = cluster_feature_means.columns.get_level_values(1)

    # Build boolean mask for matching year
    if selected_year in lvl1:
        mask = lvl1 == selected_year
    else:
        str_lvl1 = list(map(str, lvl1))
        if str(selected_year) in str_lvl1:
            mask = [s == str(selected_year) for s in str_lvl1]
        else:
            raise KeyError(f"Year {selected_year!r} not found in columns: {sorted(set(lvl1))}")

    # Slice out this year's columns and rename to variable names
    df_year = cluster_feature_means.loc[:, mask].copy()
    df_year.columns = cluster_feature_means.columns.get_level_values(0)[mask]

    # Create the figure
    fig = go.Figure()

    # Add one trace per cluster
    for cluster_label in df_year.index:
        vals = df_year.loc[cluster_label, selected_features]
        fig.add_trace(go.Scatterpolar(
            r=vals.values,
            theta=selected_features,
            fill='toself',
            name=f'Cluster {cluster_label}'
        ))

    # Final layout tweaks
    fig.update_layout(
        title=dict(
            text=f"Cluster Profiles for {selected_year}",
            x=0.5, xanchor="center", yanchor="top"
        ),
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        width=figsize[0],
        height=figsize[1],
        template="plotly_white",
    )

    return fig

# Helpers

def ct_containment(preprocessed_dfs, years, id='GeoUID', threshold=0.05):
    '''
    Returns a GeoDataFrame with census tracts which are contained
    within a census tract from the following census
    '''
    num_years = len(years)
    contained_tracts = []

    for i in range(num_years - 1):
        df1 = preprocessed_dfs[i][[id, f'area_{years[i]}', 'geometry']].copy()
        df2 = preprocessed_dfs[i + 1][[id, f'area_{years[i + 1]}', 'geometry']].copy()
        year1, year2 = years[i], years[i + 1]

        # Rename and set active geometry for join
        df1 = df1.rename(columns={id: f'{id}_1'})
        df1 = gpd.GeoDataFrame(df1, geometry='geometry', crs=preprocessed_dfs[i].crs)

        df2 = df2.rename(columns={id: f'{id}_2', 'geometry': 'geometry_2'})
        df2 = gpd.GeoDataFrame(df2, geometry='geometry_2', crs=preprocessed_dfs[i + 1].crs)

        # Spatial join: only keeps left geometry
        joined = gpd.sjoin(df1, df2, predicate='intersects', how='inner')

        # Extract correct original area columns manually
        area1_col = f'area_{year1}'
        area2_col = f'area_{year2}'

        # Merge in right-side area and geometry
        joined = joined.drop(columns=[area2_col], errors='ignore')  # remove any stub if exists
        joined = joined.merge(
            df2[[f'{id}_2', area2_col, 'geometry_2']],
            on=f'{id}_2',
            how='left'
        )

        # Compute true intersection geometry
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=UserWarning)
            joined['geometry'] = joined.apply(
                lambda row: row['geometry'].intersection(row['geometry_2']),
                axis=1
            )

        # Set geometry column again for safety
        joined = gpd.GeoDataFrame(joined, geometry='geometry', crs=preprocessed_dfs[i].crs)

        # Compute intersection area
        joined['area_intersection'] = joined.geometry.area

        # Compute true % overlap
        pct_col = f'pct_{year2}_of_{year1}'
        joined[pct_col] = joined['area_intersection'] / joined[[area1_col, area2_col]].min(axis=1)

        # Filter by threshold
        joined = joined[joined[pct_col] >= threshold]

        # Return only necessary columns
        kept = joined[[f'{id}_1', f'{id}_2', f'area_{year1}', f'area_{year2}', 'geometry', 'area_intersection', pct_col]]
        contained_tracts.append(kept)

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


def filter_columns(
    network_table: pd.DataFrame, 
    years: List[str], 
    cols: Optional[list[str]]=[]
    ) -> Tuple[List[str], List[str]]:
    '''
    Checks that the list of columns with data to be clustered is valid in the following ways:
    - Makes sure all the data in the columns are numerical or nan
    - Makes sure there is a version of each column for every year

    Parameters:
        network_table (pd.DataFrame): 
            The result of pc.create_network_table().
        
        years (List[str]): 
            A list of years considered for clustering.
          
        cols (list[str] | None): A list of the names of network table columns that should be considered in
            the clustering algorithm. If none, every numerical feature will be considered. Leaving it none is
            not recommended as many numerical features, such as network level, have little bearing on the data.
    
    Returns:
        (Tuple[List[str], List[str]]):
            a tuple of the final filtered list of columns and the column labels that will
            be used for the label dictionary.
    '''
    # Only add features that are numerical or nan. the user should have selected accordingly
    # but this is a sanity check
    col_list = []

    for col in cols:
        if col in network_table.columns.to_list():
            non_numerical_val_in_col = False
            for entry in network_table[col]:
                if isinstance(entry, str) and '_' in entry: # make sure underscores don't get converted to numbers
                    non_numerical_val_in_col = True
                    break
                try:
                    int(entry)  # see if it is either an int or an int masquerading as a string
                except ValueError:
                    try:
                        float(entry)  # see if it is either a float or a float masquerading as a string
                    except ValueError:
                        if entry != 'NaN' and entry != 'nan': # see if it is nan
                            non_numerical_val_in_col = True
                            break
            if not non_numerical_val_in_col:
                col_list.append(col)

    # Only add features for which there are variables in every year. Otherwise the shape of
    # the 3D array used for tscluster will not make sense.
    # note: we can improve on this with some version of the ppandas library (https://link.springer.com/article/10.1007/s10618-024-01054-7)
    cols_in_every_year = []
    features_list = [] # for the label dictionary
    add_to_list = True
    col_names_without_year = list(dict.fromkeys([col[:-4] for col in col_list])) # remove duplicates while preserving original order
    for col in col_names_without_year:
        add_to_list = True
        for year in years:
            if f"{col}{year}" not in col_list:
                add_to_list = False
                break
        for year in years:
            if add_to_list:
                if col[:-1] not in features_list:
                    features_list.append(col[:-1])
                cols_in_every_year.append(f"{col}{year}")

    return (cols_in_every_year, features_list)


def join_geometries(
    geofile_path: str,
    network_table: pd.DataFrame,
    year: str,
    geofile_id_col: Optional[str] = "GeoUID",
    network_table_id_col: Optional[str] = "geouid"
) -> gpd.GeoDataFrame:
    """
    Joins spatial data from a geographical data file with attribute data from a network table
    using a shared geographic identifier.

    This function is designed for researchers who work with pre-processed network tables
    (containing cluster assignments, IDs, etc.) and separately downloaded spatial files
    (like Canadian census tract GeoJSONs). It's recommended to run this function yourself
    before plotting cluster assignments with cluster_map_plot if you are using a column id different
    from the default 'geouid'.

    Parameters:
        geofile_path (str):
            File path to the geographical data file for the specified year. Can be anything readable by geopandas.

        network_table (pd.DataFrame):
            DataFrame containing attribute and cluster assignment data, including unique
            geographic identifiers for each region.

        year (str):
            The census year to match ID and cluster columns (e.g., '2016').

        geofile_id_col (str | None):
            Column name in the geographical data file that contains the geographic identifier
            (default: 'GeoUID').

        network_table_id_col (str | None):
            Prefix of the column name in the network_table used for geographic ID matching.
            The function expects a column like 'geouid_2016' if year='2016'.

    Returns:
        gpd.GeoDataFrame:
            A merged GeoDataFrame containing geometry from the GeoJSON file and attribute
            data (e.g., cluster assignments) from the network table. Only valid, non-empty
            geometries are retained.
    """

    # Validate input column name
    geoid_col = f"{network_table_id_col}_{year}"
    if geoid_col not in network_table.columns:
        raise ValueError(f"Expected column '{geoid_col}' not found in network_table.")

    # Read the GeoJSON file into a GeoDataFrame
    gdf = gpd.read_file(geofile_path)

    # Prepare a clean copy of the network table and standardize the ID format
    network_table_copy = network_table.copy(deep=True)
    network_table_copy[geoid_col] = network_table_copy[geoid_col].astype(str).str.replace(r'^\d{4}_', '', regex=True)

    # Merge the GeoDataFrame with the network table using the geographic ID
    merged_gdf = gdf.merge(network_table_copy, left_on=geofile_id_col, right_on=geoid_col)

    # Remove empty or invalid geometries
    merged_gdf = merged_gdf[~merged_gdf.is_empty & merged_gdf.geometry.notnull()]

    return merged_gdf

def cluster_means_by_year(
    network_table: pd.DataFrame,
    years: list,
    base_cols: list,
    cluster_prefix: str = 'cluster_assignment'
) -> pd.DataFrame:
    """
    Compute mean values of base_cols per cluster for each year,
    and concatenate into a MultiIndex DataFrame (variable, year).
    """
    dfs = {}
    for year in years:
        col_names = [f'{col}_{year}' for col in base_cols]
        df_year = (
            network_table
            .groupby(f'{cluster_prefix}_{year}')[col_names]
            .mean()
            .rename(columns={f'{col}_{year}': col for col in base_cols})
        )
        dfs[year] = df_year

    cluster_feature_means = pd.concat(dfs, axis=1)
    cluster_feature_means = cluster_feature_means.swaplevel(0, 1, axis=1).sort_index(axis=1)
    return cluster_feature_means