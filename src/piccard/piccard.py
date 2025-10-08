import numpy as np
import geopandas as gpd
import pandas as pd
import networkx as nx
from typing import Any, List, Tuple, Optional, Dict
from pyproj import CRS

import warnings
warnings.filterwarnings('ignore')

# Optional dependency flags
from .core.probabilistic_reasoning import PPANDAS_AVAILABLE
from .core.clustering import TSCLUSTER_AVAILABLE

# Type annotation imports
from .core.network import NetworkTable
from .core.clustering import ClusteredNetworkTable
import plotly.graph_objects as go
import plotly.express as px

# Explicit public API imports for piccard
from .core.network import (
    core_create_network,
    core_create_network_table,
    core_preprocessing
)
from .core.probabilistic_reasoning import (
    core_prob_reasoning_networks,
    core_prob_reasoning_years
)
from .core.clustering import (
    core_clustering_prep,
    core_cluster
)
from .visualization.network_visual import (
    visual_plot_subnetwork,
    visual_plot_num_areas
)
from .visualization.cluster_plots import (
    visual_plot_clusters_scatter,
    visual_plot_clusters_parallelcats,
    visual_plot_clusters_area,
    visual_plot_clusters_map,
    visual_plot_line_means,
    visual_plot_bar_means,
    visual_radar_chart_multiple_years,
    visual_radar_chart_multiple_clusters
)
from .linking.variable_linker import VariableLinker

# Module 1: Network Creation

def preprocessing(
    data: gpd.GeoDataFrame, 
    year: str, 
    id: str,
    crs: Optional[CRS] = "EPSG:3347",
    verbose: Optional[bool] = True
) -> gpd.GeoDataFrame:
    '''
    Not necessary for network table creation, but you may optionally run this function yourself, for example
    if you want details of the dataframe cleaning but not the network creation, or if you want to try out
    different CRSs.
    Returns a cleaned geopandas df of the input data. Uses parallel processing for very large (>100,000 rows) datasets.
    Also adds a column for each year with calculated areas of each census tract in that year.
    Note: Input data is assumed to have been passed through gpd.read_file() beforehand.

    Parameters:
        data (GeoDataFrame):
            The census data to be analyzed with piccard.

        year (str):
            The year that the census data was collected.

        id (str):
            The name of the unique identifier that will be used to distinguish geographical areas.

        crs (CRS | None):
            A pythonic Coordinate Reference System manager that will be used to compute areas. Default is
            EPSG:3347, a consistent, equal-area CRS based on square metres. Can be many formats; see 
            https://pyproj4.github.io/pyproj/stable/api/crs/crs.html for more information.

        verbose (bool | None):
            Whether to issue print statements about the progress of network creation. Default is true.
    
    Returns:
        GeoDataFrame: the cleaned data
    '''
    return core_preprocessing(data=data, year=year, id=id, crs=crs, verbose=verbose) 


def create_network(
    census_dfs: List[gpd.GeoDataFrame], 
    years: List[str], 
    id: str, 
    crs: Optional[CRS] = "EPSG:3347",
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

      crs (CRS | None):
            A pythonic Coordinate Reference System manager that will be used to compute areas. Default is
            EPSG:3347, a consistent, equal-area CRS based on square metres. Can be many formats; see 
            https://pyproj4.github.io/pyproj/stable/api/crs/crs.html for more information.

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
  return core_create_network(census_dfs=census_dfs, years=years, id=id, crs=crs, threshold=threshold, verbose=verbose)


def create_network_table(
    census_dfs: List[gpd.GeoDataFrame], 
    years: List[str], 
    id: str, 
    crs: Optional[CRS] = "EPSG:3347",
    threshold: Optional[float] = 0.05,
    verbose: Optional[bool] = True
) -> NetworkTable:
  '''
  Creates a NetworkTable showing the network representation of the census data in census_dfs. 
  Each feature present in the data is a column, and each possible path through the network is a row.

  Parameters:
      census_dfs (List[gpd.GeoDataFrame]):
          A list of GeoDataFrames containing the census data to be turned into a network.

      years (List[str]):
          A list of years present in census_dfs over which the network representation will be created.
          Data from years not present in years will be ignored.
      
      id (str):
          The name of the unique identifier that will be used to distinguish geographical areas.

      crs (CRS | None):
            A pythonic Coordinate Reference System manager that will be used to compute areas. Default is
            EPSG:3347, a consistent, equal-area CRS based on square metres. Can be many formats; see 
            https://pyproj4.github.io/pyproj/stable/api/crs/crs.html for more information.

      threshold (float | None):
          The percentage of overlap (divided by 100)
          that geographic areas must meet or exceed in order to have a connection.
          Default is 0.05, or 5 percent.    

      verbose (bool | None):
          Whether to issue print statements about the progress of network creation. Default is true.

  Returns:
      NetworkTable: the table.
  '''
  return core_create_network_table(
      census_dfs=census_dfs, years=years, id=id, crs=crs, threshold=threshold, verbose=verbose)


# Module 2: Clustering

def clustering_prep(
    network_table: NetworkTable,
    cols: Optional[list[str]]=[]
) -> tuple[np.ndarray[np.float64], dict[str, Any], NetworkTable]:
    '''
    Converts a piccard network table into a 3d numpy array of all possible paths and their corresponding
    features. This will be used for clustering with tscluster.
    The user can (optionally) input a list of columns that they want to be considered in the clustering algorithm, 
    and the function will check that these columns are valid.

    Note that you must run pc.create_network_table() before this function.

    Parameters:
        network_table (NetworkTable): 
            The result of pc.create_network_table().
 
        cols (list[str] | None): A list of the names of network table columns that should be considered in
            the clustering algorithm. If none, every numerical feature will be considered. Leaving it none is
            not recommended as many numerical features, such as network level, have little bearing on the data.

    Returns:
        (tuple[np.ndarray[np.float64], dict[str, Any]], NetworkTable):
            a tuple of a 3d numpy array, a corresponding dictionary of labels showing
            the shape of the array, and the network table modified so it doesn't include any of the NaN rows.
    '''
    if TSCLUSTER_AVAILABLE:
        return core_clustering_prep(network_table=network_table, cols=cols)
    else:
        raise ModuleNotFoundError("Sorry, you need to install `tscluster` to use this module of `piccard`. See the documentation for installation instructions.")


def cluster(
    network_table: NetworkTable, 
    G: nx.Graph, 
    num_clusters: int, 
    algo: Optional[str]='greedy', 
    scheme: Optional[str]='z1c1', 
    arr: Optional[np.ndarray[np.float64]]=None, 
    label_dict: Optional[dict[str, Any]]=None
) -> ClusteredNetworkTable:
    '''
    Runs one of tscluster's clustering algorithms (default is fully dynamic clustering or 'z1c1')
    and adds the resulting cluster assignments to the network table and nodes as an additional feature.
    Information about the different clustering algorithms is available here: https://tscluster.readthedocs.io/en/latest/introduction.html
    We recommend either Sequential Label Analysis ('z1c0') or the default 'z1c1'.

    Users can choose to only input the network table, in which case clustering_prep will be run for them with the default columns,
    or they can choose to run clustering_prep on their own and then have the option to apply one or both of the
    normalization methods available in tscluster.preprocessing.utils.

    Parameters:
        network_table (NetworkTable): 
            The result of pc.create_network_table().

        G (nx.Graph): 
            The result of pc.create_network().

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
        ClusteredNetworkTable: 
            The clustered network table.
    '''
    if TSCLUSTER_AVAILABLE:
        return core_cluster(
            network_table=network_table, G=G, num_clusters=num_clusters, algo=algo, scheme=scheme, arr=arr, label_dict=label_dict)
    else:
        raise ModuleNotFoundError("Sorry, you need to install `tscluster` to use this module of `piccard`. See the documentation for installation instructions.")

# Module 3: Visualization & Analysis

def plot_subnetwork(
    network_table: NetworkTable, 
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
        network_table (NetworkTable):
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
    return visual_plot_subnetwork(
        network_table=network_table, G=G, years=years, paths_to_show=paths_to_show, ids_to_show=ids_to_show, num_to_sample=num_to_sample)


def plot_num_areas(
    network_table: NetworkTable, 
    years: Optional[List[str]] = None,
) -> go.Figure:
    '''
    Plots the number of geographical areas across a subset of census years in the data.

    Parameters:
        network_table (NetworkTable):
            The result of pc.create_network_table().

        years (List[str] | None):
            A list of years to show in the subnetwork. Default is all census years present in the data.

    Returns:
        go.Figure:
            The plot of the number of geographical areas.
    '''
    return visual_plot_num_areas(network_table=network_table, years=years)


def plot_clusters_scatter(
    network_table: ClusteredNetworkTable,
    label_dict: Optional[dict[str, Any]] = None,
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
        network_table (ClusteredNetworkTable):
            The clustered network table.

        label_dict (dict[str, Any] | None):
            A custom label dictionary.

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
        List[go.Figure]:
            a list of plotly.graph_objects.Figure (you cannot show the whole list; rather, iterate through 
            the list and show each figure)
    '''
    if TSCLUSTER_AVAILABLE:
        return visual_plot_clusters_scatter(
            network_table=network_table, label_dict=label_dict, years=years,
            cluster_colours=cluster_colours, dynamic_paths_only=dynamic_paths_only,
            paths_to_show=paths_to_show, ids_to_show=ids_to_show, clusters_to_show=clusters_to_show,
            clusters_to_exclude=clusters_to_exclude, figsize=figsize, cluster_labels=cluster_labels
        )
    else:
        raise ModuleNotFoundError("Sorry, you need to install `tscluster` to use this module of `piccard`. See the documentation for installation instructions.")


def plot_clusters_parallelcats(
    network_table: ClusteredNetworkTable, 
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
        network_table (ClusteredNetworkTable):
            A ClusteredNetworkTable containing the data.

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
    if TSCLUSTER_AVAILABLE:
        return visual_plot_clusters_parallelcats(
            network_table=network_table, years=years, cluster_colours=cluster_colours,
            colour_index_year=colour_index_year, cluster_labels=cluster_labels, figsize=figsize
        )
    else:
        raise ModuleNotFoundError("Sorry, you need to install `tscluster` to use this module of `piccard`. See the documentation for installation instructions.")


def plot_clusters_area(
    network_table: ClusteredNetworkTable,
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
        network_table (ClusteredNetworkTable):
            A ClusteredNetworkTable containing the data.

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
    if TSCLUSTER_AVAILABLE:
        return visual_plot_clusters_area(
            network_table=network_table, years=years, cluster_colours=cluster_colours,
            cluster_labels=cluster_labels, figsize=figsize, stacked=stacked
        )
    else:
        raise ModuleNotFoundError("Sorry, you need to install `tscluster` to use this module of `piccard`. See the documentation for installation instructions.")


def plot_clusters_map(
    geofile_path: str,
    network_table: ClusteredNetworkTable,
    year: str,
    cluster_colours: Optional[dict[int, str]] = None,
    label_dict: Optional[dict[str, Any]] = None,
    cluster_labels: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = (700, 500),
) -> px.choropleth:
    """
    Plots cluster assignments in their associated geographical regions for a specific year using a GeoDataFrame.

    Parameters:
        geofile_path (str):
            Path to geographical data file

        network_table (ClusteredNetworkTable):
            Network table to be merged with GeoJSON

        year (str):
            Year to visualize (used in column name)

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

        figsize (Tuple[float, float] | None):
            A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).

    Returns:
        plotly.express.choropleth: 
            The interactive choropleth map
    """
    if TSCLUSTER_AVAILABLE:
        return visual_plot_clusters_map(
            year=year,
            cluster_colours=cluster_colours,
            label_dict=label_dict,
            cluster_labels=cluster_labels,
            geofile_path=geofile_path,
            network_table=network_table,
            figsize=figsize
        )
    else:
        raise ModuleNotFoundError("Sorry, you need to install `tscluster` to use this module of `piccard`. See the documentation for installation instructions.")


def plot_line_means(
        network_table: ClusteredNetworkTable,
        selected_features: List[str],
        years: Optional[List[int]] = None,
        varnames: Optional[List[str]] = None,
        cluster_colours: Optional[dict[int, str]] = None,
        cluster_labels: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (700, 500),
) -> go.Figure:
    """
    Creates an interactive line chart with one subplot per feature, showing how
    cluster mean values evolve over the selected years.

    Each subplot corresponds to a feature (variable), and each line within it
    tracks a single cluster across time, using a consistent colour per cluster.

    Parameters:
        network_table (ClusteredNetworkTable):
            A ClusteredNetworkTable containing the data.

        years (List[int] | None):
            Which years to plot. Default is every year in the network table.
            
        selected_features (List[str]):
            Which features (column names present in clustering) to plot
        
        varnames (List[str] | None):
            The custom variable names to plot

        cluster_colours (dict[int, str] | None):
            A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
            colour map will be used. If a cluster number is not part of the dict, plotly's default
            colour map will be used for that cluster.

        cluster_labels (List[str] | None): 
            A custom list of cluster names. Default is Cluster 0, ..., Cluster n.

        figsize (Tuple[float, float] | None):
            Width and height of the overall figure in pixels. Default is (700, 500).


    Returns:
        plotly.graph_objects.Figure:
            The composed line chart with subplots.
    """
    if TSCLUSTER_AVAILABLE:
        return visual_plot_line_means(
            network_table=network_table,
            selected_features=selected_features,
            years=years,
            varnames=varnames,
            cluster_colours=cluster_colours,
            cluster_labels=cluster_labels,
            figsize=figsize
        )
    else:
        raise ModuleNotFoundError("Sorry, you need to install `tscluster` to use this module of `piccard`. See the documentation for installation instructions.")


def plot_bar_means(
        network_table: ClusteredNetworkTable,
        selected_features: List[str],
        years: Optional[List[int]] = None,
        varnames: Optional[List[str]] = None,
        cluster_colours: Optional[dict[int, str]] = None,
        cluster_labels: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (700, 500),
) -> go.Figure:
    """
    Create grouped bar-chart subplots of cluster means for each year.

    For each year in `years`, plots the mean value of each feature in
    `selected_features` for every cluster. Subplots are arranged in a
    grid with two columns; colors are assigned per cluster via the
    provided `cluster_colours` mapping or default Plotly palette.

    Parameters:
        network_table (ClusteredNetworkTable):
            A ClusteredNetworkTable containing the data.

        years (List[int] | None):
            Which years to plot. Default is every year in the network table.

        selected_features (List[str]):
            List of feature names to plot on the x-axis

        varnames (List[str] | None):
            The custom variable names to plot

        cluster_colours (dict[int, str] | None):
            A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
            colour map will be used. If a cluster number is not part of the dict, plotly's default
            colour map will be used for that cluster.

        cluster_labels (List[str] | None): 
            A custom list of cluster names. Default is Cluster 0, ..., Cluster n.

        figsize (Tuple[int, float]):
            Width and height of the full figure in pixels


    Returns:
        go.Figure
        A Plotly Figure with one subplot per year, each showing grouped
        bars for clusters across the selected features

    """
    if TSCLUSTER_AVAILABLE:
        return visual_plot_bar_means(
            network_table=network_table,
            selected_features=selected_features,
            years=years,
            varnames=varnames,
            cluster_colours=cluster_colours,
            cluster_labels=cluster_labels,
            figsize=figsize
        )
    else:
        raise ModuleNotFoundError("Sorry, you need to install `tscluster` to use this module of `piccard`. See the documentation for installation instructions.")


def radar_chart_multiple_years(
        network_table: ClusteredNetworkTable,
        selected_cluster: int,
        selected_features: list,
        years: Optional[List[int]] = None,
        varnames: Optional[List[str]] = None,
        year_colours: Optional[dict[int, str]] = None,
        cluster_label: Optional[str] = None,
        figsize: Tuple[int, int] = (700, 500)
) -> go.Figure:
    """
    Creates a radar (polar) chart of selected variables for a given cluster across years

    Parameters:
        network_table (ClusteredNetworkTable):
            A ClusteredNetworkTable containing the data.

        years (List[int] | None):
            Which years to plot. Default is every year in the network table.

        selected_cluster (int):
            Which cluster to plot

        selected_features (List[str]):
            Which variables to plot.

        varnames (List[str] | None):
            The custom variable names to plot
        
        year_colours (dict[int, str] | None):
            A dict mapping indices of years to their corresponding colours. For example, if your
            data goes from 2006 to 2021, 2006 corresponds to index 0, 2011 to 1, etc. If None, plotly's default
            colour map will be used. If a year is not part of the dict, plotly's default
            colour map will be used for that year.

        cluster_label (str | None): 
            The custom label of the cluster to show. Default is Cluster n.

        figsize (width, height):
            Size of the figure in pixels
    
    Returns:
        plotly.graph_objects.Figure:
            The resulting radar chart.
    """
    if TSCLUSTER_AVAILABLE:
        return visual_radar_chart_multiple_years(
            network_table=network_table,
            selected_cluster=selected_cluster,
            selected_features=selected_features,
            years=years,
            varnames=varnames,
            year_colours=year_colours,
            cluster_label=cluster_label,
            figsize=figsize
        )
    else:
        raise ModuleNotFoundError("Sorry, you need to install `tscluster` to use this module of `piccard`. See the documentation for installation instructions.")


def radar_chart_multiple_clusters(
        network_table: ClusteredNetworkTable,
        selected_year: str,
        selected_features: List[str],
        clusters: Optional[List[int]] = None,
        varnames: Optional[List[str]] = None,
        cluster_colours: Optional[dict[int, str]] = None,
        cluster_labels: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = (700, 500),
) -> go.Figure:
    """
    Creates a radar (polar) chart of selected variables for a given year across clusters.

    Parameters:
        network_table (ClusteredNetworkTable):
            A ClusteredNetworkTable containing the data.

        clusters (List[int]):
            Which clusters to plot. Default is every cluster.

        selected_year (str):
            The year for which features will be shown.

        selected_features (List[str]):
            Which variables to plot.

        varnames (List[str] | None):
            The custom variable names to plot

        cluster_colours (dict[int, str] | None):
            A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
            colour map will be used. If a cluster number is not part of the dict, plotly's default
            colour map will be used for that cluster.

        cluster_labels (List[str] | None): 
            A custom list of cluster names. Default is Cluster 0, ..., Cluster n.

        figsize (Tuple[int, int] | None):
            Size of the figure in pixels. Default is (700, 500).

    Returns:
        go.Figure:
            The Plotly figure containing one polar trace per cluster.
    """
    if TSCLUSTER_AVAILABLE:
        return visual_radar_chart_multiple_clusters(
            network_table=network_table,
            selected_year=selected_year,
            selected_features=selected_features,
            clusters=clusters,
            varnames=varnames,
            cluster_colours=cluster_colours,
            cluster_labels=cluster_labels,
            figsize=figsize
        )
    else:
        raise ModuleNotFoundError("Sorry, you need to install `tscluster` to use this module of `piccard`. See the documentation for installation instructions.")


def prob_reasoning_networks(
    network_table_1: NetworkTable | pd.DataFrame | gpd.GeoDataFrame, 
    network_table_2: NetworkTable | pd.DataFrame | gpd.GeoDataFrame, 
    independent_vars_1: List[str], 
    independent_vars_2: List[str], 
    dependent_vars_1: List[str], 
    dependent_vars_2: List[str], 
    mismatches: Optional[dict[str, str]] = None, 
): # -> PDataFrame but python gets mad if I annotate this bc PDataFrame may not be defined
    '''
    Allows probabilistic reasoning over network representations of heterogenous/unlinked datasets using the ppandas package. 
    For more information about ppandas, visit: https://github.com/D3Mlab/ppandas/tree/master
    
    Takes in two network tables and lists of independent and dependent variables for each, performs and visualizes a join,
    and returns the resulting PDataFrame (which can be used to obtain information about conditional probabilities).
    This function is recommended if you have datasets from different sources or datasets that designate geographical
    regions using different units.
    
    The second list of independent variables must be a subset of the first, so make sure the column names are the same
    before passing them into this function. However, mismatches in independent variable column data allowed by ppandas
    are okay.

    Parameters:
        network_table_1 (NetworkTable | pd.DataFrame | gpd.GeoDataFrame): 
            The reference network table. Typically the network table associated with the data assumed to
            be more unbiased and reliable.

        network_table_2 (NetworkTable | pd.DataFrame | gpd.GeoDataFrame):
            The second network table whose independent and dependent variables will be joined into a probabilistic
            model of network_table_1.
        
        independent_vars_1 (List[str]):
            A list of independent variables associated with network_table_1. Must be columns in network_table_1.
        
        independent_vars_2 (List[str]):
            A list of independent variables associated with network_table_2. Must be columns in network_table_2
            and every column in independent_vars_2 must also appear in independent_vars_1.

        dependent_vars_1 (List[str]):
            A list of dependent variables associated with network_table_1. Must be columns in network_table_1.

        dependent_vars_2 (List[str]):
            A list of dependent variables associated with network_table_2. Must be columns in network_table_2.
            Unlike with independent variables, not every column in dependent_vars_2 also has to appear in dependent_vars_1.

        mismatches (dict[str, str] | None):
            A dictionary of the mismatches PDataFrame.pjoin will handle. Must be in format 
            {<independent variable name>: <'categorical' | 'numerical' | 'spatial'> }. See the link above for more information.

    Returns:
        PDataFrame:
            The result of joining the two probabilistic models of network tables.
    '''
    if PPANDAS_AVAILABLE:
        return core_prob_reasoning_networks(network_table_1=network_table_1,
                                            network_table_2=network_table_2,
                                            independent_vars_1=independent_vars_1,
                                            independent_vars_2=independent_vars_2,
                                            dependent_vars_1=dependent_vars_1,
                                            dependent_vars_2=dependent_vars_2,
                                            mismatches=mismatches)
    else:
        raise ModuleNotFoundError("Sorry, you need to install `ppandas` to use this module of `piccard`. See the documentation for installation instructions.")


def prob_reasoning_years(
    network_table: NetworkTable,  
    year_1: str,
    year_2: str,
    independent_vars_1: List[str], 
    independent_vars_2: List[str], 
    dependent_vars_1: List[str], 
    dependent_vars_2: List[str], 
    mismatches: Optional[dict[str, str]] = None
): # -> PDataFrame but python gets mad if I annotate this bc PDataFrame may not be defined
    '''
    Allows probabilistic reasoning over network representations of heterogenous/unlinked datasets using the ppandas package. 
    For more information about ppandas, visit: https://github.com/D3Mlab/ppandas/tree/master
    
    Takes in two years from the same network table and lists of independent and dependent variables for each, performs and visualizes a join,
    and returns the resulting PDataFrame (which can be used to obtain information about conditional probabilities).
    
    The second list of independent variables must be a subset of the first, so make sure the column names are the same
    before passing them into this function. However, mismatches in independent variable column data allowed by ppandas
    are okay.

    Parameters:
        network_table (NetworkTable): 
            The network table containing the data.

        year_1 (str):
            The first year examined.

        year_2 (str):
            The second year examined.
        
        independent_vars_1 (List[str]):
            A list of independent variables associated with year_1. Must be columns in network_table and end in year_1.
        
        independent_vars_2 (List[str]):
            A list of independent variables associated with year_2. Must be columns in network_table and end in year_2.
            The columns (minus year 2) must be a subset of independent_vars_1 (minus year 1).

        dependent_vars_1 (List[str]):
            A list of dependent variables associated with year_1. Must be columns in network_table and end in year_1.

        dependent_vars_2 (List[str]):
            A list of dependent variables associated with year_1. Must be columns in network_table and end in year_1.
            Unlike with independent variables, not every column in dependent_vars_2 also has to appear in dependent_vars_1.

        mismatches (dict[str, str] | None):
            A dictionary of the mismatches PDataFrame.pjoin will handle. Must be in format 
            {<independent variable name>: <'categorical' | 'numerical' | 'spatial'> }. See the link above for more information.

    Returns:
        PDataFrame:
            The result of joining the two probabilistic models.
    '''
    if PPANDAS_AVAILABLE:
        return core_prob_reasoning_years(
            network_table=network_table,
            year_1=year_1,
            year_2=year_2,
            independent_vars_1=independent_vars_1,
            independent_vars_2=independent_vars_2,
            dependent_vars_1=dependent_vars_1,
            dependent_vars_2=dependent_vars_2,
            mismatches=mismatches)
    else:
        raise ModuleNotFoundError("Sorry, you need to install `ppandas` to use this module of `piccard`. See the documentation for installation instructions.")


# Module 4: VariableLinker

def preprocess_census_metadata(path, type_filter = "Total"):
    """
    Preprocess census metadata from a JSON file.
    
    Reads a JSON file containing census metadata, filters for specified type records targetting the type that represents the main skeleton of the tree,
    and restructures the data for further processing.
    
    Args:
        path (str): Path to the JSON file containing census metadata
        type_filter (str): Type of records to filter for (default: "Total")
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with columns ['vector', 'type', 'description', ...]
                        where 'vector' contains the original index values
    """
        
    return VariableLinker.tree_preprocess_census_metadata(path=path, type_filter=type_filter)


def match_descriptions_jaccard(source_df: pd.DataFrame, compare_df: pd.DataFrame, similarity_threshold: float = 0.9):
    """
    Match descriptions between two DataFrames using Jaccard similarity.
    
    This function performs a two-pass matching approach:
    1. First pass: Exact description matches
    2. Second pass: Jaccard similarity matches for unmatched descriptions
    
    Args:
        source_df (pd.DataFrame): Source DataFrame with columns ['vector', 'description']
        compare_df (pd.DataFrame): Comparison DataFrame with columns ['vector', 'description']
        similarity_threshold (float): Minimum similarity score for matching (default: 0.9)
        
    Returns:
        pd.DataFrame: DataFrame with columns ['description', 'vector_base', 'vector_cmp']
                     containing the mapping between source and comparison vectors
    """
    return VariableLinker.tree_match_descriptions_jaccard(source_df=source_df, compare_df=compare_df, similarity_threshold=similarity_threshold)


def match_descriptions_transformer(source_df: pd.DataFrame, compare_df: pd.DataFrame, similarity_threshold: float = 0.9, model_name: str = 'all-mpnet-base-v2'):
    """
    Match descriptions between two DataFrames using sentence transformers.
    
    This function uses pre-trained sentence transformers to compute semantic similarity
    between census descriptions. 
    
    Performs a two-pass matching approach:
    1. First pass: Exact description matches
    2. Second pass: Sentence transformer similarity matches for unmatched descriptions
    
    Args:
        source_df (pd.DataFrame): Source DataFrame with columns ['vector', 'description']
        compare_df (pd.DataFrame): Comparison DataFrame with columns ['vector', 'description']
        similarity_threshold (float): Minimum similarity score for matching (default: 0.9)
        model_name (str): Name of the sentence transformer model to use (default: 'all-mpnet-base-v2')
        
    Returns:
        pd.DataFrame: DataFrame with columns ['description', 'vector_base', 'vector_cmp']
                     containing the mapping between source and comparison vectors
    """
    return VariableLinker.tree_match_descriptions_transformer(source_df=source_df, compare_df=compare_df, similarity_threshold=similarity_threshold, model_name=model_name)


def match_descriptions_details_sentence_transformer( source_df: pd.DataFrame, compare_df: pd.DataFrame, similarity_threshold: float = 0.9, model_name: str = 'all-mpnet-base-v2'):
    """
    Match descriptions using sentence transformers with optimized pre-encoding.
    
    Optimized version of match_descriptions_transformer that pre-encodes all descriptions
    at once for better performance. Performs exact matching on 'description' column first,
    then uses the 'details' column to find the best match when multiple exact matches exist.
    
    Args:
        source_df (pd.DataFrame): Source DataFrame with columns ['vector', 'description', 'details']
        compare_df (pd.DataFrame): Comparison DataFrame with columns ['vector', 'description', 'details']
        similarity_threshold (float): Minimum similarity score for matching (default: 0.9)
        model_name (str): Name of the sentence transformer model to use (default: 'all-mpnet-base-v2')
        
    Returns:
        pd.DataFrame: DataFrame with columns ['description', 'vector_base', 'vector_cmp']
                     containing the mapping between source and comparison vectors
    """
    return VariableLinker.tree_match_descriptions_details_sentence_transformer(source_df=source_df, compare_df=compare_df, similarity_threshold=similarity_threshold, model_name=model_name)


def match_descriptions_multithreaded(
    source_df: pd.DataFrame,
    compare_df: pd.DataFrame,
    similarity_threshold: float = 0.9,
    max_workers: int = 4
) -> pd.DataFrame:
    """
    Multithreaded version of map_descriptions with enhanced similarity matching.
    
    Performs a two-pass matching approach with multithreaded similarity processing:
    1. First pass: Exact description matches (single-threaded)
    2. Second pass: Similarity matches for unmatched descriptions (multithreaded)
    
    Changes in behavior compared to single-threaded version:
    1. Uses mutual exclusion to ensure thread-safe mapping
    2. Processes similarity matching in parallel
    
    Args:
        source_df (pd.DataFrame): Source DataFrame with columns ['vector', 'description']
        compare_df (pd.DataFrame): Compare DataFrame with columns ['vector', 'description']
        similarity_threshold (float): Minimum similarity threshold (default: 0.9)
        max_workers (int): Maximum number of worker threads (default: 4)
        
    Returns:
        pd.DataFrame: DataFrame with columns ['description', 'vector_base', 'vector_cmp']
                    containing the mapping between source and comparison vectors
    """
    return VariableLinker.tree_match_descriptions_multithreaded(
        source_df=source_df,
        compare_df=compare_df,
        similarity_threshold=similarity_threshold,
        max_workers=max_workers
    )


def merge_mappings(map_descriptions, *mappings_dfs):
    """
    Merge multiple mapping DataFrames into a single consolidated mapping.
    
    Takes a base DataFrame (typically 2021 data, or the latest year) and multiple mapping DataFrames,
    then consolidates all matching vectors for each description into a single list.
    
    Args:
        map_descriptions (pd.DataFrame): Base DataFrame with columns ['description', 'vector']
        *mappings_dfs: Variable number of mapping DataFrames with columns 
                      ['description', 'vector_base', 'vector_cmp']
        
    Returns:
        pd.DataFrame: DataFrame with columns ['description', 'vector_base', 'vector_cmp_list']
                     where vector_cmp_list contains all matching vectors from all mappings
    """
    return VariableLinker.tree_merge_mappings(map_descriptions, *mappings_dfs)


def build_tree(source_data, merged_df, tree_name, path = None):
    """
    
    Creates a hierarchical tree visualization using Graphviz, where nodes
    are colored based on how many census years they appear in. The tree structure is
    determined by parent-child relationships in the source_data.
    
    Color coding:
    - Gray: Source year only (no matches in other years)
    - Salmon: Matches in 1 other year
    - Yellow: Matches in 2 other years  
    - Light green: Matches in 3+ other years
    
    Args:
        source_data (pd.DataFrame): DataFrame with parent-child relationships
                                  containing columns ['parent_vector', 'vector']
        merged_df (pd.DataFrame): DataFrame from merge_mappings with columns
                                ['description', 'vector_base', 'vector_cmp_list']
        tree_name (str): Name for the output tree file (without extension)
        path (str, optional): Directory path for saving the tree file (default: current directory)
        
    Returns:
        Digraph: Graphviz Digraph object representing the tree visualization
    """
    return VariableLinker.tree_build_tree(source_data=source_data, merged_df=merged_df, tree_name=tree_name, path=path)


def parse_tree_to_dict(filepath):
    """
    Parse a Graphviz tree file into a dictionary.
    
    This function reads a Graphviz tree file and extracts node information including
    descriptions and year-specific vector mappings. It parses the node labels which
    contain multi-line information about each node.
    
    Args:
        filepath (str): Path to the Graphviz tree file
        
    Returns:
        Dict: Dictionary mapping node IDs to their information
    """
    return VariableLinker.tree_parse_tree_to_dict(filepath=filepath)


def extract_parent_child_relationships(filepath: str) -> Dict[str, List[str]]:
    """
    Extract parent-child relationships from tree file edges.
    
    This function reads a Graphviz tree file and extracts all parent-child relationships
    defined by edges (lines containing " -> "). It creates a mapping where each parent
    node is associated with a list of its child nodes.
    
    Args:
        filepath (str): Path to the tree file (Graphviz format)
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping parent nodes to their children
    """
    return VariableLinker.tree_extract_parent_child_relationships(filepath=filepath)


def predict_parent_nodes(tree_dict: Dict, parent_child_relationships: Dict[str, List[str]], 
                        target_years: List[str]) -> Dict[str, List[str]]:
    """
    Predict parent nodes in other years using the additive property.
    
    This function implements the core prediction algorithm. It identifies parent nodes
    that exist in some years but not others, and determines if they can be predicted
    in missing years by checking if all their children exist in those years.
    
    The additive property means: Parent_Value = Sum(Child_Values)
    If all children exist in a target year, we can predict the parent for that year.
    
    Args:
        tree_dict (Dict): Parsed tree dictionary with node info and year mappings
        parent_child_relationships (Dict[str, List[str]]): Parent to children mapping
        target_years (List[str]): Years to predict parents for (default: ['2016', '2011', '2006'])
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping parent nodes to years they can be predicted in
    """
    return VariableLinker.tree_predict_parent_nodes(tree_dict=tree_dict, parent_child_relationships=parent_child_relationships, target_years=target_years)
