import math
import numpy as np
import geopandas as gpd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from itertools import cycle, islice
from tscluster.opttscluster import OptTSCluster
from tscluster.greedytscluster import GreedyTSCluster
from tscluster.preprocessing.utils import load_data, tnf_to_ntf, ntf_to_tnf
import pandas as pd # for type annotations
import networkx as nx # for type annotations
from typing import Union, Any, List, Tuple, Optional # for type annotations

import warnings
warnings.filterwarnings('ignore')

# Clustering

def clustering_prep(
    network_table:pd.DataFrame, 
    id:str, 
    cols:Optional[list[str]]=[]
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
        
    Optional Parameters:   
        cols (list[str]): A list of the names of network table columns that should be considered in
            the clustering algorithm. If none, every numerical feature will be considered. Leaving it none is
            not recommended as many numerical features, such as network level, have little bearing on the data.

    Returns:
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
    network_table:pd.DataFrame, 
    G:nx.Graph, 
    id:str, 
    num_clusters:int, 
    algo:Optional[str]='greedy', 
    scheme:Optional[str]='z1c1', 
    arr:Optional[np.ndarray[np.float64]]=None, 
    label_dict:Optional[dict[str, Any]]=None
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

    Optional Parameters:
        algo (str): 
            The algorithm that tscluster will use, either 'greedy' (default) or 'opt'.
            'greedy' runs GreedyTSCluster, which is a faster and easier, but less accurate, method than OptTSCluster. 
            Since it doesn't require a special academic licence, we recommend 'greedy' for any non-academic users.
            'opt' runs OptTSCluster, which is guaranteed to find the optimal clustering but requires a Gurobi academic
            licence to run the clustering algorithm. More information about obtaining an academic licence can be found
            here: https://www.gurobi.com/academia/academic-program-and-licenses/
        
        scheme (str): 
            the clustering scheme. See the first paragraph for more information. Default is 'z1c1'.

        arr (np.ndarray[np.float64]): 
            the array of data to be clustered. If none, arr and label_dict will be generated by running
            pc.clustering_prep() with the default columns. See the pc.clustering_prep() documentation for why we DO NOT
            recommend leaving this blank.
        
        label_dict (dict[str, Any]): 
            the label dictionary corresponding to the data array. See 'arr'.

    Returns:
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

def plot_clusters_scatter(
    network_table: pd.DataFrame,
    arr: np.ndarray[np.float64],
    label_dict: dict[str, Any],
    tsc: Union[OptTSCluster, GreedyTSCluster],
    years: Optional[List[str]] = None,
    cluster_colours: Optional[dict[int, str]] = None,
    dynamic_paths_only: Optional[bool] = True,
    paths_to_show: Optional[List[int]] = None,
    clusters_to_show: Optional[List[int]] = None, 
    clusters_to_exclude: Optional[List[int]] = [],
    figsize: Optional[Tuple[float, float]] = (700, 500),
    cluster_labels: Optional[List[str]] = None,
    hover_labels: Optional[bool] = True,
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
    
    Optional Parameters:
        years (List[str]): 
            The years displayed on the map. Default is all years in the network table.

        cluster_colours (dict[int, str]):
            A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
            colour map will be used. If a cluster number is not part of the dict, plotly's default
            colour map will be used for that cluster.

        dynamic_paths_only (bool): 
            A boolean indicating whether to only plot dynamic entities (entities whose cluster
            assignment has changed over time). Default is true.

        paths_to_show (List[int]): 
            A list of paths (numbered according to their position in network_table) whose points 
            will be displayed on the map. Default is every path.
        
        clusters_to_show (List[int]): 
            A list of the clusters whose points will be displayed on the map. Default is every cluster.

        clusters_to_exclude (List[int]): 
            A list of the clusters whose points will NOT be displayed on the map. Default is
            an empty list.
        
        figsize (Tuple[float, float]): 
            A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).
        
        cluster_labels (List[str]): 
            A custom list of cluster names. Default is Cluster 0, ..., Cluster n.

        hover_labels (bool): 
            A boolean indicating whether you can see the x-value, y-value, and path number of
            each point on the plot if you hover your cursor over the point. Default is true.

    Returns:
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
        paths_to_show = list(range(arr.shape[1]))  # show all
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
    if dynamic_paths_only:
        dynamic_entities = set(tsc.get_dynamic_entities()[0])
        paths_to_show = [i for i in paths_to_show if i in dynamic_entities]

    # create list of figures and iterate through features
    figures = []
    for f in range(F):
        fig = go.Figure()
        used_clusters = set()
        # iterate through each path for the given feature
        for i in paths_to_show:
            mode = 'lines+markers' if hover_labels else 'lines'
            # plot lines indicating values
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=arr[:, i, f],
                    mode=mode,
                    line=dict(color='black', dash='dot'),
                    showlegend=False
                )
            )
            # plot coloured dots indicating cluster
            label_i = labels[i] if labels.ndim == 1 else labels[i, 0]
            used_clusters.add(int(label_i))
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=arr[:, i, f],
                    mode='markers',
                    marker=dict(color=colors[int(label_i)], size=6),
                    name=f"Path {i}",
                    showlegend=False
                )
            )
        # plot cluster centres
        for j in range(K):
            if j in used_clusters:
                mode = 'lines+markers' if hover_labels else 'lines'
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
            with names in the format <feature_name>_<year>, e.g., 'cluster_assignment_2016'. (This
            will automatically be done if you have already run pc.cluster())

    Optional Parameters:
        years (List[str]):
            A list of strings representing the time points to include, such as ['2011', '2016', '2021'].
            Default is every year in the network table.

        cluster_colours (dict[int, str]):
            A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
            colour map will be used. If a cluster number is not part of the dict, plotly's default
            colour map will be used for that cluster.
        
        colour_index_year (str):
            The year that will be used to determine the colours of the parallel plot. For example, if you chose
            2011 as the colour index year, every cluster in the 2011 dimension would have a colour assigned to it,
            and then the paths into and out of these clusters would be shown in those colours. Default is the
            first year in the network table, and if an invalid input is given, the default will be used.
        
        cluster_labels (List[str]): 
            A custom list of cluster names. Default is Cluster 0, ..., Cluster n.

        figsize (Tuple[float, float]): 
            A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).

    Returns:
        plotly.graph_objects.Figure: The interactive map
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

    Optional Parameters:
        years (List[str]):
            A list of strings representing the time points to include, such as ['2011', '2016', '2021'].
            Default is every year in the network table.

        cluster_colours (dict[int, str]):
            A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
            colour map will be used. If a cluster number is not part of the dict, plotly's default
            colour map will be used for that cluster.
        
        cluster_labels (List[str]): 
            A custom list of cluster names. Default is Cluster 0, ..., Cluster n.

        figsize (Tuple[float, float]): 
            A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).

        stacked (bool):
            Whether to show the area plot as a stacked plot, with all the areas on top of each other. If False,
            shows the area plot as a regular line graph. Default is True.

    Returns:
        plotly.graph_objects.Figure: The interactive map
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
    cluster_colours: Optional[dict[int, str]] = None,
    label_dict: Optional[dict[str, Any]] = None,
    cluster_labels: Optional[List[str]] = None,
    geofile_path: Optional[str] = None,
    network_table: Optional[pd.DataFrame] = None,
    gdf: Optional[gpd.GeoDataFrame] = None,
    figsize: Optional[Tuple[float, float]] = (700, 500),
) -> px.choropleth:
    """
    Plot cluster assignments in their associated geographical regions for a specific year using a GeoDataFrame.
    To properly load the geographical data, the user must provide AT LEAST ONE of the following:
    1. geofile_path AND network_table
    2. gdf

    Parameters:
        year (str):
            Year to visualize (used in column name)
    
    Optional Parameters:
        cluster_colours (dict[int, str]):
            A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
            colour map will be used. If a cluster number is not part of the dict, plotly's default
            colour map will be used for that cluster.

        label_dict(dict[str, Any]):
            The label dictionary from pc.clustering_prep() that you used in pc.cluster() or a custom 
            label dictionary. Used to determine what data will be shown when you hover over each geographical
            region. If None, only the index (path number) will be shown.

        cluster_labels (List[str]): 
            A custom list of cluster names. Default is Cluster 0, ..., Cluster n.

        geofile_path (str):
            Path to geographical data file if gdf is not passed

        network_table (pd.DataFrame):
            Network table to be merged with GeoJSON

        gdf (GeoDataFrame):
            Pre-joined GeoDataFrame (recommended for advanced users)

        figsize (Tuple[float, float]):
            A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).

    Returns:
        plotly.express.choropleth: The interactive choropleth map
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

    # set hover data based on label_dict
    hover_data = {}
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

# Helpers

def filter_columns(
    network_table:pd.DataFrame, 
    years:List[str], 
    cols:Optional[list[str]]=[]
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
        
    Optional Parameters:   
        cols (list[str]): A list of the names of network table columns that should be considered in
            the clustering algorithm. If none, every numerical feature will be considered. Leaving it none is
            not recommended as many numerical features, such as network level, have little bearing on the data.
    
    Returns:
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
    network_table_id_column: Optional[str] = "geouid"
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
    
    Optional Parameters:
        geofile_id_col (str):
            Column name in the geographical data file that contains the geographic identifier
            (default: 'GeoUID').

        network_table_id_column (str):
            Prefix of the column name in the network_table used for geographic ID matching.
            The function expects a column like 'geouid_2016' if year='2016'.

    Returns:
        gpd.GeoDataFrame:
        A merged GeoDataFrame containing geometry from the GeoJSON file and attribute
        data (e.g., cluster assignments) from the network table. Only valid, non-empty
        geometries are retained.
    """

    # Validate input column name
    geoid_col = f"{network_table_id_column}_{year}"
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
