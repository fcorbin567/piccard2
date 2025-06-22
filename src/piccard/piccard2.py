import math
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
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


def clustering_prep(
    network_table:pd.DataFrame, 
    id:str, 
    cols:list=[]
) -> tuple[np.ndarray[np.float64], dict[str, Any]]:
    '''
    Converts a piccard network table into a 3d numpy array of all possible paths and their corresponding
    features. This will be used for clustering with tscluster.
    The user can (optionally) input a list of columns that they want to be considered in the clustering algorithm, 
    and the function will check that these columns are valid.

    Note that you must run pc.create_network_table() before this function.

    Inputs:
    - network_table: The result of pc.create_network_table().
    - id: The same id inputted into pc.create_network_table().
    - cols (optional): A list of the names of network table columns that should be considered in
    the clustering algorithm. If none, every numerical feature will be considered. This is
    not recommended as many numerical features, such as network level, have little bearing on the data.

    Returns a tuple of a 3d numpy array and a corresponding dictionary of labels showing
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
    algo:str='greedy', 
    scheme:str='z1c1', 
    arr:np.ndarray[np.float64] | None =None, 
    label_dict:dict[str, Any] | None =None
) -> Union[OptTSCluster, GreedyTSCluster]:
    '''
    Runs one of tscluster's clustering algorithms (default is fully dynamic clustering or 'z1c1')
    and adds the resulting cluster assignments to the network table and nodes as an additional feature.
    Information about the different clustering algorithms is available here: https://tscluster.readthedocs.io/en/latest/introduction.html
    We recommend either Sequential Label Analysis ('z1c0') or the default 'z1c1'.

    Users can choose to only input the network table, in which case clustering_prep will be run for them with the default columns,
    or they can choose to run clustering_prep on their own and then have the option to apply one or both of the
    normalization methods available in tscluster.preprocessing.utils.

    Inputs:
    - network_table: The result of pc.create_network_table().
    - G: The result of pc.create_network().
    - id: The same id inputted into pc.create_network_table().
    - num_clusters: The number of clusters that the algorithm will find.
    - algo (optional): The algorithm that tscluster will use, either 'greedy' (default) or 'opt'.
    'greedy' runs GreedyTSCluster, which is a faster and easier, but less accurate, method than OptTSCluster. 
    Since it doesn't require a special academic licence, we recommend 'greedy' for any non-academic users.
    'opt' runs OptTSCluster, which is guaranteed to find the optimal clustering but requires a Gurobi academic
    licence to run the clustering algorithm. More information about obtaining an academic licence can be found
    here: https://www.gurobi.com/academia/academic-program-and-licenses/
    - scheme (optional): the clustering scheme. See the first paragraph for more information. Default is 'z1c1'.
    - arr (optional): the array of data to be clustered. If none, arr and label_dict will be generated by running
    pc.clustering_prep() with the default columns. See the pc.clustering_prep() documentation for why we DO NOT
    recommend leaving this blank.
    - label_dict (optional): the label dictionary corresponding to the data array.

    Returns an OptTSCluster or GreedyTSCluster object with useful labels, cluster assignments, etc for future visualizations.
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

def plot_clusters(
    network_table: pd.DataFrame,
    arr: np.ndarray[np.float64],
    label_dict: dict[str, Any],
    tsc: Union[OptTSCluster, GreedyTSCluster],
    dynamic_paths_only: bool = True,
    paths_to_show: List[int] | None = None,
    clusters_to_show: List[int] | None = None, 
    clusters_to_exclude: List[int] = [],
    figsize: Tuple[float, float] = (700, 500),
    cluster_labels: List[str] | None = None,
    x_rotation: float | int = 45,
    hover_labels: bool = True,
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

    Inputs:
    - network_table: The result of pc.create_network_table().
    - arr, label_dict: The result of pc.clustering_prep(). label_dict could also be a custom label dictionary.
    - tsc: The result of pc.cluster() (an OptTSCluster object).
    - dynamic_paths_only (optional): A boolean indicating whether to only plot dynamic entities (entities whose cluster
    assignment has changed over time). Default is true.
    - paths_to_show (optional): A list of paths (numbered according to their position in network_table) whose points 
    will be displayed on the map. Default is every path.
    - clusters_to_show (optional): A list of the clusters whose points will be displayed on the map. Default is every cluster.
    - clusters_to_exclude (optional): A list of the clusters whose points will NOT be displayed on the map. Default is
    an empty list.
    - figsize (optional): A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).
    - cluster_labels (optional): A custom list of cluster names. Default is Cluster 0, ..., Cluster n.
    - x_rotation (optional): How many degrees to rotate the ticks on the x axis by. You may want to use this parameter
    if you have a lot of timesteps and the x axis will be crowded. Default is 45 degrees.
    - hover_labels (optional): A boolean indicating whether you can see the x-value, y-value, and path number of
    each point on the plot if you hover your cursor over the point. Default is true.

    Returns a list of plotly figures (you cannot show the whole list; rather, iterate through the list and show each figure)
    '''

    # get necessary data from tsc
    cluster_centres = tsc.cluster_centers_
    labels = tsc.labels_

    # prepare the variables we will use to iterate through features and cluster centres
    F = arr.shape[2]
    K = cluster_centres.shape[1] if cluster_centres is not None else np.unique(labels).size

    # set default values and colours
    if paths_to_show is None:
        paths_to_show = list(range(arr.shape[1]))  # show all
    if clusters_to_show is None:
        clusters_to_show = list(range(K))
    if cluster_labels is None:
        cluster_labels = [str(i) for i in range(K)]
    colors = plotly.colors.qualitative.Plotly
    if K > len(colors):
        colors = list(islice(cycle(colors), K))

    # filter entities
    paths_to_show = [
        i for i in paths_to_show
        if any(int(c) in clusters_to_show for c in network_table.iloc[i][-len(label_dict['T']):])
        and all(int(c) not in clusters_to_exclude for c in network_table.iloc[i][-len(label_dict['T']):])
    ]
    if dynamic_paths_only:
        dynamic_entities = set(tsc.get_dynamic_entities()[0])
        paths_to_show = [i for i in paths_to_show if i in dynamic_entities]

    figures = []

    for f in range(F):
        fig = go.Figure()
        used_clusters = set()
        # iterate through each path
        for i in paths_to_show:
            mode = 'lines+markers' if hover_labels else 'lines'
            # plot lines indicating values
            fig.add_trace(
                go.Scatter(
                    x=label_dict['T'],
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
                    x=label_dict['T'],
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
                        x=label_dict['T'],
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
            xaxis=dict(tickangle=x_rotation),
            legend_title="Legend",
        )
        figures.append(fig)

    return figures


def parallel_plot(
    network_table: pd.DataFrame, 
    feature_name: str, 
    years: List[str], 
    title: str = "Tract Paths Across Years",
    height: int = 600
) -> go.Figure:
    """
    Creates an interactive parallel categories (parallel sets) plot using Plotly Express
    to visualize how categorical features (e.g., cluster assignments) evolve over time.

    Each column in the plot corresponds to a time point (e.g., a census year), and each
    path across the columns represents a "temporal path" of a tract or unit as it transitions
    across categories.

    Parameters:
        network_table (pd.DataFrame):
            A DataFrame containing the data. Expected to include one column per year
            with names in the format <feature_name>_<year>, e.g., 'cluster_assignment_2016'.

        feature_name (str):
            The prefix of the column names to be visualized (e.g., 'cluster_assignment').

        years (List[str]):
            A list of strings representing the time points to include, such as ['2011', '2016', '2021'].

        Optional Parameters:
            title (str, optional):
                Title for the plot. Default is 'Tract Paths Across Years'.

            height (int, optional):
                Height of the plot in pixels. Default is 600.

    Returns:
        plotly.graph_objects.Figure:
            A Plotly Figure object that can be displayed with .show() or used in dashboards.

    """

    # Construct column names dynamically from feature name and years
    columns = [f"{feature_name}_{year}" for year in years]

    # Ensure all target columns exist in the dataframe
    for col in columns:
        if col not in network_table.columns:
            raise ValueError(f"Column '{col}' not found in the provided DataFrame.")

    # Generate the plot
    fig = px.parallel_categories(
        network_table,
        dimensions=columns,
        labels={col: f"{col[-4:]}" for col in columns},
        title=title,
        template="plotly_white",
        height=height, )
    fig.update_layout(font_size=15)
    return fig


def cluster_count_plot(
    network_table: pd.DataFrame,
    feature_name: str,
    years: List[str],
    title: str = "Cluster",
    x_label: str = "Year",
    y_label: str = "Number of Census Tracts",
    legend_title: str = "Cluster",
    figure_size: Tuple[int, int] = (12, 6),
    stacked: bool = True,
) -> plt:
    """
    Plots the change in cluster composition over time using an area chart.

    Each area represents the number of units (e.g., census tracts) assigned to a specific cluster
    across different years. The input DataFrame should contain categorical cluster assignments
    per year with column names formatted as <feature_name>_<year>.

    Parameters:
        network_table (pd.DataFrame):
            A DataFrame containing the data. Expected to include one column per year
            with names in the format <feature_name>_<year>, e.g., 'cluster_assignment_2016'.

        feature_name (str):
            The prefix of the column names to be visualized (e.g., 'cluster_assignment').

        years (List[str]):
            A list of strings representing the time points to include, such as ['2011', '2016', '2021'].

        Optional Parameters:
            title (str):
            Plot title. Default is "Cluster Composition Over Time".

            x_label (str):
                X-axis label. Default is "Year".

            y_label (str):
                Y-axis label. Default is "Number of Census Tracts".

            legend_title (str):
                Title for the legend. Default is "Cluster".

            figure_size (Tuple[int, int]):
                Size of the figure. Default is (12, 6).

            stacked (bool):
                Whether to stack the area plot. Default is True.
    Returns:
       plt or matplotlib.figure.Figure:
            Matplotlib plot or figure object
    """

    # Count number of tracts in each cluster for that year and sort the cluster labels
    cluster_counts = pd.DataFrame()
    for year in years:
        column = f'{feature_name}_{year}'
        if column not in network_table.columns:
            raise ValueError(f"Expected column '{column}' not found in DataFrame.")
        counts = network_table[
            column].value_counts().sort_index()  # Sorting just to ensure the ordering will be correct
        cluster_counts[year] = counts

    # Transpose so years are rows (X-axis) and clusters are columns (stacked areas)
    cluster_counts = cluster_counts.astype(int).T

    # Plot setup
    figure, axes = plt.subplots(figsize=figure_size)
    cluster_counts.plot(kind="area", stacked=stacked, ax=axes, cmap="tab20")

    axes.set_title(title, fontsize=16)
    axes.set_xlabel(x_label, fontsize=12)
    axes.set_ylabel(y_label, fontsize=12)

    # Legend outside the plot for readability
    plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')  # Adding color keys
    plt.tight_layout()  # Adjusting the padding

    return plt


def cluster_map_plot(
    year: str,
    cluster_col_prefix: str,
    geofile_path: Optional[str] = None,
    network_table: Optional[pd.DataFrame] = None,
    gdf: Optional[gpd.GeoDataFrame] = None,
    figure_size: tuple = (10, 10),
) -> plt.Figure:
    """
    Plot cluster assignments for a specific year using a GeoDataFrame

    Parameters:
        year (str):
            Year to visualize (used in column name)

        cluster_col_prefix (str):
            Prefix for cluster assignment column (e.g., 'cluster_assignment')

        geofile_path (str, optional):
            Path to geographical data file if gdf is not passed

        network_table (pd.DataFrame, optional):
            Network table to be merged with GeoJSON

        gdf (GeoDataFrame, optional):
            Pre-joined GeoDataFrame (recommended for advanced users)

        figure_size (tuple):
            Size of the figure

    Returns:
        matplotlib.figure.Figure: The figure object for further use or saving
    """

    cluster_col = f"{cluster_col_prefix}_{year}"

    # Load and join if no GeoDataFrame was passed
    if gdf is None:
        if geofile_path is None or network_table is None:
            raise ValueError("Either `gdf` or both `geofile_path` and `network_table` must be provided.")
        gdf = join_geometries(geofile_path, network_table, year)

    # Ensure the cluster column exists
    if cluster_col not in gdf.columns:
        raise ValueError(f"Column '{cluster_col}' not found in GeoDataFrame.")

    # Once we have the data we will make the plot
    gdf[cluster_col] = gdf[cluster_col].astype(str)

    # Plot
    figure, axes = plt.subplots(1, 1, figsize=figure_size)
    gdf.plot(
        column=cluster_col,
        cmap="tab10",
        linewidth=0.2,
        edgecolor='grey',
        legend=True,
        ax=axes
    )

    axes.set_title(f"Clusters in {year}", fontsize=15)
    axes.axis("off")

    plt.tight_layout()  # Adjusting the padding

    return figure

# Helpers

def filter_columns(network_table, years, cols=[]):
    '''
    Checks that the list of columns with data to be clustered is valid in the following ways:
    - Makes sure all the data in the columns are numerical or nan
    - Makes sure there is a version of each column for every year
    Returns a tuple of the final filtered list of columns and the column labels that will
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
        geofile_id_col: str = "GeoUID",
        network_table_id_column: str = "geouid"
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
