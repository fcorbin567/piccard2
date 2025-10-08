Module 3: Visualization and Analysis
==========================

Module 3 Functions
-------------------

Network Visualizations
---------------------

``plot_subnetwork``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Draws a subgraph of the network representation. If neither a specific list of ids to show nor a specific
list of paths to show are given, picks num_to_sample random nodes from the first census year in the data
and plots a subnetwork of their paths.

Hovering over each node shows the paths the node is part of.

*Parameters:*

* ``network_table`` (NetworkTable): 
    The NetworkTable containing the data.
 
* ``G`` (nx.Graph):
    The network containing the data.

* ``years`` (List[str] | None):
    A list of years to show in the subnetwork. Default is all census years present in the data.

* ``paths_to_show`` (List[int] | None):
    A list of paths (numbered according to their position in network_table) whose points 
    will be plotted in the subnetwork.

* ``ids_to_show`` (List[str] | None):
    A list of ids (use the same type of id you used when creating the graph and network table) that
    will be plotted in the subnetwork. If both ``paths_to_show`` and ``ids_to_show`` are given, the function
    will only consider ``ids_to_show``.

* ``num_to_sample`` (int | None):
    The number of random nodes to plot the paths of in the subnetwork. Default is 4. 
    Note: A large ``num_to_sample`` value may result in an unorganized and hard-to-read visualization.

*Returns:*

* ``plotly.graph_objects.Figure``: 
    The interactive subnetwork plot.


``plot_num_areas``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plots the number of geographical areas across a subset of census years in the data.

*Parameters:*

* ``network_table`` (NetworkTable): 
    The NetworkTable containing the data.

* ``years`` (List[str] | None):
    A list of years to show in the plot. Default is all census years present in the data.

*Returns:*

* ``plotly.graph_objects.Figure``: 
    The plot of the number of geographical areas.


Clustering Visualizations
------------------------

``plot_clusters_scatter``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

*Parameters:*

* ``network_table`` (ClusteredNetworkTable): 
    The ClusteredNetworkTable containing the data.
 
* ``label_dict`` (dict[str, Any] | None):
    A custom label dictionary.

* ``years`` (List[str] | None):
    A list of years to show in the subnetwork. Default is all census years present in the data.

* ``cluster_colours`` (dict[int, str] | None):
    A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
    colour map will be used. If a cluster number is not part of the dict, plotly's default
    colour map will be used for that cluster.

* ``dynamic_paths_only`` (bool | None): 
    A boolean indicating whether to only plot dynamic entities (entities whose cluster
    assignment has changed over time). Default is true.

* ``paths_to_show`` (List[int] | None):
    A list of paths (numbered according to their position in ``network_table``) whose points 
    will be plotted. Default is every path.

* ``ids_to_show`` (List[str] | None):
    A list of ids (use the same type of id you used when creating the graph and network table) that
    will be plotted. Default is every id. If both ``paths_to_show`` and ``ids_to_show`` are given, the function
    will only consider ``ids_to_show``.

* ``clusters_to_show`` (List[int] | None): 
    A list of the clusters whose points will be displayed on the map. Default is every cluster.

* ``clusters_to_exclude`` (List[int] | None): 
    A list of the clusters whose points will NOT be displayed on the map. Default is
    an empty list.
        
* ``figsize`` (Tuple[float, float] | None): 
    A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).
        
* ``cluster_labels`` (List[str] | None): 
    A custom list of cluster names. Default is Cluster 0, ..., Cluster n.

*Returns:*

* ``List[plotly.graph_objects.Figure]``:
    a list of plotly.graph_objects.Figure (you cannot show the whole list; rather, iterate through 
    the list and show each figure)


``plot_clusters_parallelcats``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates an interactive parallel categories (parallel sets) plot to visualize how cluster 
assignments evolve over time.

Each column in the plot corresponds to a time point (e.g., a census year), and each
path across the columns represents a "temporal path" of a tract or unit as it transitions
across categories.

*Parameters:*

* ``network_table`` (ClusteredNetworkTable): 
    The ClusteredNetworkTable containing the data.

* ``years`` (List[str] | None):
    A list of years to show. Default is all census years present in the data.

* ``cluster_colours`` (dict[int, str] | None):
    A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
    colour map will be used. If a cluster number is not part of the dict, plotly's default
    colour map will be used for that cluster.

* ``colour_index_year`` (str | None):
    The year that will be used to determine the colours of the parallel plot. For example, if you chose
    2011 as the colour index year, every cluster in the 2011 dimension would have a colour assigned to it,
    and then the paths into and out of these clusters would be shown in those colours. Default is the
    first year in the network table, and if an invalid input is given, the default will be used.

* ``cluster_labels`` (List[str] | None): 
    A custom list of cluster names. Default is Cluster 0, ..., Cluster n.
        
* ``figsize`` (Tuple[float, float] | None): 
    A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).

*Returns:*

* ``plotly.graph_objects.Figure``:
    The interactive parallel categories plot.


``plot_clusters_area``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates an interactive area chart to visualize how cluster assignments evolve over time.

Each column in the plot corresponds to a time point (e.g., a census year), and each
path across the columns represents a "temporal path" of a tract or unit as it transitions
across categories.

*Parameters:*

* ``network_table`` (ClusteredNetworkTable): 
    The ClusteredNetworkTable containing the data.

* ``years`` (List[str] | None):
    A list of years to show. Default is all census years present in the data.

* ``cluster_colours`` (dict[int, str] | None):
    A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
    colour map will be used. If a cluster number is not part of the dict, plotly's default
    colour map will be used for that cluster.

* ``cluster_labels`` (List[str] | None): 
    A custom list of cluster names. Default is Cluster 0, ..., Cluster n.
        
* ``figsize`` (Tuple[float, float] | None): 
    A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).

* ``stacked`` (bool | None):
    Whether to show the area plot as a stacked plot, with all the areas on top of each other. If False,
    shows the area plot as a regular line graph. Default is True.

*Returns:*

* ``plotly.graph_objects.Figure``:
    The interactive area plot.


``plot_clusters_map``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Plots cluster assignments in their associated geographical regions for a specific year using a GeoDataFrame.

*Parameters:*

* ``geofile_path`` (str):
    Path to geographical data file

* ``network_table`` (ClusteredNetworkTable):
    Network table to be merged with GeoJSON

* ``year`` (str):
    Year to visualize (used in column name)

* ``cluster_colours`` (dict[int, str] | None):
    A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
    colour map will be used. If a cluster number is not part of the dict, plotly's default
    colour map will be used for that cluster.

* ``label_dict`` (dict[str, Any] | None):
    The label dictionary from pc.clustering_prep() that you used in pc.cluster() or a custom 
    label dictionary. Used to determine what data will be shown when you hover over each geographical
    region. If None, only the index (path number) will be shown.

* ``cluster_labels`` (List[str] | None): 
    A custom list of cluster names. Default is Cluster 0, ..., Cluster n.

* ``figsize`` (Tuple[float, float] | None):
    A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).

*Returns:*

* ``plotly.express.choropleth``: 
    The interactive choropleth map


``plot_line_means``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates an interactive line chart with one subplot per feature, showing how
cluster mean values evolve over the selected years.

For each year in ``years``, plots the mean value of each feature in
``selected_features`` for every cluster.

*Parameters:*

* ``network_table`` (ClusteredNetworkTable): 
    The ClusteredNetworkTable containing the data.

* ``years`` (List[str] | None):
    A list of years to show. Default is all census years present in the data.

* ``selected_features`` (List[str]):
    Which features (column names present in clustering) to plot
        
* ``varnames`` (List[str] | None):
    The custom variable names to plot

* ``cluster_colours`` (dict[int, str] | None):
    A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
    colour map will be used. If a cluster number is not part of the dict, plotly's default
    colour map will be used for that cluster.

* ``cluster_labels`` (List[str] | None): 
    A custom list of cluster names. Default is Cluster 0, ..., Cluster n.
        
* ``figsize`` (Tuple[float, float] | None): 
    A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).

*Returns:*

* ``plotly.graph_objects.Figure``:
    The line chart with subplots.


``plot_bar_means``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create grouped bar-chart subplots of cluster means for each year.

For each year in ``years``, plots the mean value of each feature in
``selected_features`` for every cluster. Subplots are arranged in a
grid with two columns.

*Parameters:*

* ``network_table`` (ClusteredNetworkTable): 
    The ClusteredNetworkTable containing the data.

* ``years`` (List[str] | None):
    A list of years to show. Default is all census years present in the data.

* ``selected_features`` (List[str]):
    Which features (column names present in clustering) to plot
        
* ``varnames`` (List[str] | None):
    The custom variable names to plot

* ``cluster_colours`` (dict[int, str] | None):
    A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
    colour map will be used. If a cluster number is not part of the dict, plotly's default
    colour map will be used for that cluster.

* ``cluster_labels`` (List[str] | None): 
    A custom list of cluster names. Default is Cluster 0, ..., Cluster n.
        
* ``figsize`` (Tuple[float, float] | None): 
    A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).

*Returns:*

* ``plotly.graph_objects.Figure``:
    The bar chart with subplots.


``radar_chart_multiple_years``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates a radar (polar) chart of selected variables for a given cluster across years.

*Parameters:*

* ``network_table`` (ClusteredNetworkTable): 
    The ClusteredNetworkTable containing the data.

* ``years`` (List[str] | None):
    A list of years to show. Default is all census years present in the data.

* ``selected_cluster`` (int):
    Which cluster to plot

* ``selected_features`` (List[str]):
    Which features (column names present in clustering) to plot
        
* ``varnames`` (List[str] | None):
    The custom variable names to plot

* ``year_colours`` (dict[int, str] | None):
    A dict mapping indices of years to their corresponding colours. For example, if your
    data goes from 2006 to 2021, 2006 corresponds to index 0, 2011 to 1, etc. If None, plotly's default
    colour map will be used. If a year is not part of the dict, plotly's default
    colour map will be used for that year.

* ``cluster_label`` (str | None): 
    The custom label of the cluster to show. Default is Cluster n.
        
* ``figsize`` (Tuple[float, float] | None): 
    A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).

*Returns:*

* ``plotly.graph_objects.Figure``:
    The radar chart.


``radar_chart_multiple_clusters``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates a radar (polar) chart of selected variables for a given year across clusters.

*Parameters:*

* ``network_table`` (ClusteredNetworkTable): 
    The ClusteredNetworkTable containing the data.

* ``clusters`` (List[int] | None):
    A list of clusters to show. Default is all clusters present in the data.

* ``selected_year`` (str):
    Which year to plot

* ``selected_features`` (List[str]):
    Which features (column names present in clustering) to plot
        
* ``varnames`` (List[str] | None):
    The custom variable names to plot

* ``cluster_colours`` (dict[int, str] | None):
    A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
    colour map will be used. If a cluster number is not part of the dict, plotly's default
    colour map will be used for that cluster.

* ``cluster_labels`` (List[str] | None): 
    A custom list of cluster names. Default is Cluster 0, ..., Cluster n.
        
* ``figsize`` (Tuple[float, float] | None): 
    A tuple indicating the width and height of each figure that will be shown. Default is (700, 500).

*Returns:*

* ``plotly.graph_objects.Figure``:
    The radar chart.


Probabilistic Analysis
------------------------

``prob_reasoning_networks``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Allows probabilistic reasoning over network representations of heterogenous/unlinked datasets using the ``ppandas`` package. 
For more information about ``ppandas``, visit: https://github.com/D3Mlab/ppandas/tree/master
    
Takes in two network tables and lists of independent and dependent variables for each, performs and visualizes a join,
and returns the resulting PDataFrame (which can be used to obtain information about conditional probabilities).
This function is recommended if you have datasets from different sources or datasets that designate geographical
regions using different units.
    
The second list of independent variables must be a subset of the first, so make sure the column names are the same
before passing them into this function. However, mismatches in independent variable column data allowed by ``ppandas``
are okay.

*Parameters:*

* ``network_table_1`` (NetworkTable \| pd.DataFrame \| gpd.GeoDataFrame): 
    The reference network table. Typically the network table associated with the data assumed to
    be more unbiased and reliable.

* ``network_table_2`` (NetworkTable \| pd.DataFrame \| gpd.GeoDataFrame):
    The second network table whose independent and dependent variables will be joined into a probabilistic
    model of network_table_1.
        
* ``independent_vars_1`` (List[str]):
    A list of independent variables associated with network_table_1. Must be columns in network_table_1.
        
* ``independent_vars_2`` (List[str]):
    A list of independent variables associated with network_table_2. Must be columns in network_table_2
    and every column in independent_vars_2 must also appear in independent_vars_1.

* ``dependent_vars_1`` (List[str]):
    A list of dependent variables associated with network_table_1. Must be columns in network_table_1.

* ``dependent_vars_2`` (List[str]):
    A list of dependent variables associated with network_table_2. Must be columns in network_table_2.
    Unlike with independent variables, not every column in dependent_vars_2 also has to appear in dependent_vars_1.

* ``mismatches`` (dict[str, str] | None):
    A dictionary of the mismatches PDataFrame.pjoin will handle. Must be in format 
    {<independent variable name>: <'categorical' \| 'numerical' \| 'spatial'> }. See the link above for more information.

*Returns:*

* ``PDataFrame``:
    The result of joining the two probabilistic models of network tables.


``prob_reasoning_years``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Allows probabilistic reasoning over network representations of heterogenous/unlinked datasets using the ``ppandas`` package. 
For more information about ``ppandas``, visit: https://github.com/D3Mlab/ppandas/tree/master
    
Takes in two years from the same network table and lists of independent and dependent variables for each, performs and visualizes a join,
and returns the resulting PDataFrame (which can be used to obtain information about conditional probabilities).
    
The second list of independent variables must be a subset of the first, so make sure the column names are the same
before passing them into this function. Mismatches in independent variable column data allowed by ``ppandas``
are okay.

*Parameters:*

* ``network_table`` (NetworkTable): 
    The network table containing the data.

* ``year_1`` (str):
    The first year examined.

* ``year_2`` (str):
    The second year examined.
        
* ``independent_vars_1`` (List[str]):
    A list of independent variables associated with year_1. Must be columns in network_table and end in year_1.
        
* ``independent_vars_2`` (List[str]):
    A list of independent variables associated with year_2. Must be columns in network_table and end in year_2.
    The columns (minus year 2) must be a subset of independent_vars_1 (minus year 1).

* ``dependent_vars_1`` (List[str]):
    A list of dependent variables associated with year_1. Must be columns in network_table and end in year_1.

* ``dependent_vars_2`` (List[str]):
    A list of dependent variables associated with year_1. Must be columns in network_table and end in year_1.
    Unlike with independent variables, not every column in dependent_vars_2 also has to appear in dependent_vars_1.

* ``mismatches`` (dict[str, str] | None):
    A dictionary of the mismatches PDataFrame.pjoin will handle. Must be in format 
    <independent variable name>: <'categorical' \| 'numerical' \| 'spatial'> }. See the link above for more information.

*Returns:*

* ``PDataFrame``:
    The result of joining the two probabilistic models.
