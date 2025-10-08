import numpy as np
import pandas as pd
from typing import Optional, List, Any, Tuple
from itertools import cycle, islice
import plotly.graph_objects as go
import plotly
import plotly.express as px
from plotly.subplots import make_subplots
import math
from ..core.clustering import *
from ..core.network import core_join_geometries

def visual_plot_clusters_scatter(
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
    table = network_table.table
    tsc = network_table.tsc
    arr = network_table.arr
    if label_dict is not None:
        network_table.modify_label_dict(label_dict)
    label_dict = network_table.label_dict

    # get necessary data from tsc
    cluster_centres = tsc.cluster_centers_
    labels = tsc.labels_

    # prepare the variables we will use to iterate through features and cluster centres
    F = arr.shape[2]
    K = network_table.num_clusters

    # verify years exist in network table
    if years is None:
        years = network_table.years
    for year in years:
        column = f'cluster_assignment_{year}'
        if column not in table.columns:
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
    new_table = table.copy(deep=True)
    for year in years:
        if year not in years:
            new_table[f'cluster_assignment_{year}'] = [
                np.nan for _ in range(len(table[f'cluster_assignment_{year}']))]

    # filter entities using paths_to_show, clusters_to_show, clusters_to_exclude, dynamic_paths_only
    paths_to_show = [
        i for i in paths_to_show
        if any(int(c) in clusters_to_show for c in new_table.iloc[i][-len(label_dict['T']):])
        and all(int(c) not in clusters_to_exclude for c in new_table.iloc[i][-len(label_dict['T']):])
    ]
    if ids_to_show is not None:
        paths_to_show = [
        i for i in paths_to_show
        if any(c in ids_to_show for c in [table.iloc[i][label_dict['T'].index(j)] for j in years])
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
            path_ids = [table.iloc[i][label_dict['T'].index(j)] for j in years]
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

def visual_plot_clusters_parallelcats(
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
    # add default years and cluster labels
    table = network_table.table
    num_clusters = network_table.num_clusters
    if years is None:
        years = network_table.years
    if cluster_labels is None:
        cluster_labels = [str(i) for i in range(num_clusters)]

    # create a list of valid columns across years
    columns = []
    for year in years:
        column = f'cluster_assignment_{year}'
        if column not in table.columns:
            raise ValueError(f"Expected column '{column}' not found in DataFrame.")
        else:
            columns.append(column)
    
    # create a list of dimensions (labelled vertical bars)
    dimensions = []
    for col in columns:
        values = table[col]
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
    if color_col not in table.columns:
        raise ValueError(f"Coloring year '{colour_index_year}' not found in the table.")
    else:
        color_values = table[color_col]
    
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

def visual_plot_clusters_area(
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
    table = network_table.table
    num_clusters = network_table.num_clusters
    if years is None:
        years = network_table.years

    # build count table: rows = years, columns = clusters
    cluster_counts = pd.DataFrame()
    for year in years:
        col = f"cluster_assignment_{year}"
        if col not in table.columns:
            raise ValueError(f"Expected column '{col}' not found in DataFrame.")
        counts = table[col].value_counts().sort_index()
        cluster_counts[year] = counts
    cluster_counts = cluster_counts.fillna(0).astype(int).T

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
            # cumulative[cluster] = y_vals
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

def visual_plot_line_means(
        network_table: ClusteredNetworkTable,
        selected_features: List[str],
        years: Optional[List[int]] = None,
        varnames: Optional[List[str]] = None,
        cluster_colours: Optional[dict[int, str]] = None,
        cluster_labels: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (900, 600),
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
            Which features (base_cols) to plot
        
        varnames (List[str] | None):
            The custom variable names to plot

        cluster_colours (dict[int, str] | None):
            A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
            colour map will be used. If a cluster number is not part of the dict, plotly's default
            colour map will be used for that cluster.

        cluster_labels (List[str] | None): 
            A custom list of cluster names. Default is Cluster 0, ..., Cluster n.

        figsize (Tuple[float, float]):
            Width and height of the overall figure in pixels.


    Returns:
        plotly.graph_objects.Figure:
            The composed line chart with subplots.
    """
    table = network_table.table
    num_clusters = network_table.num_clusters
    if years is None:
        years = network_table.years

    cluster_feature_means = core_cluster_means_by_year(table, years, selected_features)
    # 1) Pick a distinct color palette & map clusters → colors
    clusters = list(cluster_feature_means.index)

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
        subplot_titles=[v.replace('_', ' ').title() for v in selected_features] 
        if varnames is None else varnames,
    )

    # 3) Add one trace per cluster per subplot, forcing line+marker colors
    for col_idx, var in enumerate(selected_features, start=1):
        df_var = cluster_feature_means[var]
        x_vals = df_var.columns.astype(int)

        for i, cluster in enumerate(clusters):
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=df_var.loc[cluster],
                    mode='lines+markers',
                    name=f'Cluster {cluster}' if cluster_labels is None else cluster_labels[i],
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
            title_text=f"Mean {var.replace('_', ' ')}" if varnames is None else varnames[col_idx - 1],
            tickmode="array",
            tickvals=years,
            ticktext=[str(y) for y in years],
            row=1,
            col=col_idx
        )

    # 4) Final layout
    fig.update_layout(
        title="Mean Variable Values by Cluster Over Time",
        width=figsize[0],
        height=figsize[1],
        legend_title_text="Cluster",
        hovermode="x unified",
        template="plotly_white",
        margin=dict(t=80, b=40, l=60, r=20)
    )
    return fig

def visual_plot_bar_means(
        network_table: ClusteredNetworkTable,
        selected_features: List[str],
        years: Optional[List[int]] = None,
        varnames: Optional[List[str]] = None,
        cluster_colours: Optional[dict[int, str]] = None,
        cluster_labels: Optional[List[str]] = None,
        figsize: Tuple[float, float] = (900, 600),
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
    table = network_table.table
    num_clusters = network_table.num_clusters
    if years is None:
        years = network_table.years

    cluster_feature_means = core_cluster_means_by_year(table, years, selected_features)

    # 1) determine grid
    rows = math.ceil(len(years) / 2)
    # 2) prepare color map for clusters
    clusters = list(cluster_feature_means.index)

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
        for i, cluster in enumerate(clusters):
            fig.add_trace(
                go.Bar(
                    x=[v.replace('_', ' ') for v in selected_features] if varnames is None else varnames,
                    y=df_year.loc[cluster],
                    name=f'Cluster {cluster}' if cluster_labels is None else cluster_labels[i],
                    marker_color=colors[cluster],
                    showlegend=(idx == 0)  # legend only on first subplot
                ),
                row=r,
                col=c
            )

    # 6) final layout tweaks
    fig.update_layout(
        title="Mean Variable Values by Cluster Over Time",
        width=figsize[0],
        height=figsize[1],
        bargap=0.2,
        legend_title_text="Cluster",
        template="plotly_white"
    )
    # tighten subplot margins
    fig.update_layout(margin=dict(t=80, b=50, l=50, r=20))
    return fig

def visual_radar_chart_multiple_years(
        network_table: ClusteredNetworkTable,
        selected_cluster: int,
        selected_features: list,
        years: Optional[List[int]] = None,
        varnames: Optional[List[str]] = None,
        year_colours: Optional[dict[int, str]] = None,
        cluster_label: Optional[str] = None,
        figsize: Tuple[int, int] = (900, 600)
) -> go.Figure:
    """
    Create a radar (polar) chart of selected variables for a given cluster across years

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
    table = network_table.table
    num_clusters = network_table.num_clusters
    if years is None:
        years = network_table.years

    cluster_feature_means = core_cluster_means_by_year(table, years, selected_features)
    # Get the year level of the MultiIndex
    lvl1 = cluster_feature_means.columns.get_level_values(1)
    fig = go.Figure()

    colors = []
    if year_colours:
        for i in range(num_clusters):
            if i in year_colours:
                colors.append(year_colours[i])
            else:
                colors.append(plotly.colors.qualitative.Plotly[i])
    else:
        colors = plotly.colors.qualitative.Plotly
        if num_clusters > len(colors):
            colors = list(islice(cycle(colors), num_clusters))

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
            theta=result.index if varnames is None else varnames,
            fill='toself',
            name=str(year),
            line=dict(color=colors[idx]),
            marker=dict(color=colors[idx]),
        ))

        fig.update_layout(
            title=dict(
                text=f"Cluster {selected_cluster} Yearly Profile" 
                if cluster_label is None else f"{cluster_label} Yearly Profile",
                x=0.5, xanchor="center"
            ),
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            width=figsize[0],
            height=figsize[1],
            template="plotly_white",
        )

    return fig

def visual_radar_chart_multiple_clusters(
        network_table: ClusteredNetworkTable,
        selected_year: str,
        selected_features: List[str],
        clusters: Optional[List[int]] = None,
        varnames: Optional[List[str]] = None,
        cluster_colours: Optional[dict[int, str]] = None,
        cluster_labels: Optional[List[str]] = None,
        figsize: Optional[Tuple[int, int]] = (900, 600),
) -> go.Figure:
    """
    Draw a radar chart of the given variables for every cluster in `cluster_feature_means`,
    all on the same figure, for the specified year

    Parameters:
        network_table (ClusteredNetworkTable):
            A ClusteredNetworkTable containing the data.

        clusters (List[int]):
            Which clusters to plot. Default is every cluster.

        selected_year (str):
            The year for which features will be shown.

        selected_features (List[str]):
            Which variables to plot.

        cluster_colours (dict[int, str] | None):
            A dict mapping cluster numbers to their corresponding colours. If None, plotly's default
            colour map will be used. If a cluster number is not part of the dict, plotly's default
            colour map will be used for that cluster.

        cluster_labels (List[str] | None): 
            A custom list of cluster names. Default is Cluster 0, ..., Cluster n.

        figsize (Tuple[int, int] | None):
            Size of the figure in pixels. Default is (900, 600).

    Returns:
        go.Figure:
            The Plotly figure containing one polar trace per cluster.
    """
    table = network_table.table
    num_clusters = network_table.num_clusters

    cluster_feature_means = core_cluster_means_by_year(table, [selected_year], selected_features)
    # Extract the year-level values
    lvl1 = cluster_feature_means.columns.get_level_values(1)

    if clusters is None:
        clusters = [i for i in range(num_clusters)]
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
    for i, cluster_label in enumerate(df_year.index):
        if cluster_label in clusters:
            vals = df_year.loc[cluster_label, selected_features]
            fig.add_trace(go.Scatterpolar(
                r=vals.values,
                theta=selected_features if varnames is None else varnames,
                line=dict(color=colors[i]),
                marker=dict(color=colors[i]),
                fill='toself',
                name=f'Cluster {cluster_label}' if cluster_labels is None else cluster_labels[i]
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

def visual_plot_clusters_map(
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

        network_table (NetworkTable):
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
    # Load and merge geodata
    gdf = core_join_geometries(geofile_path, network_table, year)

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
    # Add the ID column for the specific year
    id_col = f'{network_table.id}_{year}'
    if id_col.lower() in gdf.columns:
        hover_data[id_col.lower()] = True
    else:
         print(f"Warning: ID column '{id_col}' not found in GeoDataFrame for year {year}")


    if cluster_labels:
        hover_data['cluster_name'] = False

    if label_dict:
        for feature in label_dict['F']:
            col_name = f'{feature}_{year}'
            if col_name in gdf.columns:
                hover_data[col_name] = True
            else:
                # Handle cases where the column might not exist for a given year
                print(f"Warning: Hover data column '{col_name}' not found in GeoDataFrame for year {year}")


    # create colours
    num_clusters = network_table.num_clusters

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
        title=f"Cluster Assignments in {year}",
        width=figsize[0],
        height=figsize[1],
        legend_title_text="Cluster"
    )

    return fig
