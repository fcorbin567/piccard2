import random
from typing import Optional, List
import networkx as nx
import plotly.graph_objects as go
from ..core.network import NetworkTable

def visual_plot_subnetwork(
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
    table = network_table.table
    # create valid list of years
    all_years = network_table.years
    if years is None:
        years = all_years
    else:
        years = [year for year in years if int(year) in all_years]
        if len(years) == 0:
            years = all_years
    all_years = [str(year) for year in all_years]
    years = [str(year) for year in years]

    # organize node names by year and network table path number
    paths_for_each_year = [list(table[table.columns[i]]) for i in range(len(years))]

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

def visual_plot_num_areas(
    network_table: NetworkTable, 
    years: Optional[List[str]] = None,
) -> go.Figure:
    '''
    Plots the number of geographical areas across a subset of census years in the data.
    Note: Assumes the first column in the dataframe contains the ID.

    Parameters:
        network_table (NetworkTable):
            The result of pc.create_network_table().

        years (List[str] | None):
            A list of years to show in the subnetwork. Default is all census years present in the data.

    Returns:
        go.Figure:
            The plot of the number of geographical areas.
    '''
    table = network_table.table
    id_label = network_table.id
    if years is None:
        years = network_table.years

    ct_per_year = []
    num_years = len(years)
    for i in range(num_years):
        ids_list = list(table[table.columns[i]])
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
