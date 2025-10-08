# Optional dependency: tscluster
try:
    import tscluster
    TSCLUSTER_AVAILABLE = True
except ImportError:
    TSCLUSTER_AVAILABLE = False

import numpy as np
import pandas as pd
from typing import Optional, Any, Tuple, Union, List
import networkx as nx
from tscluster.opttscluster import OptTSCluster
from tscluster.greedytscluster import GreedyTSCluster
from tscluster.preprocessing.utils import load_data, tnf_to_ntf, ntf_to_tnf

from .network import NetworkTable

class ClusteredNetworkTable(NetworkTable):
    '''
    A table showing the network representation of census data and the cluster assignments in the network.
    Each feature present in the data (including the cluster assignment for each year) is a column, 
    and each possible path through the network is a row.
    '''
    def __init__(
        self,
        table: pd.DataFrame,
        years: list[str],
        id: str,
        num_clusters: int,
        tsc: Union[OptTSCluster, GreedyTSCluster],
        arr: np.ndarray[np.float64], 
        label_dict: dict[str, Any]
    ):
        '''
        Constructor
        '''
        super().__init__(table, years, id) 
        self.num_clusters = num_clusters
        self.tsc = tsc
        self.arr = arr
        self.label_dict = label_dict

    def modify_label_dict(
        self,
        new_label_dict: dict[str, Any]
    ):
        '''
        Modifies the label dictionary.
        '''
        self.label_dict = new_label_dict

def core_clustering_prep(
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
    table = network_table.table
    years = network_table.years

    # default to considering all features
    if cols == []:
        cols = table.columns.to_list()

    # Filter columns
    filtered_cols = core_filter_columns(network_table, cols)
    network_table = filtered_cols[2]
    table = network_table.table

    # Extract features for each year and add them to a 2D array representing that year. 
    # Then add that array to a list of arrays representing the 3D array used for tscluster.
    list_of_arrays = []
    for year in years:
        year_statistics = table[[col for col in filtered_cols[0] if year in col]].to_numpy()
        list_of_arrays.append(year_statistics)
    
    # Filter out entities whose features are entirely NaN
    # Run load_data now so we get access to variables necessary for tnf_to_ntf
    list_of_arrays = load_data(list_of_arrays)[0]
    ntf_list_of_arrays = tnf_to_ntf(list_of_arrays)
    # Interpolate remaining nan values for clustering
    clean_entities = []
    index = -1
    for entity in ntf_list_of_arrays:
        index += 1
        transposed_entity = entity.T
        valid = True
        for i, row in enumerate(transposed_entity):
            nans = np.isnan(row)
            x = np.arange(len(row))
            if np.all(nans):
                valid = False  # entire row is NaN, skip entity
                table = table.drop(index=index)
                index -= 1
                break
            elif np.any(nans):
                transposed_entity[i] = np.interp(x, x[~nans], row[~nans])
        if valid:
            clean_entities.append(transposed_entity.T)

    # Convert to 3D numpy array
    if len(clean_entities) == 0:
        raise ValueError("All entities were invalid (contained all-NaN rows). No data left for clustering.")
    try:
        list_of_arrays = np.stack(clean_entities)  # shape: (N entities, T time, F features)
    except ValueError as e:
        raise ValueError("Entities have inconsistent shapes and cannot be stacked.") from e
    
    list_of_arrays = ntf_to_tnf(list_of_arrays)
                
    # Return the final numpy array and create a corresponding label dictionary.
    # This can then be preprocessed using tscluster's scalers.
    label_dict = {
    'T': years,
    'N': list(range(list_of_arrays.shape[1])),
    'F': filtered_cols[1]
    }

    network_table.modify_table(table)

    return (list_of_arrays, label_dict, network_table)

def core_cluster(
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

    Users can choose to only input the network table, in which case core_clustering_prep will be run for them with the default columns,
    or they can choose to run core_clustering_prep on their own and then have the option to apply one or both of the
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
            pc.core_clustering_prep() with the default columns. See the pc.core_clustering_prep() documentation for why we DO NOT
            recommend leaving this blank.
        
        label_dict (dict[str, Any] | None): 
            the label dictionary corresponding to the data array. See 'arr'.

    Returns:
        ClusteredNetworkTable: 
            The clustered network table.
    '''
    # Get the data into the correct format. See the documentation for core_clustering_prep
    if arr is None and label_dict is None:
        arr, label_dict = core_clustering_prep(network_table)
    
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

    table = network_table.table
    years = network_table.years

    # Add cluster assignments to network table
    cluster_assignments_table = tsc.get_named_labels(label_dict=label_dict)
    for year in years:
        table[f'cluster_assignment_{year}'] = list(cluster_assignments_table[year])

    # Add cluster assignments to graph nodes
    nodes_list = list(G.nodes(data=True))
    for node in nodes_list:
            year = node[0][:4]
            cluster = table.loc[table[f'geouid_{year}'] == node[0]]
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
                        old_cluster_distance += (abs(int(node[1][label_dict['F'][i]]) - int(old_dict[i])))
                        new_cluster_distance += (abs(int(node[1][label_dict['F'][i]]) - int(dict[i])))
                    if old_cluster_distance < new_cluster_distance:
                        cluster = node[1]['cluster_assignment']
                node[1]['cluster_assignment'] = cluster
            elif 'cluster_assignment' not in node[1]:
                node[1]['cluster_assignment'] = np.nan
    
    network_table.modify_table(table)
    
    return ClusteredNetworkTable(network_table.table, network_table.years, network_table.id, num_clusters, tsc, arr, label_dict)

# -----------------------Helper Functions-----------------------

def core_filter_columns(
    network_table: NetworkTable, 
    cols: Optional[list[str]]=[]
    ) -> Tuple[List[str], List[str], NetworkTable]:
    '''
    Checks that the list of columns with data to be clustered is valid in the following ways:
    - Makes sure all the data in the columns are numerical or nan
    - Makes sure there is a version of each column for every year

    Parameters:
        network_table (NetworkTable): 
            The result of pc.create_network_table().
          
        cols (list[str] | None): A list of the names of network table columns that should be considered in
            the clustering algorithm. If none, every numerical feature will be considered. Leaving it none is
            not recommended as many numerical features, such as network level, have little bearing on the data.
    
    Returns:
        (Tuple[List[str], List[str]], NetworkTable):
            a tuple of the final filtered list of columns and the column labels that will
            be used for the label dictionary. Also returns the possibly modified network table.
    '''
    table = network_table.table
    years = network_table.years
    # Only add features that are numerical or nan. the user should have selected accordingly
    # but this is a sanity check
    col_list = []

    for col in cols:
        if col in table.columns.to_list():
            non_numerical_val_in_col = False
            for entry in table[col]:
                if isinstance(entry, str) and '_' in entry: # make sure underscores don't get converted to numbers
                    non_numerical_val_in_col = True
                    break
                try:
                    int(entry)  # see if it is either an int or an int masquerading as a string
                except (ValueError, TypeError):
                    try:
                        float(entry)  # see if it is either a float or a float masquerading as a string
                    except (ValueError, TypeError):
                        if entry != 'NaN' and entry != 'nan': # see if it is nan
                            non_numerical_val_in_col = True
                            break
                except OverflowError: # set infinity to nan
                    index = table.loc[(table == entry).any(axis=1)].index[0]
                    table[col][index] = np.nan                       
            if not non_numerical_val_in_col:
                col_list.append(col)

    network_table.modify_table(table)

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

    return (cols_in_every_year, features_list, network_table)

def core_cluster_means_by_year(
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