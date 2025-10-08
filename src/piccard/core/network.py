import warnings
from typing import Optional, List
import geopandas as gpd
from pyproj.crs import CRS
from shapely import buffer
from pyproj.exceptions import CRSError
import networkx as nx
import pandas as pd
import math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
try:
    import swifter
    SWIFTER_AVAILABLE = True
except ImportError:
    SWIFTER_AVAILABLE = False
import shapely

class NetworkTable():
    '''
    A table showing the network representation of census data. 
    Each feature present in the data is a column, and each possible path through the network is a row.
    '''
    def __init__(
        self,
        table: pd.DataFrame,
        years: list[str],
        id: str
    ):
        '''
        Constructor
        '''
        self.table = table
        self.years = years
        self.id = id
    
    def modify_table(
        self,
        new_table: pd.DataFrame
    ):
        '''
        Modifies the table.
        '''
        self.table = new_table

def core_preprocessing(
    data: gpd.GeoDataFrame, 
    year: str, 
    id: str,
    crs: Optional[CRS] = "EPSG:3347",
    verbose: Optional[bool] = True
) -> gpd.GeoDataFrame:
    '''
    Returns a cleaned geopandas df of the input data. Uses parallel processing for very large (>100,000 rows) datasets.
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
    process_data = data.copy()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)

        try:
            if process_data.crs != crs:
                process_data = process_data.to_crs(crs=crs)
        except CRSError:
            raise CRSError(f"{crs} is not a valid CRS")

        # Identify complex geometries
        def is_complex(g):
            try:
                return g.geom_type == 'Polygon' and len(g.exterior.coords) > 500
            except:
                return False
        
        # identify whether to run buffering in parallel
        use_swifter = SWIFTER_AVAILABLE and len(process_data) > 100_000 # more than 100,000 rows
        complexity_check = (
            process_data.geometry.swifter.apply(is_complex)
            if use_swifter else process_data.geometry.apply(is_complex)
        )

        if complexity_check.any():
            if verbose:
                print(f"PREPROCESSING: buffering {complexity_check.sum()} complex geometries with shapely.buffer()...")
            geom_to_buffer = process_data.loc[complexity_check, 'geometry']
            buffered = shapely.buffer(geom_to_buffer.values, -0.000001)
            process_data.loc[complexity_check, 'geometry'] = buffered

        process_data[f'area_{year}'] = process_data.geometry.area

        process_data[id] = f"{year}_" + process_data[id].astype(str)

    return process_data 

def core_create_network(
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
  preprocessed_dfs = [core_preprocessing(census_dfs[i], years[i], id, crs=crs, verbose=verbose) for i in range(len(census_dfs))]
  if verbose:
      print(f'Preprocessing complete')
  contained_cts = core_ct_containment(preprocessed_dfs, years, id, threshold, verbose)
  nodes = core_get_nodes(contained_cts, id, threshold)
  if verbose:
      print('All nodes found')
  attributes = core_get_attributes(nodes, census_dfs, years, id)
  if verbose:
      print('All attributes found')

  G = nx.from_pandas_edgelist(nodes, f'{id}_1', f'{id}_2')
  nx.set_node_attributes(G, attributes.set_index(id).to_dict('index'))
  if verbose:
      print('Graph created')

  return G

def core_create_network_table(
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
  num_years = len(years)
  num_joins = math.ceil(num_years/2)
  final_cols = [id + '_' + col_name for col_name in years]
  network_table = pd.DataFrame()
  drop_cols = final_cols[1:]

  preprocessed_dfs = [core_preprocessing(census_dfs[i], years[i], id, crs, verbose) for i in range(len(census_dfs))]
  if verbose:
      print('Preprocessing complete')
  contained_cts = core_ct_containment(preprocessed_dfs, years, id, threshold, verbose)
  nodes = core_get_nodes(contained_cts, id, threshold)
  if verbose:
      print('All nodes found')

  #all_paths returns a three item tuple
  all_paths = core_find_all_paths(nodes, num_joins, id)
  all_paths_df = all_paths[0]
  left_cols = all_paths[1]

  #Dividing all network paths into full paths and partial paths
  na_df = all_paths_df[all_paths_df.isnull().any(axis=1)]
  no_na_df = all_paths_df[~all_paths_df.isnull().any(axis=1)]

  full_paths = core_find_full_paths(no_na_df, final_cols)
  full_paths_list = full_paths.to_numpy().flatten()

  partial_paths = core_find_partial_paths(na_df, years, left_cols, final_cols, full_paths_list)
  if verbose:
      print('All possible paths through the graph found')

  network_table = pd.concat([full_paths, partial_paths])
  network_table = network_table[final_cols]
  network_table = network_table.T.drop_duplicates().T
  network_table = network_table.drop_duplicates(subset=drop_cols, keep='last')
  network_table.sort_values(by=final_cols[0], ignore_index=True)

  attributes = core_get_attributes(nodes, census_dfs, years, id)
  final_table = core_attach_attributes(network_table, attributes, years, final_cols, id)
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

  return NetworkTable(final_table, years, id)

# -----------------------Helper Functions-----------------------

def core_ct_containment(preprocessed_dfs, years, id='GeoUID', threshold=0.05, verbose=True):
    '''
    Returns a list of GeoDataFrames with tracts from one year
    that intersect tracts from the following year, filtered by threshold.
    Parallelizes if total data size > 20,000 rows.
    '''
    total_rows = sum(len(df) for df in preprocessed_dfs)
    use_parallel = total_rows > 20_000
    if verbose:
        print(f"CT_CONTAINMENT: total rows = {total_rows} | parallel = {use_parallel}")

    tasks = [
        (preprocessed_dfs[i], preprocessed_dfs[i + 1], years[i], years[i + 1], id, threshold)
        for i in range(len(years) - 1)
    ]

    if use_parallel:
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(core_process_year_pair_wrapped, tasks))
    else:
        results = [core_process_year_pair(*args) for args in tasks]

    return results

def core_get_nodes(contained_tracts_df, id, threshold=0.05):
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

def core_process_year_pair(df1, df2, year1, year2, id='GeoUID', threshold=0.05):
    """
    Helper for ct_containment.
    Computes intersecting census tracts between two years and returns a filtered GeoDataFrame.
    Applies spatial join and intersection, computes overlap area and percentage,
    and filters by a minimum threshold of shared area.
    """
    area1_col = f'area_{year1}'
    area2_col = f'area_{year2}'

    df1 = df1[[id, area1_col, 'geometry']].rename(columns={id: f'{id}_1'})
    df2 = df2[[id, area2_col, 'geometry']].rename(columns={id: f'{id}_2'})

    df1 = gpd.GeoDataFrame(df1, geometry='geometry', crs='EPSG:3347')
    df2 = gpd.GeoDataFrame(df2, geometry='geometry', crs='EPSG:3347')

    # Spatial join (geometry from df1 is preserved)
    joined = gpd.sjoin(df1, df2, predicate='intersects', how='inner')

    # Merge to get the right-hand geometry back in
    joined = joined.merge(df2[[f'{id}_2', area2_col, 'geometry']], on=f'{id}_2', how='left', suffixes=('', '_2'))

    # Vectorized intersection
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        joined['geometry'] = shapely.intersection(joined['geometry'], joined['geometry_2'])

    joined = gpd.GeoDataFrame(joined, geometry='geometry', crs='EPSG:3347')
    joined['area_intersection'] = joined.geometry.area

    pct_col = f'pct_{year2}_of_{year1}'
    joined[pct_col] = joined['area_intersection'] / joined[[area1_col, area2_col]].min(axis=1)

    joined = joined[joined[pct_col] >= threshold]

    kept_cols = [f'{id}_1', f'{id}_2', area1_col, area2_col, 'geometry', 'area_intersection', pct_col]
    return joined[kept_cols]

def core_assign_node_level(row, years, id):
  """
  Assigns the level of a node in the network based on its relative year in the
  network
  Example: All 2021 nodes are in level 3 in a graph with years 2011, 2016, 2021
  """
  for i in range(len(years)):
    if row[id].startswith(str(years[i])):
      return i+1
    
def core_get_attributes(nodes, census_dfs, years, id):
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
  all_attr['network_level'] = all_attr.apply(lambda x: core_assign_node_level(x, years, id), axis=1)
  return all_attr

def core_join_geometries(
    geofile_path: str,
    network_table: NetworkTable,
    year: str,
) -> gpd.GeoDataFrame:
    """
    Joins spatial data from a geographical data file with attribute data from a network table
    using a shared geographic identifier.

    This function is designed for researchers who work with pre-processed network tables
    (containing cluster assignments, IDs, etc.) and separately downloaded spatial files
    (like Canadian census tract GeoJSONs).

    Parameters:
        geofile_path (str):
            File path to the geographical data file for the specified year. Can be anything readable by geopandas.

        network_table (NetworkTable):
            NetworkTable containing attribute and cluster assignment data, including unique
            geographic identifiers for each region.

        year (str):
            The census year to match ID and cluster columns (e.g., '2016').

    Returns:
        gpd.GeoDataFrame:
            A merged GeoDataFrame containing geometry from the GeoJSON file and attribute
            data (e.g., cluster assignments) from the network table. Only valid, non-empty
            geometries are retained.
    """
    table = network_table.table
    network_table_id_col = network_table.id
    # Validate input column name
    geoid_col = f"{network_table_id_col.lower()}_{year}"
    if geoid_col not in table.columns:
        raise ValueError(f"Expected column '{geoid_col}' not found in table.")

    # Read the GeoJSON file into a GeoDataFrame
    gdf = gpd.read_file(geofile_path)

    # Prepare a clean copy of the network table and standardize the ID format
    table_copy = table.copy(deep=True)
    table_copy[geoid_col] = table_copy[geoid_col].astype(str).str.replace(r'^\d{4}_', '', regex=True)

    # Merge the GeoDataFrame with the network table using the geographic ID
    merged_gdf = gdf.merge(table_copy, left_on=network_table_id_col, right_on=geoid_col)

    # Remove empty or invalid geometries
    merged_gdf = merged_gdf[~merged_gdf.is_empty & merged_gdf.geometry.notnull()]

    return merged_gdf

def core_find_partial_paths(partial_paths_df, years, left_cols, final_cols, exclude_nodes):
  '''
  Return all partial paths present in input data.
  Note: Define a partial path as a path in the network where the starting and
        ending nodes are of any year (i.e., not a full path).
  '''

  all_partial_paths = partial_paths_df.T.drop_duplicates().T
  all_partial_paths = all_partial_paths[~all_partial_paths[left_cols[0]].isin(exclude_nodes)]

  first_year_partials = core_first_year_partial_paths(all_partial_paths, years, final_cols)
  unique_partials = core_unique_partial_paths(all_partial_paths, years, left_cols, final_cols)
  all_partials = pd.concat([unique_partials, first_year_partials])

  return all_partials

def core_process_year_pair_wrapped(args):
    """
    Helper for ct_containment.
    Wrapper for process_year_pair to enable parallel execution with ProcessPoolExecutor.
    Unpacks arguments passed as a single tuple.
    """
    return core_process_year_pair(*args)

def core_find_all_paths(nodes_df, num_joins, id):
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

def core_find_full_paths(full_paths_df, final_cols):
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

def core_first_year_partial_paths(all_partial_paths, years, final_cols):
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

def core_unique_partial_paths(all_partial_paths, years, left_cols, final_cols):
    '''
    Return all unique partial paths between two consecutive input years.
    Note: Define a partial path as a path in the network where the starting and
        ending nodes are of any year (i.e., not a full path).
    '''
    num_years = len(years)
    unique_partials = pd.DataFrame()

    for i in range(1, num_years):
        curr_year = years[i] + '_'
        prev_year = years[i - 1] + '_'

        curr_year_mask = all_partial_paths.iloc[:, 0].str.startswith(curr_year)
        prev_year_mask = all_partial_paths.iloc[:, 0].str.startswith(prev_year)

        curr_year_partials = all_partial_paths[curr_year_mask]
        prev_year_partials = all_partial_paths[prev_year_mask]

        curr_year_mask = ~curr_year_partials[left_cols[0]].isin(prev_year_partials)
        curr_year_unique = curr_year_partials[curr_year_mask]

        curr_year_unique = curr_year_unique.dropna(axis=1).T.drop_duplicates().T

        # Prepend NaN for missing earlier years
        for k in range(i):
            curr_year_unique.insert(0, final_cols[k], np.nan)

        # Fix width: pad to match final_cols if needed
        if not curr_year_unique.empty:
            current_cols = curr_year_unique.shape[1]
            missing_cols = len(final_cols) - current_cols

            for j in range(missing_cols):
                curr_year_unique.insert(curr_year_unique.shape[1], f'_pad_{j}', np.nan)

            # Trim if too long
            curr_year_unique = curr_year_unique.iloc[:, :len(final_cols)]

            # Assign correct column names
            curr_year_unique.columns = final_cols

        unique_partials = pd.concat([unique_partials, curr_year_unique], ignore_index=True)

    return unique_partials


def core_attach_attributes(network_table, attributes, years, final_cols, id):
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

# Public API aliases for piccard (must be after all function definitions)
create_network = core_create_network
create_network_table = core_create_network_table
