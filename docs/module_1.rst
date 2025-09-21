Module 1: Network Creation
==========================

NetworkTable class
------------------

.. code-block:: python 

    class NetworkTable():
        '''
        A table showing the network representation of census data. 
        Each feature present in the data is a column, and each possible path through the network is a row.
        '''
        
*Instance Variables:*
~~~~~~~~~~~~~~~~~~~~~~~

- ``table`` (``pandas.DataFrame``): The table, presented as a ``pandas`` DataFrame.
- ``years`` (List[str]): The census years present in the table.
- ``id`` (str): The unique geographical id used to distinguish geographical areas in the table.

*Methods:*
~~~~~~~~~~~

- ``modify_table``: Takes a new ``pandas`` DataFrame as an argument and sets ``table`` to the new DataFrame.

Module 1 Functions
-------------------

``preprocessing``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not necessary for network table creation, but you may optionally run this function yourself, for example
if you want details of the dataframe cleaning but not the network creation, or if you want to try out
different CRSs.
Returns a cleaned ``geopandas`` df of the input data. Uses parallel processing for very large (>100,000 rows) datasets.
Also adds a column for each year with calculated areas of each census tract in that year.
Note: Input data is assumed to have been passed through ``gpd.read_file()`` beforehand.

*Parameters:*

* ``data`` (GeoDataFrame):
    The census data to be analyzed with piccard.

* ``year`` (str):
    The year that the census data was collected.

* ``id`` (str):
    The name of the unique identifier that will be used to distinguish geographical areas.

* ``crs`` (CRS | None):
    A pythonic Coordinate Reference System manager that will be used to compute areas. Default is
    EPSG:3347, a consistent, equal-area CRS based on square metres. Can be many formats; see 
    https://pyproj4.github.io/pyproj/stable/api/crs/crs.html for more information.

* ``verbose`` (bool | None):
    Whether to issue print statements about the progress of network creation. Default is true.

*Returns:*

* ``gpd.GeoDataFrame``: the cleaned data

``create_network``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates a ``networkx`` network representation of the temporal connections present in ``census_dfs`` over ``years`` 
when each yearly geographic area has at most ``threshold`` percentage of overlap with its 
corresponding area(s) in the next year. Represents geographical areas as nodes, and temporal connections
as edges.

*Parameters:*

* ``census_dfs`` (List[gpd.GeoDataFrame]):
    A list of GeoDataFrames containing the census data to be turned into a network.

* ``years`` (List[str]):
    A list of years present in ``census_dfs`` over which the network representation will be created.
    Data from years not present in years will be ignored.

* ``id`` (str):
    The name of the unique identifier that will be used to distinguish geographical areas.

* ``crs`` (CRS | None):
    A pythonic Coordinate Reference System manager that will be used to compute areas. Default is
    EPSG:3347, a consistent, equal-area CRS based on square metres. Can be many formats; see 
    https://pyproj4.github.io/pyproj/stable/api/crs/crs.html for more information.

* ``threshold`` (float | None):
    The percentage of overlap (divided by 100)
    that geographic areas must meet or exceed in order to have a connection.
    Default is 0.05, or 5 percent.  

* ``verbose`` (bool | None):
    Whether to issue print statements about the progress of network creation. Default is true.

*Returns:*

* ``nx.Graph``: The ``networkx`` graph containing the nodes (geographical areas) and edges (geographical overlap)
          created in the new network representation.


``create_network_table``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creates a ``NetworkTable`` showing the network representation of the census data in ``census_dfs``. 
Each feature present in the data is a column, and each possible path through the network is a row.

*Parameters:*

* ``census_dfs`` (List[gpd.GeoDataFrame]):
    A list of GeoDataFrames containing the census data to be turned into a network.

* ``years`` (List[str]):
    A list of years present in ``census_dfs`` over which the network representation will be created.
    Data from years not present in years will be ignored.

* ``id`` (str):
    The name of the unique identifier that will be used to distinguish geographical areas.

* ``crs`` (CRS | None):
    A pythonic Coordinate Reference System manager that will be used to compute areas. Default is
    EPSG:3347, a consistent, equal-area CRS based on square metres. Can be many formats; see 
    https://pyproj4.github.io/pyproj/stable/api/crs/crs.html for more information.

* ``threshold`` (float | None):
    The percentage of overlap (divided by 100)
    that geographic areas must meet or exceed in order to have a connection.
    Default is 0.05, or 5 percent.  

* ``verbose`` (bool | None):
    Whether to issue print statements about the progress of network creation. Default is true.

*Returns:*

* ``NetworkTable``: the table.     