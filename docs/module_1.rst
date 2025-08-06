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