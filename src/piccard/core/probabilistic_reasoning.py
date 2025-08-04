import pandas as pd
import geopandas as gpd
from typing import Optional, List
from p_frame import PDataFrame
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.network import NetworkTable

def core_prob_reasoning_networks(
    network_table_1: NetworkTable | pd.DataFrame | gpd.GeoDataFrame, 
    network_table_2: NetworkTable | pd.DataFrame | gpd.GeoDataFrame, 
    independent_vars_1: List[str], 
    independent_vars_2: List[str], 
    dependent_vars_1: List[str], 
    dependent_vars_2: List[str], 
    mismatches: Optional[dict[str, str]] = None, 
    modify_tables: Optional[bool] = False
) -> PDataFrame:
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

        modify_tables (bool | None):
            A boolean indicating whether to modify table values to fix the mismatch(es), if applicable.
            Default is False.

    Returns:
        PDataFrame:
            The result of joining the two probabilistic models of network tables.

    TODO: Finish handling mismatches by modifying network tables
    '''
    if type(network_table_1) == gpd.GeoDataFrame or type(network_table_1) == pd.DataFrame:
        table_1 = network_table_1
    else:
        table_1 = network_table_1.table
    if type(network_table_2) == gpd.GeoDataFrame or type(network_table_2) == pd.DataFrame:
        table_2 = network_table_2
    else:
        table_2 = network_table_2.table
    if set(independent_vars_1).union(set(independent_vars_2)) != set(independent_vars_1):
        raise ValueError("Please make sure independent_vars_2 contains only variables that are also in independent_vars_1.")
    all_vars_1 = independent_vars_1 + dependent_vars_1
    all_vars_2 = independent_vars_2 + dependent_vars_2
    pdf_1 = PDataFrame(independent_vars = independent_vars_1, data = table_1[all_vars_1])
    pdf_2 = PDataFrame(independent_vars = independent_vars_2, data = table_2[all_vars_2])
    # modify network tables according to mismatches if necessary
    joined_pdf = pdf_1.pjoin(pdf_2, mismatches=mismatches)
    if mismatches is not None and modify_tables:
        if any(mismatch not in table_1.columns or mismatch not in table_2.columns for mismatch in mismatches.keys()):
            raise ValueError("Please make sure mismatch keys correspond to columns in network tables.")
        for var in mismatches.keys():
            cpd = joined_pdf.bayes_net.get_cpds(var)
            # handle mismatches here!!
            # if cpd is not None:
                # fixed_states = cpd.state_names[cpd.variable]
    return joined_pdf



def core_prob_reasoning_years(
    network_table: NetworkTable,  
    year_1: str,
    year_2: str,
    independent_vars_1: List[str], 
    independent_vars_2: List[str], 
    dependent_vars_1: List[str], 
    dependent_vars_2: List[str], 
    mismatches: Optional[dict[str, str]] = None, 
    modify_tables: Optional[bool] = False
) -> PDataFrame:
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
            The network table

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

        modify_tables (bool | None):
            A boolean indicating whether to modify table values to fix the mismatch(es), if applicable.
            Default is False.

    Returns:
        PDataFrame:
            The result of joining the two probabilistic models of network tables.

    TODO: Finish handling mismatches by modifying network tables
    '''
    if any(var[-4:] != year_1 for var in list(set(independent_vars_1) | set(dependent_vars_1))):
        raise ValueError("Please make sure all variables in independent_vars_1 and dependent_vars_1 end in year_1.")
    if any(var[-4:] != year_2 for var in list(set(independent_vars_2) | set(dependent_vars_2))):
        raise ValueError("Please make sure all variables in independent_vars_2 and dependent_vars_2 end in year_2.")
    if set([var[:-4] for var in independent_vars_1]).union(set([var[:-4] for var in independent_vars_2])) != set([var[:-4] for var in independent_vars_1]):
        raise ValueError("Please make sure independent_vars_2 (excluding years) contains only variables that are also in independent_vars_1 (excluding years).")
    all_vars_1 = independent_vars_1 + dependent_vars_1
    all_vars_2 = independent_vars_2 + dependent_vars_2
    # removing year from column names
    table = network_table.table
    table_1 = table[all_vars_1]
    for var in all_vars_1:
        table_1[var[:-5]] = table_1[var]
        table_1.drop(columns=[f'{var}'], inplace=True)
    table_2 = table[all_vars_2]
    for var in all_vars_2:
        table_2[var[:-5]] = table_2[var]
        table_2.drop(columns=[f'{var}'], inplace=True)
    pdf_1 = PDataFrame(independent_vars = [var[:-5] for var in independent_vars_1], data = table_1)
    pdf_2 = PDataFrame(independent_vars = [var[:-5] for var in independent_vars_2], data = table_2)
    # modify network tables according to mismatches if necessary
    joined_pdf = pdf_1.pjoin(pdf_2, mismatches=mismatches)
    if mismatches is not None and modify_tables:
        if any(mismatch not in table.columns for mismatch in mismatches.keys()):
            raise ValueError("Please make sure mismatch keys correspond to columns in network tables.")
        for var in mismatches.keys():
            cpd = joined_pdf.bayes_net.get_cpds(var)
            # handle mismatches here!!
            # if cpd is not None:
                # fixed_states = cpd.state_names[cpd.variable]
    return joined_pdf
