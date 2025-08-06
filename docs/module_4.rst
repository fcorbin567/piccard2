Module 4: VariableLinker
==================

Overview
--------

VariableLinker is a Python framework designed for visualizing the links between census variables across multiple years. It provides multiple approaches for matching census variables between different years and creates hierarchical tree visualizations that show how these variables are connected.

Key Features:

* **Multiple Matching Algorithms**: Jaccard similarity and sentence transformers
* **Hierarchical Visualization**: Creates tree structures showing the parent-child relationships in census data
* **Colour-coded Results**: Visual indicators for data consistency across years

Use Cases:

* Census data harmonization across multiple years
* Tracking changes in census variables over time
* Visualizing data consistency and evolution


Installation and Setup
----------------------

Prerequisites

.. code-block:: bash

  pip install -r requirements.txt


Importing VariableLinker

.. code-block:: python

  import sys
  import os


  # Add the src/piccard directory to Python path
  
  current_dir = os.getcwd()
  src_path = os.path.join(current_dir, '..', 'src', 'piccard')
  sys.path.append(src_path)

  from variable_linker import VariableLinker



Core Concepts
-------------

1. Census Metadata Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

VariableLinker works with census metadata JSON files that contain:

* **Vector identifiers**: Unique codes for census variables
* **Descriptions**: Human-readable descriptions of census variables
* **Types**: Categories like "Total", "Male", "Female"
* **Details**: Additional contextual information

2. Matching Process
~~~~~~~~~~~~~~~~~~~~

The framework performs two-pass matching:

* **Exact Match**: Find identical descriptions across years
* **Similarity Match**: Use similarity algorithms for inexact matches

3. Tree Visualization
~~~~~~~~~~~~~~~~~~~~~

* **Nodes**: Represent census variables
* **Edges**: Show parent-child relationships
* **Colours**: Indicate consistency across years

  * Grey: Source year only
  * Salmon: Matches in 1 other year
  * Yellow: Matches in 2 other years
  * Light green: Matches in 3+ other years


VariableLinker Class Reference
------------------------------

Class Overview
~~~~~~~~~~~~~~

.. code-block:: python

  class VariableLinker:
      """
      A class for processing census metadata and creating tree visualizations.
      
      This class provides functionality for:
      - Preprocessing census metadata from JSON files
      - Computing similarity between census descriptions using various methods
      - Matching descriptions across different census years
      - Building hierarchical tree visualizations with colour-coding
      """

Static Methods
~~~~~~~~~~~~~~~

.. list-table:: VariableLinker Static Methods

   :header-rows: 1
   :widths: 20 20 20 20

   * - Method
     - Parameters
     - Returns
     - Description
   * - ``preprocess_census_metadata``
     - ``path, type_filter``
     - ``pd.DataFrame``
     - Preprocess census metadata
   * - ``jaccard_similarity``
     - ``sentence1, sentence2``
     - ``float``
     - Compute Jaccard similarity
   * - ``process_discription_text``
     - ``text``
     - ``set``
     - Process and tokenize text
   * - ``normalize_ranges``
     - ``text``
     - ``str``
     - Normalize numeric ranges
   * - ``match_descriptions_jaccard``
     - ``source_df, compare_df, threshold``
     - ``pd.DataFrame``
     - Jaccard-based matching
   * - ``match_descriptions_transformer``
     - ``source_df, compare_df, threshold, model``
     - ``pd.DataFrame``
     - Transformer-based matching
   * - ``match_descriptions_details_sentence_transformer``
     - ``source_df, compare_df, threshold, model``
     - ``pd.DataFrame``
     - Advanced transformer matching
   * - ``merge_mappings``
     - ``map_descriptions, *mappings_dfs``
     - ``pd.DataFrame``
     - Merge multiple mappings
   * - ``build_tree``
     - ``source_data, merged_df, tree_name, path``
     - ``Digraph``
     - Build tree visualization
   * - ``parse_tree_to_dict``
     - ``filepath``
     - ``Dict``
     - Parse tree file to dictionary
   * - ``extract_parent_child_relationships``
     - ``filepath``
     - ``Dict[str, List[str]]``
     - Extract parent-child relationships
   * - ``predict_parent_nodes``
     - ``tree_dict, parent_child_relationships, target_years``
     - ``Dict[str, List[str]]``
     - Predict missing parent nodes


``preprocess_census_metadata(path, type_filter="Total")``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Preprocesses census metadata from JSON files.

*Parameters:*

- ``path`` (str): Path to the JSON file containing census metadata
- ``type_filter`` (str): Type of records to filter for (default: "Total")

*Returns:*

- ``pd.DataFrame``: Preprocessed DataFrame with columns ['vector', 'type', 'description', ...]

*Example:*

.. code-block:: python

    data_2021 = VariableLinker.preprocess_census_metadata("census_ca21_full_metadata.json")



``jaccard_similarity(sentence1, sentence2)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computes Jaccard similarity between two census descriptions.

*Parameters:*

- ``sentence1`` (str): First census description
- ``sentence2`` (str): Second census description

*Returns:*

- ``float``: Jaccard similarity score between 0.0 and 1.0

``process_discription_text(text)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Processes and tokenizes census text for similarity comparison.

*Parameters:*

- ``text`` (str): Raw census description text

*Returns:*

- ``set``: Set of processed tokens (words and numbers, excluding stopwords)

``normalize_ranges(text)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Normalizes numeric ranges in text for consistent processing.

*Parameters:*

- ``text`` (str): Text containing potential numeric ranges

*Returns:*

- ``str``: Text with normalized numeric ranges

``parse_tree_to_dict(filepath)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parses a Graphviz tree file into a dictionary structure.

*Parameters:*

- ``filepath`` (str): Path to the Graphviz tree file

*Returns:*

- ``Dict``: Dictionary mapping node IDs to their information including descriptions, year mappings, and colours

*Example:*

.. code-block:: python

  tree_dict = VariableLinker.parse_tree_to_dict("my_tree")


``extract_parent_child_relationships(filepath)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extracts parent-child relationships from tree file edges.

*Parameters:*

- ``filepath`` (str): Path to the tree file (Graphviz format)

*Returns:*

- ``Dict[str, List[str]]``: Dictionary mapping parent nodes to their children

*Example:*

.. code-block:: python

  relationships = VariableLinker.extract_parent_child_relationships("my_tree")


``predict_parent_nodes(tree_dict, parent_child_relationships, target_years)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Predicts parent nodes in other years using the additive property.

*Parameters:*

- ``tree_dict`` (Dict): Parsed tree dictionary with node info and year mappings
- ``parent_child_relationships`` (Dict[str, List[str]]): Parent to children mapping
- ``target_years`` (List[str]): Years to predict parents for (default: ['2016', '2011', '2006'])

*Returns:*

- ``Dict[str, List[str]]``: Dictionary mapping parent nodes to years in which they can be predicted

*Example:*

.. code-block:: python

    predictions = VariableLinker.predict_parent_nodes(tree_dict, relationships, ['2016', '2011'])



Matching Approaches
--------------------

1. Jaccard Similarity Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``match_descriptions_jaccard()``

Uses token-based similarity to match descriptions across years.

**Advantages:**

* Good for exact and near-exact matches
* Language-agnostic

**Disadvantages:**

* May miss semantic similarities
* Sensitive to phrasing

**Usage:**

.. code-block:: python

    jaccard_mapping = VariableLinker.match_descriptions_jaccard(
        source_df=data_2021, 
        compare_df=data_2016, 
        similarity_threshold=0.9
    )


2. Sentence Transformer Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``match_descriptions_transformer()``

Uses pre-trained sentence transformers for semantic similarity matching.

**Advantages:**

* Captures semantic meaning
* Better for paraphrased descriptions
* Robust to word variations
* Faster than Jaccard since it uses vectorization

**Disadvantages:**

* Limited ability to process numeric values and ranges in text descriptions

**Usage:**

.. code-block:: python

    transformer_mapping = VariableLinker.match_descriptions_transformer(
        source_df=data_2021,
        compare_df=data_2016,
        similarity_threshold=0.9,
        model_name='all-mpnet-base-v2'
    )


3. Advanced Sentence Transformer Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``match_descriptions_details_sentence_transformer()``

Enhanced version of sentence transformer that uses details for breaking ties when multiple exact matches are found.

**Advantages:**

* Attempts better disambiguation using details field
* More sophisticated exact matching strategy

**Disadvantages**

* Performance evaluation indicates higher error rate than basic transformer
* Higher computational complexity without performance benefit

**Usage:**

.. code-block:: python

  advanced_mapping = VariableLinker.match_descriptions_details_sentence_transformer(
      source_df=data_2021,
      compare_df=data_2016,
      similarity_threshold=0.9
  )


4. Multithreaded Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method:** ``match_descriptions_multithreaded()`` (from multithreaded_mapping.py)

Jaccard similarity approach with multithreaded execution for enhanced performance on large datasets.

**Advantages:**

* Parallel processing for similarity matching phase
* Configurable number of worker threads (default: 4)
* Thread-safe operations for similarity matching

**Usage:**

.. code-block:: python

  from multithreaded_mapping import match_descriptions_multithreaded

  multithreaded_mapping = match_descriptions_multithreaded(
      source_df=data_2021,
      compare_df=data_2016,
      similarity_threshold=0.9,
      max_workers=8
  )

Workflow Examples
------------------

Basic Workflow
~~~~~~~~~~~~~~~~

.. code-block:: python

  # 1. Load and preprocess data
  data_2021 = VariableLinker.preprocess_census_metadata("census_ca21_full_metadata.json")
  data_2016 = VariableLinker.preprocess_census_metadata("census_ca16_full_metadata.json")


  # 2. Perform matching
  mapping_21_16 = VariableLinker.match_descriptions_jaccard(
      source_df=data_2021, 
      compare_df=data_2016, 
      similarity_threshold=0.9
  )


  # 3. Merge mappings
  merged_df = VariableLinker.merge_mappings(data_2021, mapping_21_16)


  # 4. Build visualization
  tree = VariableLinker.build_tree(data_2021, merged_df, "my_tree", "output_path")


Multi-Year Workflow
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python


  # Load data for multiple years
  data_2006 = VariableLinker.preprocess_census_metadata("census_ca06_full_metadata.json")
  data_2011 = VariableLinker.preprocess_census_metadata("census_ca11_full_metadata.json")
  data_2016 = VariableLinker.preprocess_census_metadata("census_ca16_full_metadata.json")
  data_2021 = VariableLinker.preprocess_census_metadata("census_ca21_full_metadata.json")


  # Match against 2021 (latest year)
  mapping_21_16 = VariableLinker.match_descriptions_jaccard(data_2021, data_2016, 0.9)
  mapping_21_11 = VariableLinker.match_descriptions_jaccard(data_2021, data_2011, 0.9)
  mapping_21_06 = VariableLinker.match_descriptions_jaccard(data_2021, data_2006, 0.9)


  # Merge all mappings
  merged_df = VariableLinker.merge_mappings(data_2021, mapping_21_16, mapping_21_11, mapping_21_06)


  # Build comprehensive tree
  tree = VariableLinker.build_tree(data_2021, merged_df, "multi_year_tree", "trees/")


Comparison of Approaches
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python


  # Jaccard approach
  jaccard_mapping = VariableLinker.match_descriptions_jaccard(data_2021, data_2016, 0.9)
  jaccard_merged = VariableLinker.merge_mappings(data_2021, jaccard_mapping)
  jaccard_tree = VariableLinker.build_tree(data_2021, jaccard_merged, "jaccard_tree", "trees/")


  # Transformer approach
  transformer_mapping = VariableLinker.match_descriptions_transformer(data_2021, data_2016, 0.9)
  transformer_merged = VariableLinker.merge_mappings(data_2021, transformer_mapping)
  transformer_tree = VariableLinker.build_tree(data_2021, transformer_merged, "transformer_tree", "trees/")


  # Multithreaded approach
  from multithreaded_mapping import match_descriptions_multithreaded
  multithreaded_mapping = match_descriptions_multithreaded(data_2021, data_2016, 0.9, 8)
  multithreaded_merged = VariableLinker.merge_mappings(data_2021, multithreaded_mapping)
  multithreaded_tree = VariableLinker.build_tree(data_2021, multithreaded_merged, "multithreaded_tree", "trees/")



Advanced Features
------------------

Custom Similarity Thresholds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different thresholds can be used for different types of data:

.. code-block:: python


  # Strict matching for critical variables
  critical_mapping = VariableLinker.match_descriptions_jaccard(data_2021, data_2016, 0.95)


  # Relaxed matching for exploratory analysis
  exploratory_mapping = VariableLinker.match_descriptions_jaccard(data_2021, data_2016, 0.7)



Model Selection for Transformers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python


  # Use different transformer models
  mapping_mini = VariableLinker.match_descriptions_transformer(
      data_2021, data_2016, 0.9, 'all-MiniLM-L6-v2'
  )
  mapping_mpnet = VariableLinker.match_descriptions_transformer(
      data_2021, data_2016, 0.9, 'all-mpnet-base-v2'
  )



Tree Analysis and Prediction
-----------------------------

Overview
~~~~~~~~~~

VariableLinker provides advanced functionality for analyzing existing tree structures and predicting missing parent nodes based on the additive property of census data.

Key Concepts
~~~~~~~~~~~~

* Additive Property

  In census data, parent variables often represent the sum of their child variables:

  Parent_Value = Sum(Child_Values)

  This property allows us to predict parent nodes in years where they don't exist or did not get matched, as long as all their children are available in those years.

* Tree Parsing

  The framework can parse existing Graphviz tree files to extract:

  * Node descriptions and metadata
  * Year-specific vector mappings
  * Parent-child relationships
  * Colour-coding information

Workflow for Tree Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python


  # 1. Parse existing tree file
  tree_dict = VariableLinker.parse_tree_to_dict("existing_tree.gv")


  # 2. Extract parent-child relationships
  relationships = VariableLinker.extract_parent_child_relationships("existing_tree.gv")


  # 3. Predict missing parent nodes
  predictions = VariableLinker.predict_parent_nodes(
      tree_dict=tree_dict,
      parent_child_relationships=relationships,
      target_years=['2016', '2011', '2006']
  )


  # 4. Analyze predictions
  for parent_node, predictable_years in predictions.items():
      print(f"Parent '{parent_node}' can be predicted in years: {predictable_years}")


Complete Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python


  # Load and process census data
  data_2021 = VariableLinker.preprocess_census_metadata("census_ca21_full_metadata.json")
  data_2016 = VariableLinker.preprocess_census_metadata("census_ca16_full_metadata.json")


  # Create initial tree
  mapping_21_16 = VariableLinker.match_descriptions_jaccard(data_2021, data_2016, 0.9)
  merged_df = VariableLinker.merge_mappings(data_2021, mapping_21_16)
  tree = VariableLinker.build_tree(data_2021, merged_df, "analysis_tree", "trees/")


  # Analyze the created tree
  tree_dict = VariableLinker.parse_tree_to_dict("trees/analysis_tree")
  relationships = VariableLinker.extract_parent_child_relationships("trees/analysis_tree")


  # Predict missing parents
  predictions = VariableLinker.predict_parent_nodes(tree_dict, relationships)


  # Generate report
  print("=== Tree Analysis Report ===")
  print(f"Total nodes in tree: {len(tree_dict)}")
  print(f"Parent-child relationships: {len(relationships)}")
  print(f"Predictable parent nodes: {len(predictions)}")

  for parent, years in predictions.items():
      parent_desc = tree_dict[parent]['description']
      print(f"\nParent: {parent_desc}")
      print(f"  Node ID: {parent}")
      print(f"  Predictable in years: {years}")


Prediction Algorithm Details
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The prediction algorithm works as follows:

* **Year Analysis**: Identifies the years in which the parent currently exists
* **Child Verification**: For each target year, checks if ALL children exist
* **Prediction**: If all children exist in a target year, the parent can be predicted

Example Scenario:
~~~~~~~~~~~~~~~~~~

Parent: "Total Population"
Children: ["Male Population", "Female Population"]

If "Male Population" and "Female Population" both exist in 2016,
but "Total Population" doesn't exist in 2016,
then "Total Population" can be predicted for 2016.


Use Cases for Tree Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Data Completeness Assessment**: Identify missing parent nodes across years
* **Prediction Validation**: Verify which parent nodes can be reliably predicted


Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Memory Usage
  
  * Large datasets may require significant RAM
  * Consider processing in chunks for very large datasets
  * Use multithreaded approach for better memory management


Data Structures
---------------

Input DataFrame Format
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  {
      'vector': 'v_CA21_1234',
      'type': 'Total',
      'description': 'Population aged 25-34 years',
      'details': 'Detailed description...'
  }


Output Mapping Format
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  {
      'description': 'Population aged 25-34 years',
      'vector_base': 'v_CA21_1234',
      'vector_cmp': 'v_CA16_1234'
  }


Merged Mapping Format
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  {
      'description': 'Population aged 25-34 years',
      'vector_base': 'v_CA21_1234',
      'vector_cmp_list': ['v_CA16_1234', 'v_CA11_1234', 'v_CA06_1234']
  }

Troubleshooting
----------------

Import Errors
~~~~~~~~~~~~~~

.. code-block:: python


  # Solution: Add correct path
  import sys
  sys.path.append('../src/piccard')
  from variable_linker import VariableLinker


File Not Found Errors
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python


  # Check file paths
  import os
  print("Current directory:", os.getcwd())
  print("Files available:", os.listdir('.'))


Memory Issues
~~~~~~~~~~~~~~

* Reduce batch size for large datasets
* Use multithreaded approach
* Process data in chunks

Poor Matching Results
~~~~~~~~~~~~~~~~~~~~~~

* Adjust similarity threshold
* Try different matching approaches
* Check data quality and consistency


Configuration Options
---------------------

Similarity Thresholds

* **Strict**: 0.95+ for critical variables
* **Standard**: 0.9 for most use cases
* **Relaxed**: 0.7-0.8 for exploratory analysis

Transformer Models

* ``'all-MiniLM-L6-v2'``: Fast, good accuracy
* ``'all-mpnet-base-v2'``: Best accuracy, slower
* Other Transformer Models can be found at [SBERT Pretrained Models](https://sbert.net/docs/sentence_transformer/pretrained_models.html)
