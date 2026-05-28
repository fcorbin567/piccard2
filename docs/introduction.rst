Introduction
============

Overview
--------

Urban researchers rely on census data to identify and analyze demographic trends over time. Understanding these trends is essential for planning new infrastructure, supporting immigrant communities, and providing local services, among several other goals. However, due to the same demographic trends that census data is meant to illuminate, population changes can lead to new census boundaries being drawn. This makes the seemingly simple task of analyzing changes in a region difficult because a region defined by a specific boundary in any given year may not exist in other years.

``piccard`` combines a novel solution to this problem with data clustering algorithms to streamline the data analysis process. ``piccard``'s solution significantly improves on the traditional one, geographical harmonization, which involves defining a common set of regions across all years and fitting data to these regions. This method always introduces some amount of error, and harmonization methods are not readily available for some types of data, which makes it difficult to analyze and visualize that data.

``piccard`` represents census regions as nodes in a graph network. Each census region in a specific year represents a node, and two nodes are connected if they represent consecutive census years and share at least a specific percentage of geographical overlap. When identifying trends in a specific region over time, every path through the graph containing that region is analyzed.

``piccard`` integrates network creation, data clustering, and visualization into one tool, an approach that makes useful analysis possible for data that cannot easily be harmonized. Also, ``piccard`` is able to efficiently create networks by utilizing parallel processing for large datasets and incorporating flexibility over different coordinate systems.

For more information about the theory behind ``piccard``, see `this research paper <https://doi.org/10.31235/osf.io/a3gtd>`_.

Modules
--------

The functionality of ``piccard`` is broken into four modules. 

The first module, network creation, focuses on efficiently processing census data and representing it as a graph network. 

The second module, clustering, uses ``tscluster``'s flexible time-series clustering algorithms to cluster census regions in ``piccard`` networks. 

The third module, visualization and analysis, provides highly customizable and accessible visualizations of ``piccard`` networks and clustering results, and also offers probabilistic analysis supported by ``ppandas``.

Finally, the fourth module, VariableLinker, is a tool for understanding the links between census variables over time. Like census regions, census variables change considerably over time, and VariableLinker allows users to match variables based on their semantic meaning over time.

Tests
------

To run the tests for ``piccard``, clone the repository and run the following commands in the root directory: 

```
pip install -e .
pytest --import-mode=importlib tests/
```

Do not attempt to run the tests without installing piccard as a package via pip, as the tests rely on relative imports that will not work otherwise.

Example usage
-------------

For a real-life example using the first three modules of ``piccard``, see this `Colab notebook  <https://colab.research.google.com/drive/1hbB9azjewuebulMy-qAA0VleBHsrb1SI?usp=sharing>`_.

For a real-life example of the fourth module, see this `Colab notebook  <https://colab.research.google.com/drive/15IjPsANO3YiRZSQUtPONaZcd_ot2u2-i?usp=sharing>`_.

Licence
-------

This software is distributed under a CC0-1.0 Licence.

GitHub repository
-----------------

To report a bug, contribute a fix, or look at the code behind ``piccard``, see this `GitHub repository  <https://github.com/fcorbin567/piccard2>`_.