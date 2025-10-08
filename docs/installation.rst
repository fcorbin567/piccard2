Installation
============

Requirements
------------

* pandas>=1.3.0
* numpy>=1.20.0
* geopandas>=0.10.0
* shapely>=1.8.0
* pyproj>=3.0.0
* networkx>=2.6.0
* matplotlib>=3.5.0
* plotly>=5.0.0
* nltk>=3.6.0
* sentence-transformers>=2.0.0
* scikit-learn>=1.0.0
* graphviz>=0.20.0
* swifter>=1.3.0
* typing-extensions>=4.0.0
* hatchling>=1.0.0 

In addition, when using the second and third modules, you will need to install ``tscluster`` and ``ppandas`` respectively.
Keep reading this section for instructions on installing those packages.

Installing ``piccard``
--------------------

To install the current released version:

.. code-block:: shell

    pip install piccard==1.1.1


To install the pre-release version via git:

.. code-block:: shell

    pip install git+https://github.com/ecorbin567/piccard2.git

Then import:

.. code-block:: python
    
    from piccard import piccard as pc


Installing ``tscluster``
----------------------

``tscluster`` requires the following:

* Python 3.8+
* numpy>=1.26 
* scipy>=1.10 
* gurobipy>=11.0 
* tslearn>=0.6.3   
* h5py>=3.10
* pandas>=2.2
* matplotlib>=3.8

Note that you will need a Gurobi licence when using OptTSCluster with large model size. See `here <https://support.gurobi.com/hc/en-us/articles/12684663118993-How-do-I-obtain-a-Gurobi-license>`_ for more about Gurobi licences.

To install the current released version:

.. code-block:: shell

    pip install tscluster


To install the pre-release version via git:

.. code-block:: shell

    pip install git+https://github.com/tscluster-project/tscluster.git


Installing ``ppandas``
----------------------

``ppandas`` requires the following:

* pgmpy==0.1.9
* networkx==2.4
* matplotlib
* python-interval
* geopandas
* geovoronoi

To install via git:

.. code-block:: shell

    pip install git+https://github.com/ecorbin567/ppandas.git