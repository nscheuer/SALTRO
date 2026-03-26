Installation
============

System Setup
------------

Install required system packages:

.. code-block:: bash

    sudo apt update
    sudo apt upgrade
    sudo apt install cmake ccache
    sudo apt install -y build-essential python3-dev

Virtual Environment for Python Interface
-----------------------------------------

Create and activate a Python virtual environment:

.. code-block:: bash

    sudo apt install python3-venv
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

Building the Project
--------------------

Release Build
~~~~~~~~~~~~~

.. code-block:: bash

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSALTRO_BUILD_PYTHON=ON
    cmake --build build -j

Debug Build
~~~~~~~~~~~

For debugging with gdb, use Debug mode:

.. code-block:: bash

    sudo apt install gdb
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DSALTRO_BUILD_PYTHON=ON
    cmake --build build -j

Testing
-------

Run the test suite:

.. code-block:: bash

    cd build
    ctest --output-on-failure
    cd ../tests
    pytest -q

Documentation Generation
-------------------------

Set up the documentation environment:

.. code-block:: bash

    python3 -m venv venv
    pip install breathe pydata_sphinx_theme
    sudo apt install doxygen
    source venv/bin/activate

Generate the documentation:

.. code-block:: bash

    cd docs/
    doxygen
    python3 generate_rst.py
    python3 -m sphinx -W -b html source build

Cleaning Up Documentation Build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To clean up documentation build artifacts, remove these directories:

.. code-block:: bash

    rm -rf docs/build
    rm -rf docs/source/api
    rm -rf docs/source/html
    rm -rf docs/source/latex
    rm -rf docs/source/xml

