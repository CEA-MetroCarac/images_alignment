Welcome to the Image_Alignment documentation !
==============================================


**Images_Alignment** is an application dedicated to ease the images alignment using affine transformation matrices calculated either from the `pyStackReg <https://github.com/glichtner/pystackreg)>`_ or `SIFT <https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_sift.html>`_ algorithm, a combined approach, or from a user driven approach (specifying manually the matching points).

Once the parameters have been set, they can be automatically applied to a set of images as part of a workflow.

**This documentation mainly focuses on using the application in GUI mode.**

For a **scripting mode usage**, please refer to the final section of this documentation.


Installation
------------

Assuming you have a python environment which is activated in a terminal, the application can be installed thanks the ``pip`` command::

    pip install git+https://github.com/CEA-MetroCarac/images_alignment.git

Once installed, the application can be launched directly from a terminal (with the previous python environment activated), by::

    images_alignment


Tests execution
---------------

For developers, the full package including examples and tests can be installed via::

    git clone https://github.com/CEA-MetroCarac/images_alignment.git
    cd images_alignment
    pip install .


The tests can be executed via the following commands::

    pip install pytest
    pytest tests



Image_Alignment documentation
-----------------------------


.. toctree::
   :maxdepth: 1
   :caption: GUI Mode

   user_guide/intro
   user_guide/visualization
   user_guide/parameters
   user_guide/registration
   user_guide/processing
   user_guide/application

.. toctree::
   :maxdepth: 1
   :caption: Scripting Mode

   user_guide/scripting

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   api/modules.rst