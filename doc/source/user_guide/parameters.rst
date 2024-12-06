Files selection / Parameters settings
-------------------------------------


Files selection
~~~~~~~~~~~~~~~

The images are loaded using the ``Select Files`` buttons associated with each of the **Fixed Image** and **Moving Image** boxes.

If the images are oriented differently, it is possible to rotate them by 90°, 180°, or 270° using the ``Options`` tab.

.. figure::  ../_static/options.png
   :align:   center
   :width:   40%

   ``Options`` tab.


The **alignment** is performed in pairs. By consequence the number of selected images must meet the following criteria:

- **N-to-N Alignment**: The number of **Fixed** images must match the number of **Moving** images. In this case, the images will be processed pairwise in the order in which they appear in their respective file lists.


- **1-to-N Alignment**: There must be exactly one **Fixed** image and any number of **Moving** images. In this case, all **Moving** images will be aligned to the single **Fixed** image provided.

.. figure::  ../_static/N_to_N.png
   :align:   center
   :width:   95%

   Illustration of different files selection use cases.


Defining ROIs
~~~~~~~~~~~~~

In addition to defining ROIs (regions of interest) using the mouse, they can also be specified textually in the dedicated fields, following the convention: **[xmin, xmax, ymin, ymax]**, based on the coordinate system of the image displayed in **Panel 4**.


.. figure::  ../_static/roi.png
   :align:   center
   :width:   35%
