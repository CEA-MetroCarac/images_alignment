Visualization
-------------

.. figure::  ../_static/visualization.png
   :align:   center
   :width:   75%

.. raw:: html

   <br>

Once the images to be processed are selected (see next section), **Panel 2** provides the following features (from left to right):

- Changing the image representation mode: switching between displaying images in their original colors or binarized versions (see the Registration algorithm section for more details).

- Adjusting image resolution during display: ensuring faster rendering, especially for high-resolution images with several thousand pixels in each direction.

- Selecting the juxtaposition mode: choosing between horizontal or vertical alignment of the images.

- Displaying the Combined images considering the full frame or their overlapping regions only (enabling ``Apply Mask``).


**Panel 3** as 'thumbnails' allows users to select the type of image (**Fixed image**, **Moving image**,  **Juxtaposed images** or **Combined image**) to view in within **Panel 4**.


**Panel 4** consists of large image displaying, dotted with navigation toolbars capabilities (like Panning, Zooming, and Saving). This window enables:

- to crop images to regions of interest (ROI) related to the **Fixed image** or the **Moving image**,  using the mouse (by left-clicking and dragging).

- to manually define matching points with the mouse in the **Juxtaposed image** mode in the frame of the **User-Driven** registration approach (see further).


.. figure::  ../_static/cropping_and_matching.png
   :align:   center
   :width:   100%

   (left) cropping and (right) user-driven matching point definition.


