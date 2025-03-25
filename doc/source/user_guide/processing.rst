Processing
----------

Once all the previous parameters have been defined, the alignment of the selected images is performed by pressing the ``REGISTRATION`` button.

An indicative value of the "goodness" of the alignment (**score**) is displayed in the application's console (bottom left). This value is calculated based on the proportion of well-overlapping binarized images relative to the total overlapping area. The quality of the overlap can be assessed by viewing the **Combined image** in binary mode (with ``Apply Mask`` enabled). From a visual point of view, red and green zones indicate areas of non-overlap (equivalent to a 'XOR' operation) between the original binarized **Fixed** and **Moving** images.

.. figure::  ../_static/scoring.png
   :align:   center
   :width:   100%

   **Combined images** after registration in gray (left) and binarization (right) mode respectively leading to a score of **95.8%** with ``StackReg``.


With each registration computation, the matrix corresponding to the **affine transformation** applied to the **Moving image** (regarding the transformations activated in the ``Options`` tab) is displayed in the terminal associated with the application's execution::

    [[  0.77767435   0.43040319 -24.41735391]
     [ -0.63508967   1.16203181  52.11186476]
     [  0.           0.           1.        ]]

The ``INV`` mode allows the roles of **Fixed** and **Moving** images to be reversed during the registration processing.

After registration, each of these images can be saved via the ``SAVE IMAGES`` button.

The alignment parameters (e.g., ROIs, thresholds, ...) can also be saved in a .json file using ``SAVE PARAMS`` for later reuse with ``RELOAD PARAMS``.

.. note::
    To reduce computation times during the registration processing, high-resolution images are sub-sampled to a size of **512 pixels** per  dimension. This criterion can be adjusted through the ``Max image size`` parameter under the **Registration** section in the ``Options`` tab.