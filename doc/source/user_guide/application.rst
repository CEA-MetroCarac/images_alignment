Workflow
--------

The application of the previously defined parameters to a set of images is handled through the **Application** section via the ``APPLY TO ALL`` button.

This requires the user to have pre-defined a **"results"** folder to store the alignment results. This **"results"** folder will contain two sub-folders: **fixed_images** and **moving_images**, which will contain the processed images.

.. note::
    Since the user can choose to invert the "Fixed/Moving" roles using the ``INV`` button, each **Fixed** (corresponding to a moving image in the ``INV`` mode) is saved with an extension related to its processing order in the workflow.

.. warning::
    To facilitate the viewing of the realigned images after processing, the images displayed in the visualization windows are those located in the **"results"** directories. Changing the images in the processed set can be done using the image selection frames. Consequently, the displayed images are no longer the original images.

    Any new registration will then apply to these processed images and potentially overwrite them.

    To reprocess the original images, it is necessary to uncheck the ``Show results`` checkbox.