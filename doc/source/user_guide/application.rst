Workflow
--------

The application of the previously defined parameters to a set of images is handled through the **Application** section via the ``APPLY TO ALL`` button.

This requires the user to have pre-defined a **"results"** folder to store the alignment results. This **"results"** folder will contain two sub-folders: **fixed_images** and **moving_images**, which will contain the processed images.

.. note::
    In the case of a 1-to-N image processing, the ``INV`` mode can not be activated.

.. warning::
    To facilitate the viewing of the realigned images after processing, the images displayed in the visualization windows are those located in the **"results"** directories.

    **Consequently**, any new alignment will then be performed from these already processed images, with the risk to overwrite them when saving.

    **To go back to the original images, uncheck the ``Show results`` checkbox.**