The registration is based on a `transformation matrix` calculated from 2 images: a ``moving`` and a ``fixed`` image.

Once the 2 images have been loaded, the registration workflow that consists in images resizing and binarization as preliminary steps, is automatically performed (default mode).

The resulting `transformation matrix` can be applied to a set of images, assuming that these images are consistant with the one used previoulsy as 'moving image'.

# image parameters settings

The 2 images are loaded from the 2 dedicated boxes (**moving image** and **fixed image**) located at the top left of the window.

Once loaded, the images can be cropped via the 2 ranges sliders defining the cropping area: the first one for the `H`orizonal range selection and the second one for the `V`ertical range selection.

The thresholds sliders are attached to the binarization processing. 
To make the binarized images compatibles each other, the binarization step realizes automatic background detections. Nevertheless, in case of incorrect background determination, the user has the possibility to reverse each one of the binarized images.

![](_static/image_settings.png)


# images processing

The images processing consists in a **resizing** step, a **binarization** step and a **registration** one.

![](_static/images_processing.png)


**Resizing** consists in a image projection from the low to the high image resolution in the aim to work with identical images shapes and make the registration possible. There is no constraint about which one of the 2 images (moving or fixed) should be low or high resolution. 
Nonetheless, it is important to note that to preserve the same x-y scaling during the projection, a x or y padding can be added to the low image resolution.

**Binarization** relying on the threshold parameters defined above is an important step. For a correct registration, the binarized images should have the most similar rendering as possible.

**Registration** is performed according to the [pyStackreg](https://pystackreg.readthedocs.io/en/latest/readme.html) library, considering the "scaled rotation" transformation (translation + rotation + scaling).

Once the **registration** has been done in automatic or iterative (step-by-step) mode, the user has the possibility to adjust manually the registration (translation and rotation). The center of rotation is defined from the xc, yc relative coordinates associated to the resized images.

The resulting `transformation matrix` is displayed at any time in the bottom of the corresponding layout.

# images viewing

2 images view modes are available:

- `Difference` (default mode), used to highlight the differences between the 2 binarized-registered images

- `Overlay` which represents the fused (averaged) image in gray colors

![](_static/view_mode.png)


# images batch application

Once the registration has been done, the registration worflow (cropping, resizing, binarization and registration) can be applied similarly to a set of images defined in a `INPUT directory`. The resulting registred images are saved in the `OUTPUT directory` as well as the ``fixed`` image used as reference.

![](_static/apply.png)


# Saving and reloading

All the workflow parameters and the transformation matrix can be saved to and reloaded from  a `.json` file.

![](_static/save_reload.png)
