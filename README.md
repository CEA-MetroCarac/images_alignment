<p align="center" width="100%">
    <img align="center" width=250 src=https://cea-metrocarac.github.io/images_registration/images_alignment.png>
</p>

# Images_Alignment

**Images_Alignment** is an application dedicated to facilitate pairs of images realignment using affine transformation matrices calculated either from the [pyStackReg](https://github.com/glichtner/pystackreg) or [SIFT](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_sift.html) algorithm, a combined approach or from user driven matching points.

Once the parameters have been set, these ones can be applied to a set of images (optional).

<p align="center" width="100%">
    <img align="center" width="75%" src=https://cea-metrocarac.github.io/images_registration/appli.png>
</p>

<p align="center" width="100%">
    <img align="center" width="40%" src=https://cea-metrocarac.github.io/images_registration/example_series_1.png>
</p>

<p align="center" width="100%">
    <img align="center" width="40%" src=https://cea-metrocarac.github.io/images_registration/example_series_2.png>
</p>


*(The 2 figures above are extracted from the `example.py` available [here](https://github.com/CEA-MetroCarac/images_registration/images_alignment/examples/example.py).*

# Installation

Assuming you have a python environment which is activated in a console, the application can be installed by the ``pip`` command:

```bash
pip install git+https://github.com/CEA-MetroCarac/images_registration.git
```

### Usage

The application can be launching directly from a console (after activating your python environment) by:

```bash
images_alignment
```

for more informations concerning the application usage and more particularly the parameters setting, see the [documentation](https://github.com/CEA-MetroCarac/images_registration/doc).

### Authors information

In case you use this application to align images for a study which leads to an article, please cite:

- Patrick Quéméré, Univ. Grenoble Alpes, CEA, Leti, F-38000 Grenoble, France, https://github.dev/CEA-MetroCarac/images_registration

- P. Thevenaz, U. E. Ruttimann and M. Unser, "A pyramid approach to subpixel registration based on intensity," in IEEE Transactions on Image Processing, vol. 7, no. 1, pp. 27-41, Jan. 1998, doi: 10.1109/83.650848.