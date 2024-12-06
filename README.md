[![Docs](https://img.shields.io/badge/docs-passing-brightgreen)](https://CEA-MetroCarac.github.io/images_alignment)


<p align="center" width="100%">
    <img align="center" width=250 src=https://cea-metrocarac.github.io/images_registration/images_alignment.png>
</p>

# Images_Alignment

**Images_Alignment** is an application dedicated to ease the images alignment using affine transformation matrices calculated either from the [pyStackReg](https://github.com/glichtner/pystackreg) or [SIFT](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_sift.html) algorithm, a combined approach, or from a user driven approach (specifying manually the matching points).

Once the parameters have been set, these ones can be applied automatically to a set of images as a workflow.

<p align="center" width="100%">
    <img align="center" width="75%" src=https://cea-metrocarac.github.io/images_registration/appli.png> <br>
    <em>View of the GUI</em> 
</p>


<p align="center" width="100%">
    <img align="center" width="60%" src=https://cea-metrocarac.github.io/images_registration/example_series_1.png> 
</p>

<p align="center" width="100%">
    <img align="center" width="60%" src=https://cea-metrocarac.github.io/images_registration/example_series_2.png> <br>
    <em>Application to a set of 3 images.</em>

(The figures above have been generated from `example.py` available [here](https://github.com/CEA-MetroCarac/images_alignment/images_alignment/examples/example.py).)

## Installation and Usage

Assuming you have a python environment which is activated in a terminal, the application can be installed with the ``pip`` command:

```bash
pip install git+https://github.com/CEA-MetroCarac/images_registration.git
```

Once installed, the application can be launched directly from a terminal (with the previous python environment activated), by:

```bash
images_alignment
```

For more information concerning the usage and the parameters settings, see the [documentation](https://github.com/CEA-MetroCarac/images_registration/doc).

## Authors information

In case you use this application to align images for a study which leads to an article, please cite:

- Patrick Quéméré, Univ. Grenoble Alpes, CEA, Leti, F-38000 Grenoble, France, https://github.dev/CEA-MetroCarac/images_registration

- P. Thevenaz, U. E. Ruttimann and M. Unser, "A pyramid approach to subpixel registration based on intensity," in IEEE Transactions on Image Processing, vol. 7, no. 1, pp. 27-41, Jan. 1998, doi: 10.1109/83.650848.