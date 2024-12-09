[![Docs](https://img.shields.io/badge/docs-passing-brightgreen)](https://CEA-MetroCarac.github.io/images_alignment)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14338698.svg)](https://doi.org/10.5281/zenodo.14338698)


<p align="center" width="100%">
    <img align="center" width=250 src=https://cea-metrocarac.github.io/images_alignment/logo.png>
</p>

# Images_Alignment

**Images_Alignment** is an application dedicated to ease the images alignment using affine transformation matrices calculated either from the [pyStackReg](https://github.com/glichtner/pystackreg) or [SIFT](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_sift.html) algorithm, a combined approach, or from a user driven approach (specifying manually the matching points).

Once the parameters have been set, they can be automatically applied to a set of images as part of a workflow.

<p align="center" width="100%">
    <img align="center" width="75%" src=https://cea-metrocarac.github.io/images_alignment/appli.png> <br>
    <em>View of the GUI</em> 
</p>


<p align="center" width="100%">
    <img align="center" width="60%" src=https://cea-metrocarac.github.io/images_alignment/example_series_1.png> 
</p>

<p align="center" width="100%">
    <img align="center" width="60%" src=https://cea-metrocarac.github.io/images_alignment/example_series_2.png> <br>
    <em>Application to a set of 3 images.</em>

(The figures above have been generated from `example.py` available [here](https://github.com/CEA-MetroCarac/images_alignment/images_alignment/examples/example.py).)

## Installation and Usage

Assuming you have a python environment which is activated in a terminal, the application can be installed with the ``pip`` command:

```bash
pip install git+https://github.com/CEA-MetroCarac/images_alignment.git
```

Once installed, the application can be launched directly from a terminal (with the previous python environment activated), by:

```bash
images_alignment
```

For more information concerning the usage and the parameters settings, see the [documentation](https://CEA-MetroCarac.github.io/images_alignment).

## Acknowledgements

This work, carried out on the CEA-PFNC (Platform for Nanocharacterisation), was supported by the “Recherche Technologique de Base” program of the French National Research Agency (ANR).

## Authors information

In case you use this application to align images for a study which leads to an article, please cite:

- Quéméré, P. (2024). Images_alignment : A python package to ease images alignment with a dedicated GUI (2024.1). Zenodo. https://doi.org/10.5281/zenodo.14338698

- P. Thevenaz, U. E. Ruttimann and M. Unser, "A pyramid approach to subpixel registration based on intensity," in IEEE Transactions on Image Processing, vol. 7, no. 1, pp. 27-41, Jan. 1998, doi: 10.1109/83.650848.
