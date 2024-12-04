![](https://cea-metrocarac.github.io/images_registration/images_alignment.png)

# Images_Alignment

![](https://cea-metrocarac.github.io/images_registration/appli.png)

**Images_Alignment** is an application dedicated to realign pairs of images by applying affine transformation matrices calculated either from the pyStackReg or Sift algorithm, a combined approach or from user driven coupling vectors.

Once the parameters have been set, these ones can be applied to a set of images (optional).

![](https://cea-metrocarac.github.io/images_registration/example_series_1.png)

![](https://cea-metrocarac.github.io/images_registration/example_series_2.png)

*(Figures above are extracted from the `example.py` available [here](https://github.com/CEA-MetroCarac/images_registration/images_alignment/examples/example.py).*

# Installation

Assuming you have already a python environment, to install the application with **pip**:

```bash
pip install git+https://github.com/CEA-MetroCarac/images_registration.git
```

### Usage

Once your python environment has been activated, you can simply launch the application in a terminal writing:

```bash
images_alignment
```

for more informations concerning the application parameters setting, see the [documentation](https://github.com/CEA-MetroCarac/images_registration/doc).

### Authors informations

In case you use this web application to align images for a study which leads to an article, please cite:

- Patrick Quéméré, Univ. Grenoble Alpes, CEA, Leti, F-38000 Grenoble, France, https://github.dev/CEA-MetroCarac/images_registration

- P. Thevenaz, U. E. Ruttimann and M. Unser, "A pyramid approach to subpixel registration based on intensity," in IEEE Transactions on Image Processing, vol. 7, no. 1, pp. 27-41, Jan. 1998, doi: 10.1109/83.650848.