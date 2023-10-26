
![](https://cea-metrocarac.github.io/images_registration/images_registration.png)


# Images_registration

**images_registration** is a web application dedicated :

- 1/ to the calculation of the `transformation matrix` related to a pair of images and 

![](https://cea-metrocarac.github.io/images_registration/images_registration4.png)


- 2/ to the application of the previous `transformation matrix` to a set of images.

![](https://cea-metrocarac.github.io/images_registration/images_registration3.png)


# Installation

All the application is contained in the `app.py` file located at the root of the repository.

For a fast install and assuming you have already a python environment, you can just `copy` + `paste` this file and update your env. with the required packages listed in the `requirements.txt` file.

For a full install, clone the project from the following [git](https://git-scm.com/downloads) command:

```bash
git clone https://github.dev/CEA-MetroCarac/images_registration.git
```

Create and activate a dedicated virtual environment as explained [here](https://realpython.com/python-virtual-environments-a-primer/), and install the `requirements.txt` file:

```bash
pip install -r requirements.txt -U
```

### Usage

Once your python environment has been created and activated, you can launch the web application in a terminal typing:

```python
python app.py
```
This instruction will launch the web application directly in your web browser.

for more informations concerning the application parameters setting, see the [documentation](https://github.com/CEA-MetroCarac/images_registration/tree/main/doc) .


### Authors informations

In case you use this web application to align images for a study wich leads to an article, please cite:

- Patrick Quéméré, Univ. Grenoble Alpes, CEA, Leti, F-38000 Grenoble, France, https://github.dev/CEA-MetroCarac/images_registration

- P. Thevenaz, U. E. Ruttimann and M. Unser, "A pyramid approach to subpixel registration based on intensity," in IEEE Transactions on Image Processing, vol. 7, no. 1, pp. 27-41, Jan. 1998, doi: 10.1109/83.650848.