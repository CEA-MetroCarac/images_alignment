
![](https://cea-metrocarac.github.io/images_registration/images_registration.png)

# Images_registration

**images_registration** is an application dedicated to 1/ register a pair of images then 2/ to apply the transformation matrix to a set of images.


# Installation

All the application is contained in the `app.py` file located at the root of the repository.

For a fast install, assuming you have already a python environment, you can just `copy` + `paste` this file and update your env. with the required packages listed in the `requirements.txt` file.

For a full install, install [git](https://git-scm.com/downloads), then clone the project:

```bash
git clone https://github.dev/CEA-MetroCarac/images_registration.git
```

Create and activate a virtual environment, as explained in [Python Virtual Environments: A Primer](https://realpython.com/python-virtual-environments-a-primer/), and install the `requirements.txt` file:

```bash
pip install -r requirements.txt -U
```

### Usage

Once your python environment has been created and activated, you can launch the application on a Panel server via the instruction:

```python
panel serve app.py --show --autoreload
```

See the [documentation](https://github.com/CEA-MetroCarac/images_registration/tree/main/doc) for more details.


### Authors informations

In case you use this application to align images for a study wich leads to an article, please cite:

- https://github.dev/CEA-MetroCarac/images_registration