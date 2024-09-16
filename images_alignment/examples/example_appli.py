"""
Example via the Tkinter application
"""
from pathlib import Path
from tkinter import Tk

from images_alignment.application.app import App
from images_alignment.examples.utils import UserTempDirectory, images_generation
from images_alignment.examples.utils import ROIS
from images_alignment import REG_MODELS


def ex_appli(dirname, img_name, registration_model):
    """ Example based on 3 duplicated moving images using the application """

    fnames_fixed, fnames_moving = images_generation(dirname, img_name, nimg=2)

    root = Tk()
    app = App(root,
              fnames_fixed=fnames_fixed,
              fnames_moving=fnames_moving,
              rois=ROIS[img_name])

    app.model.registration_model = registration_model

    if registration_model in REG_MODELS[:3]:
        app.model.registration()

    # apply the transformation to the set of images
    # app.view.apply_to_all(dirname_res=dirname / 'results')

    app.view.update()

    root.mainloop()


if __name__ == '__main__':
    DIRFUNC = UserTempDirectory  # use the user temp location
    # DIRFUNC = tempfile.TemporaryDirectory  # use a TemporaryDirectory
    IMG_NAMES = ['camera', 'astronaut', 'shepp_logan_phantom']

    with DIRFUNC() as tmpdir:
        dirname = Path(tmpdir) / "images_alignement"
        dirname.mkdir(exist_ok=True)

        ex_appli(dirname, IMG_NAMES[1], REG_MODELS[2])
