"""
Example via the Tkinter application
"""
from pathlib import Path
from tkinter import Tk

from images_alignment.application.app import App
from images_alignment import REG_MODELS

from utils import UserTempDirectory, images_generation
from utils import ROIS


def example_appli(dirname, img_name, registration_model):
    """ Example based on 3 duplicated moving images using the application """

    input_dirname = dirname / 'example_appli'
    input_dirname.mkdir(exist_ok=True)

    fnames_fixed, fnames_moving = images_generation(input_dirname, img_name, nimg=2)

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
        dirname = Path(tmpdir) / "images_alignment"
        example_appli(dirname, IMG_NAMES[2], REG_MODELS[1])
