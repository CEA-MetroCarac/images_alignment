"""
Example via the Tkinter application
"""
from pathlib import Path
from tkinter import Tk

from images_alignment.application.app import App
from images_alignment.examples.utils import UserTempDirectory, images_generation
from images_alignment.examples.utils import ROIS


def ex_appli(dirname, img_name, registration_model):
    """ Example based on 3 duplicated moving images using the application """

    fnames_fixed, fnames_moving = images_generation(dirname, img_name, nimg=2)

    root = Tk()
    app = App(root,
              fnames_fixed=fnames_fixed,
              fnames_moving=fnames_moving,
              rois=ROIS[img_name])

    if registration_model in ["StackReg", "SIFT"]:
        app.model.registration(registration_model=registration_model)

    # apply the transformation to the set of images
    # app.view.apply_to_all(dirname_res=dirname / 'results')

    app.view.update()

    root.mainloop()


if __name__ == '__main__':
    DIRFUNC = UserTempDirectory  # use the user temp location
    # DIRFUNC = tempfile.TemporaryDirectory  # use a TemporaryDirectory

    IMG_NAMES = ['camera', 'astronaut']
    REGISTRATION_MODELS = ['StackReg', 'SIFT', 'User-Driven']

    with DIRFUNC() as tmpdir:
        dirname = Path(tmpdir) / "images_alignement"
        dirname.mkdir(exist_ok=True)

        ex_appli(dirname, IMG_NAMES[1], REGISTRATION_MODELS[1])
