"""
Example via the panel application
"""
from pathlib import Path
import tempfile
from tkinter import Tk

from images_alignment.application.app_tkinter import App
from example import images_generation, UserTempDirectory


def example_tkinter(dirname, registration_model=None):
    """ Example based on 3 duplicated moving images with additional patterns """

    fnames_fixed, fnames_moving = images_generation(dirname)

    root = Tk()
    app = App(root,
              fnames_fixed=fnames_fixed,
              fnames_moving=fnames_moving,
              thresholds=[0.4, 0.4],
              bin_inversions=[False, False]
              )

    if registration_model == 'StackReg':
        app.model.cropping(0, area=[50, 220, 130, 330])
        app.model.cropping(1, area=[40, 140, 120, 190])
        app.model.resizing()
        app.model.registration(registration_model='StackReg')

    elif registration_model == 'SIFT':
        app.model.registration(registration_model='SIFT')

    # apply the transformation to the set of images
    # app.apply_to_all(dirname_res=dirname / 'results')

    app.view.registration_model.set(app.model.registration_model)
    app.view.update_plots()

    root.mainloop()


if __name__ == '__main__':
    dirfunc = UserTempDirectory  # use the user temp location
    # dirfunc = tempfile.TemporaryDirectory  # use a TemporaryDirectory

    with dirfunc() as tmpdir:
        dirname = Path(tmpdir) / "images_alignement"
    dirname.mkdir(exist_ok=True)

    # example_tkinter(dirname, registration_model='StackReg')
    example_tkinter(dirname, registration_model='SIFT')
