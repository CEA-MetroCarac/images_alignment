"""
Example via the panel application
"""
from pathlib import Path
import tempfile
from tkinter import Tk

from images_alignment.application.app import App
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
        app.model.areas = [[130, 330, 50, 220], [120, 190, 40, 140]]
        app.model.registration(registration_model='StackReg')

    elif registration_model == 'SIFT':
        app.model.registration(registration_model='SIFT')

    # apply the transformation to the set of images
    # app.view.apply_to_all(dirname_res=dirname / 'results')

    app.view.update()

    root.mainloop()


if __name__ == '__main__':
    dirfunc = UserTempDirectory  # use the user temp location
    # dirfunc = tempfile.TemporaryDirectory  # use a TemporaryDirectory

    with dirfunc() as tmpdir:
        dirname = Path(tmpdir) / "images_alignement"
    dirname.mkdir(exist_ok=True)

    # example_tkinter(dirname)
    example_tkinter(dirname, registration_model='StackReg')
    # example_tkinter(dirname, registration_model='SIFT')
    # example_tkinter(dirname, registration_model='User-Driven')
