"""
Example via the panel application
"""
from pathlib import Path
import tempfile

from images_alignment.application.app import App
from example import images_generation, UserTempDirectory


def example_panel(dirname):
    """ Example based on 3 duplicated moving images with additional patterns """

    fnames_fixed, fnames_moving = images_generation(dirname)

    app = App(fnames_fixed=fnames_fixed,
              fnames_moving=fnames_moving,
              thresholds=[0.15, 0.15],
              bin_inversions=[False, False])

    imgalign = app.model
    imgalign.cropping(1, area_percent=[0.40, 0.95, 0.25, 1.00])
    imgalign.resizing()
    imgalign.binarization()
    imgalign.registration(registration_model='StackReg')

    # apply the transformation to the set of images
    app.apply_to_all(dirname_res=dirname / 'results')

    app.update_plots()
    app.window.show()


if __name__ == '__main__':
    dirfunc = UserTempDirectory  # use the user temp location
    # dirfunc = tempfile.TemporaryDirectory  # use a TemporaryDirectory

    with dirfunc() as tmpdir:
        dirname = Path(tmpdir) / "images_alignement"
        dirname.mkdir(exist_ok=True)

        example_panel(dirname)
