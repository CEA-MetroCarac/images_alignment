"""
Example
"""
from pathlib import Path
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from images_alignment import ImagesAlign
from images_alignment.examples.utils import UserTempDirectory, images_generation
from images_alignment.examples.utils import ROIS


def example(dirname, img_name, registration_model):
    """ Example """

    fnames_fixed, fnames_moving = images_generation(dirname, img_name)

    imgalign = ImagesAlign(fnames_fixed=fnames_fixed,
                           fnames_moving=fnames_moving,
                           rois=ROIS[img_name])

    plt.close()  # to close the default figure

    fig0, ax0 = plt.subplots(1, 3, figsize=(12, 4))
    fig0.tight_layout()
    fig0.canvas.manager.set_window_title("Original images")
    imgalign.plot_all(ax=ax0)

    imgalign.registration(registration_model=registration_model)

    fig1, ax1 = plt.subplots(1, 3, figsize=(12, 4))
    fig1.tight_layout()
    fig1.canvas.manager.set_window_title("Processed images")
    imgalign.plot_all(ax=ax1)

    fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4))
    fig2.tight_layout()
    fig2.canvas.manager.set_window_title("Processed images (Binarized)")
    imgalign.binarized = True
    imgalign.plot_all(ax=ax2)


if __name__ == '__main__':
    DIRFUNC = UserTempDirectory  # use the user temp location
    # DIRFUNC = tempfile.TemporaryDirectory  # use a TemporaryDirectory

    IMG_NAMES = ['camera', 'astronaut']
    REGISTRATION_MODELS = ['StackReg', 'SIFT', 'User-Driven']

    with DIRFUNC() as tmpdir:
        dirname = Path(tmpdir) / "images_alignement"
        dirname.mkdir(exist_ok=True)

        example(dirname, IMG_NAMES[0], REGISTRATION_MODELS[0])

    plt.show()
