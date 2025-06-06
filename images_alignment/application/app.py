"""
PySide-application for the images alignment
"""
import sys
from PySide6.QtWidgets import QApplication

from images_alignment import ImagesAlign
from images_alignment.application.view import View


class App:
    """
    Application for the images alignment

    Parameters
    ----------
    fnames_fixed, fnames_moving: iterables of str, optional
        Images pathnames related to fixed and moving images resp. to handle
    rois: list of 2 iterables, optional
        rois (regions of interest) attached to the fixed and moving images, each defining as:
         [xmin, xmax, ymin, ymax]
    thresholds: iterable of 2 floats, optional
        Thresholds used to binarize the images. Default values are [0.5, 0.5]
    bin_inversions: iterable of 2 bools, optional
        Activation keywords to reverse the image binarization
    """

    def __init__(self,
                 fnames_fixed=None,
                 fnames_moving=None,
                 rois=None,
                 thresholds=None,
                 bin_inversions=None):
        self.model = ImagesAlign(fnames_fixed=fnames_fixed,
                                 fnames_moving=fnames_moving,
                                 rois=rois,
                                 thresholds=thresholds,
                                 bin_inversions=bin_inversions)

        self.view = View(self.model)
        if fnames_fixed:
            self.view.fselectors[0].select_files(fnames=fnames_fixed)
        if fnames_moving:
            self.view.fselectors[1].select_files(fnames=fnames_moving)
        self.view.window.show()


def launcher(fname_json=None):
    """ Launch the appli """

    qapp = QApplication(sys.argv)
    appli = App()

    if fname_json:
        appli.model.reload_params(fname_json=fname_json, obj=appli.model)

    sys.exit(qapp.exec())


if __name__ == '__main__':
    launcher()
