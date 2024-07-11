"""
Application for images registration
"""
import sys
import numpy as np
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import AffineTransform
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac
from scipy.interpolate import RegularGridInterpolator


class Terminal:
    """ Class to 'write' in the console """

    def write(self, message):
        """ Write message in the console """
        sys.stdout.write(message)
        sys.stdout.flush()


def gray_conversion(img):
    """ Convert RGBA or RGB image to gray image """
    if img.ndim == 4:
        img = rgba2rgb(img)
    if img.ndim == 3:
        img = rgb2gray(img)
    return img


def image_normalization(img):
    """ Normalize image in range [0., 1.] """
    vmin, vmax = img.min(), img.max()
    return (img - vmin) / (vmax - vmin)


def edges_trend(img):
    """ Estimate if the average value along the image edges is > 0.5 """
    shape = img.shape
    mask = np.ones_like(img, dtype=bool)
    mask[1:-1, 1:-1] = 0
    sum_edges = np.sum(img[mask])
    return (sum_edges / (2 * (shape[0] + shape[1] - 2))) > 0.5


def padding(img1, img2):
    """ Add image padding """
    shape1 = img1.shape
    shape2 = img2.shape

    hmax = max(img1.shape[0], img2.shape[0])
    wmax = max(img1.shape[1], img2.shape[1])

    img1_pad = np.pad(img1, ((0, hmax - shape1[0]), (0, wmax - shape1[1])))
    img2_pad = np.pad(img2, ((0, hmax - shape2[0]), (0, wmax - shape2[1])))

    return img1_pad, img2_pad


def interpolation(img1, img2):
    """
    Interpolate the low resolution image on the high resolution image support

    Parameters
    ----------
    img1, img2: numpy.ndarray((m, n)), numpy.ndarray((p, q))

    Returns
    -------
    img1_int, img2_int: numpy.ndarrays((r, s))
        The linearly interpolated arrays on the high resolution support of size
        (r, s) = max((m, n), (p, q))
    success: bool
        Status related to differentiation between low and high resolution images
    """
    shape1 = np.array(img1.shape)
    shape2 = np.array(img2.shape)
    success = True

    if (shape1 == shape2).all():
        return img1, img2, success
    elif (shape1 <= shape2).all():
        img_lr = img1
        img_hr = img2
    elif (shape1 >= shape2).all():
        img_lr = img2
        img_hr = img1
    else:
        success = False
        return None, None, success

    # image padding to have same scale ratio between images
    ratio0 = img_hr.shape[0] / img_lr.shape[0]
    ratio1 = img_hr.shape[1] / img_lr.shape[1]
    if ratio0 < ratio1:
        pad = int((img_hr.shape[1] / ratio0) - img_lr.shape[1])
        img_lr = np.pad(img_lr, ((0, 0), (int(pad / 2), pad - int(pad / 2))))
    else:
        pad = int((img_hr.shape[0] / ratio1) - img_lr.shape[0])
        img_lr = np.pad(img_lr, ((int(pad / 2), pad - int(pad / 2)), (0, 0)))

    # low resolution support
    row_lr = np.linspace(0, 1, img_lr.shape[0])
    col_lr = np.linspace(0, 1, img_lr.shape[1])

    # high resolution support
    row_hr = np.linspace(0, 1, img_hr.shape[0])
    col_hr = np.linspace(0, 1, img_hr.shape[1])

    # interpolation
    interp = RegularGridInterpolator((row_lr, col_lr), img_lr)
    rows_hr, cols_hr = np.meshgrid(row_hr, col_hr, indexing='ij')
    pts = np.vstack((rows_hr.ravel(), cols_hr.ravel())).T
    img_lr_int = interp(pts).reshape(img_hr.shape)

    if (shape1 <= shape2).all():
        return img_lr_int, img_hr, success
    else:
        return img_hr, img_lr_int, success


def sift(img1, img2, model_class=None):
    """
    SIFT feature detection and descriptor extraction

    Parameters
    ----------
    img1, img2: numpy.ndarray((m, n)), numpy.ndarray((p, q))
        The input images
    model_class: Objet
        'model_class' used by `RANSAC
        <https://scikit-image.org/docs/stable/api/skimage.measure.html>`_.
        If None, consider ``AffineTransform`` from skimage.transform.

    Returns
    -------
    tmat: numpy.ndarrays((3, 3))
        The related transformation matrix
    keypoints: list of 2 numpy.ndarray((n, 2)
        Keypoints coordinates as (row, col) related to the 2 input images.
    descriptors: list of 2 numpy.ndarray((n, p)
        Descriptors associated with the keypoints.
    matches: numpy.ndarray((q, 2))
        Indices of corresponding matches returned by
        skimage.feature.match_descriptors.

    """
    if model_class is None:
        model_class = AffineTransform

    sift_ = SIFT(upsampling=2)

    keypoints = []
    descriptors = []
    for img in [img1, img2]:
        sift_.detect_and_extract(img)
        keypoints.append(sift_.keypoints)
        descriptors.append(sift_.descriptors)

    matches = match_descriptors(descriptors[0], descriptors[1],
                                cross_check=True, max_ratio=0.8)

    src = keypoints[0][matches[:, 0]][:, ::-1]
    dst = keypoints[1][matches[:, 1]][:, ::-1]
    tmat = ransac((src, dst), model_class,
                  min_samples=4, residual_threshold=2)[0].params

    return tmat, keypoints, descriptors, matches
