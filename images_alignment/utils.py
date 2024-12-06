"""
Utilities functions or Classes
"""
import sys
from pathlib import Path
import numpy as np
import tifffile
from skimage.color import rgba2rgb, rgb2gray, gray2rgb
from skimage.transform import AffineTransform, resize
from skimage.feature import SIFT, match_descriptors
from skimage.measure import ransac


class Terminal:
    """ Class to 'write' into the console """

    def write(self, message):
        """ Write message into the console """
        sys.stdout.write(message)
        sys.stdout.flush()


def flatten(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def fnames_multiframes_from_list(fnames):
    fnames_new = []
    for fname in fnames:
        fnames_new.append(fnames_multiframes(fname))
    return flatten(fnames_new)


def fnames_multiframes(fname):
    """ Return/create fnames in the TMP_DIR if the .tif has multiple frames """
    from images_alignment import TMP_DIR

    name = Path(fname).name
    try:
        with tifffile.TiffFile(fname) as tif:
            num_frames = len(tif.pages)
            if num_frames > 1:
                fnames = []
                for i in range(num_frames):
                    fnames.append(TMP_DIR / f"({i}) {name}")
                    img = tif.pages[i].asarray()
                    tifffile.imwrite(fnames[-1], img, dtype=img.dtype)
                return fnames
            else:
                return fname
    except:
        return fname


def gray_conversion(img):
    """ Convert RGBA or RGB image to gray image """
    if img.ndim == 3 and img.shape[2] == 4:
        img = rgba2rgb(img)
    if img.ndim == 3:
        img = rgb2gray(img)
    return img


def rescaling_factor(imgs, max_size):
    """ Return a 'global' rescaling factor satisfying 'max_size' """
    rfac = 1.
    size_max = max(max(imgs[0].shape), max(imgs[1].shape))
    if size_max > max_size:
        rfac = max_size / size_max
    return rfac


def rescaling_factors(imgs, max_size):
    """Return the rescaling factors satisfying max_size for both item of imgs"""
    return (min(1., max_size / max(imgs[0].shape)),
            min(1., max_size / max(imgs[1].shape)))


def imgs_rescaling(imgs, max_size):
    """Rescale images according to 'max_size'"""
    rfacs = rescaling_factors(imgs, max_size)
    if rfacs[0] < 1:
        imgs[0] = rescaling(imgs[0], rfacs[0])
    if rfacs[1] < 1:
        imgs[1] = rescaling(imgs[1], rfacs[1])
    return imgs, rfacs


def imgs_conversion(imgs):
    """ Uniformize the number of dimension/channels (for images composition) """

    def convert(img, img_target):
        if img.ndim == 2:
            img = gray2rgb(img)
        if img.shape[2] == 3 and img_target.shape[2] == 4:
            dtype = img.dtype
            alpha_channel = np.ones((img.shape[0], img.shape[1]), dtype=dtype)
            if dtype == np.uint8:
                alpha_channel *= 255
            img = np.dstack([img, alpha_channel])
        return img

    shape0 = imgs[0].shape
    shape1 = imgs[1].shape

    if imgs[0].ndim > imgs[1].ndim or \
            (len(shape0) == len(shape1) == 3 and shape0[2] > shape1[2]):
        imgs[1] = convert(imgs[1], imgs[0])

    elif imgs[0].ndim < imgs[1].ndim or \
            (len(shape0) == len(shape1) == 3 and shape0[2] < shape1[2]):
        imgs[0] = convert(imgs[0], imgs[1])

    return imgs


def image_normalization(img):
    """ Normalize image in range [0., 1.] """
    vmin, vmax = np.nanmin(img), np.nanmax(img)
    return (img - vmin) / (vmax - vmin)


def absolute_threshold(img, relative_threshold):
    """ Return the absolute threshold to use when binarizing a 'img' """
    hist, edges = np.histogram(img[~np.isnan(img)].flatten(), bins=1000)
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]

    delta = np.abs(cdf - relative_threshold)
    ind = np.where(delta == delta.min())[0][-1]  # keep the last 'min. item'

    abs_threshold = edges[ind + 1]

    return abs_threshold


def resizing(img1, img2):
    """ Resize the images to have similar shape (requested for pyStackReg) """
    # if img1.size <= img2.size:
    #     img1 = resize(img1, img2.shape, preserve_range=True)
    # else:
    #     img2 = resize(img2, img1.shape, preserve_range=True)
    img2 = resize(img2, img1.shape, preserve_range=True)
    return [img1, img2]


def cropping(img, area, verbosity=True):
    """ Return cropped image according to the given area """
    if area is None:
        return img

    assert np.asarray(area).dtype == int
    shape = img.shape
    xmin, xmax, ymin, ymax = area
    imin, imax = shape[0] - ymax, shape[0] - ymin
    jmin, jmax = xmin, xmax
    imin, imax = min(shape[0], max(0, imin)), min(shape[0], max(0, imax))
    jmin, jmax = min(shape[1], max(0, jmin)), min(shape[1], max(0, jmax))
    if imin == imax or jmin == jmax:
        if verbosity:
            size = (shape[1], shape[0])
            print(f"Warning: the cropping area {area} is not suitable with image of size {size}")
        return img
    else:
        return img[imin:imax, jmin:jmax]


def padding(img1, img2):
    """ Add image padding """
    shape1 = img1.shape
    shape2 = img2.shape

    hmax = max(shape1[0], shape2[0])
    wmax = max(shape1[1], shape2[1])

    pad_width1 = [[0, hmax - shape1[0]], [0, wmax - shape1[1]]]
    pad_width2 = [[0, hmax - shape2[0]], [0, wmax - shape2[1]]]

    if len(shape1) == 3:
        pad_width1 += [[0, 0]]
    if len(shape2) == 3:
        pad_width2 += [[0, 0]]

    img1_pad = np.pad(img1, pad_width1)
    img2_pad = np.pad(img2, pad_width2)

    return img1_pad, img2_pad


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
    points: list of 2 numpy.ndarray((n, 2)
        Keypoints coordinates as (row, col) related to the 2 input images.
    """
    if model_class is None:
        model_class = AffineTransform

    sift_ = SIFT()

    keypoints = []
    descriptors = []
    for img in [img1, img2]:
        sift_.detect_and_extract(img)
        keypoints.append(sift_.keypoints)
        descriptors.append(sift_.descriptors)

    matches = match_descriptors(descriptors[0], descriptors[1],
                                cross_check=True, max_ratio=0.9)

    src = np.asarray(keypoints[0][matches[:, 0]]).reshape(-1, 2)[:, ::-1]
    dst = np.asarray(keypoints[1][matches[:, 1]]).reshape(-1, 2)[:, ::-1]
    model, inliers = ransac((src, dst), model_class,
                            min_samples=4, residual_threshold=2)

    tmat = model.params
    points = [src[inliers], dst[inliers]]

    return tmat, points


def concatenate_images(img1, img2, alignment='horizontal'):
    """
    concatenate img1 and img2

    Parameters
    ----------
    img1 : (m, n [, 3]) array
        First grayscale or color image.
    img2 : (p, q [, 3]) array
        Second grayscale or color image.
    alignment : {'horizontal', 'vertical'}, optional
        Whether to show images side by side, ``'horizontal'``, or one above
        the other, ``'vertical'``.
    """
    new_shape1 = list(img1.shape)
    new_shape2 = list(img2.shape)

    shape = img1.shape
    offset = np.array([shape[1], shape[0]])

    if alignment == 'horizontal':
        new_height = max(img1.shape[0], img2.shape[0])
        new_shape1[0] = new_height
        new_shape2[0] = new_height
        new_img1 = np.zeros(new_shape1, dtype=img1.dtype)
        new_img2 = np.zeros(new_shape2, dtype=img2.dtype)
        offset1_y = new_height - img1.shape[0]
        offset2_y = new_height - img2.shape[0]
        new_img1[offset1_y:offset1_y + img1.shape[0], :img1.shape[1]] = img1
        new_img2[offset2_y:offset2_y + img2.shape[0], :img2.shape[1]] = img2
        img = np.concatenate([new_img1, new_img2], axis=1)
        offset[1] = 0

    elif alignment == 'vertical':
        new_width = max(img1.shape[1], img2.shape[1])
        new_shape1[1] = new_width
        new_shape2[1] = new_width
        new_img1 = np.zeros(new_shape1, dtype=img1.dtype)
        new_img2 = np.zeros(new_shape2, dtype=img2.dtype)
        new_img1[:img1.shape[0], :img1.shape[1]] = img1
        new_img2[:img2.shape[0], :img2.shape[1]] = img2
        img = np.concatenate([new_img2, new_img1], axis=0)
        offset[0] = 0

    return img, offset


def rescaling(img, rfac=0.25):
    """ Return image with the rescaling_factor applied in the 2 dimensions """
    if rfac == 1.:
        return img
    else:
        shape = img.shape
        shape2 = (int(shape[0] * rfac), int(shape[1] * rfac))
        img2 = resize(img, shape2, order=0)  # rescale not working with rgb img
        return img2
