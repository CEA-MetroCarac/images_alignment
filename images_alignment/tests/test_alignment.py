import pytest
from pytest import approx
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import rotate
from skimage.io import imsave, imread

from images_alignment import ImagesAlign
from images_alignment.examples.utils import images_generation
from images_alignment.examples.utils import ROIS


@pytest.fixture
def create_object():
    def _create_object(tmp_path):
        input_dirname = tmp_path / 'inputs'
        output_dirname = tmp_path / 'results'
        input_dirname.mkdir(exist_ok=True)
        output_dirname.mkdir(exist_ok=True)

        # fixed image (high resolution squared image)
        img1 = shepp_logan_phantom()
        imsave(input_dirname / 'img1.tif', img1)

        # moving images (low resolution rectangular images)
        img2 = shepp_logan_phantom()[::4, ::4]  # low image resolution
        y, x = np.meshgrid(np.arange(img2.shape[0]), np.arange(img2.shape[1]))
        for k in range(3):
            img2_ = img2.copy()
            radius = 2 + k
            mask = (x - 70) ** 2 + (y - 70) ** 2 < radius ** 2
            img2_[mask] = 1.
            img2_ = rotate(img2_, 10, center=(40, 60), cval=0)  # rotation
            img2_ = np.pad(img2_, ((40, 0), (60, 0)))  # padding
            imsave(input_dirname / f'img2_{k + 1}.tif', img2_)

        imgalign = ImagesAlign(fnames_fixed=[input_dirname / 'img1.tif'],
                               fnames_moving=[input_dirname / 'img2_1.tif',
                                              input_dirname / 'img2_2.tif',
                                              input_dirname / 'img2_3.tif'])
        return imgalign, output_dirname

    return _create_object


def test_registration(create_object, tmp_path):
    imgalign, _ = create_object(tmp_path)

    registration_model = 'StackReg'
    imgalign.registration(registration_model=registration_model)
    assert imgalign.results[registration_model]['score'] == approx(72.0610634411937)
    assert imgalign.tmat == approx(np.array([[1.01297237, 0.03126468, 17.07876408],
                                             [0.3250302, 0.79615517, 4.38526333],
                                             [0., 0., 1.]]))
    assert np.nansum(imgalign.img_reg) == approx(11196.403461189417)

    registration_model = 'SIFT'
    imgalign.registration(registration_model=registration_model)
    assert imgalign.results[registration_model]['score'] == approx(97.32952636882769)
    assert imgalign.tmat == approx(np.array([[2.49758879e-01, 4.15468237e-02, 5.00321264e+01],
                                             [-4.21972871e-02, 2.45555432e-01, 4.81580265e+01],
                                             [0., 0., 1.]]))
    assert np.nansum(imgalign.img_reg) == approx(19787.42383715999)

    registration_model = 'SIFT + StackReg'
    imgalign.registration(registration_model=registration_model)
    assert imgalign.results[registration_model]['score'] == approx(96.92460447068967)
    assert imgalign.tmat == approx(np.array([[2.55172779e-01, 4.43017681e-02, 4.82570457e+01],
                                             [-4.45469168e-02, 2.52658559e-01, 4.68055973e+01],
                                             [0., 0., 1.]]))
    assert np.nansum(imgalign.img_reg) == approx(18786.13305862431)


def test_apply_to_all(create_object, tmp_path):
    imgalign, output_dirname = create_object(tmp_path)

    imgalign.apply_to_all(dirname_res=output_dirname)

    arr1 = imread(output_dirname / 'moving_images' / 'img2_1.tif')
    arr2 = imread(output_dirname / 'moving_images' / 'img2_2.tif')
    arr3 = imread(output_dirname / 'moving_images' / 'img2_3.tif')

    assert np.nanmean(arr1) == approx(0.07716601854777502)
    assert np.nanmean(arr2) == approx(0.07795758567503669)
    assert np.nanmean(arr3) == approx(0.07894589554264218)


def test_roi(tmp_path):
    fnames_fixed, fnames_moving = images_generation(tmp_path, 'camera')

    imgalign = ImagesAlign(fnames_fixed=fnames_fixed,
                           fnames_moving=fnames_moving,
                           rois=ROIS['camera'])

    registration_model = 'StackReg'
    imgalign.registration(registration_model=registration_model)

    assert imgalign.results[registration_model]['score'] == approx(95.77442094662638)
    assert imgalign.tmat == approx(np.array([[0.77767435, 0.43040319, -24.41735391],
                                             [-0.63508967, 1.16203181, 52.11186476],
                                             [0., 0., 1.]]))
    assert np.nansum(imgalign.img_reg) == approx(2710033.1827184735)


def test_threshold(tmp_path):
    fnames_fixed, fnames_moving = images_generation(tmp_path, 'camera')

    imgalign = ImagesAlign(fnames_fixed=fnames_fixed,
                           fnames_moving=fnames_moving,
                           rois=ROIS['camera'],
                           thresholds=[0.6, 0.7])

    registration_model = 'StackReg'
    imgalign.registration(registration_model=registration_model)

    assert imgalign.results[registration_model]['score'] == approx(91.12472251911535)
    assert imgalign.tmat == approx(np.array([[0.78909822, 0.43127827, -25.70185029],
                                             [-0.64060665, 1.19538331, 48.72397305],
                                             [0., 0., 1.]]))
    assert np.nansum(imgalign.img_reg) == approx(2670264.833361759)
