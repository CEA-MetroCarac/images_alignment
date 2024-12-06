Scripting
---------

Every operation performed through the GUI can also be executed in scripting mode.

Below is a brief illustration of image alignment by adjusting a few parameters::

    from images_alignment import ImagesAlign

    imgalign = ImagesAlign(fixed_images=['img1.tif'], fnames_moving=['img2_1.tif', 'img2_2.tif', 'img2_3.tif', ...])

    imgalign.angles = [0, 90]
    imgalign.thresholds = [0.50, 0.65]
    imgalign.rois = [[130, 330, 290, 460], [5, 80, 130, 210]]
    imgalign.registration_model = 'StackReg'

    imgalign.apply_to_all(dirname_res='./results')
