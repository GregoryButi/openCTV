

import subprocess
import os
import shutil
import numpy as np
import nibabel as nib
import json

def run_segmentation(input_files, model, model_config):

    # define executables
    nnUNetscript_path = '../../Process/ImageSegmentation_nnUNet/BashScripts/run_nnUNet_inference.sh'
    nnUNet_directory = os.path.abspath(f'../../Models/nnUNet')

    # create temporary directory
    tmp_dir = 'tmp'
    os.mkdir(tmp_dir)

    # copy input files into temporary directory
    for file, i in zip(input_files, np.arange(len(input_files))):
        shutil.copyfile(file, os.path.join(tmp_dir, f'image_000{i+1}.nii.gz'))

    # run nnUNet inference
    subprocess.call([os.path.abspath(nnUNetscript_path), tmp_dir, model, nnUNet_directory, model_config])

    # load structures
    segmentations_obj = nib.load(os.path.join(tmp_dir, 'image.nii.gz'))
    with open(os.path.join(tmp_dir, 'dataset.json'), 'r') as f:
        dictionary = json.load(f)

    segmentations = segmentations_obj.get_fdata()
    affine = segmentations_obj.affine
    num_classes = len(dictionary['labels'])

    # convert and add to dictionary
    one_hot = np.eye(num_classes)[segmentations.astype(int)]
    dictionary['array'] = one_hot
    dictionary['affine'] = affine

    # delete temporary directory
    shutil.rmtree(tmp_dir)

    return dictionary