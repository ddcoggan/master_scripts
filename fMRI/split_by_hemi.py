import os, glob
import nibabel as nib
import numpy as np

def split_by_hemi(input_path='None'):
    input_dir = os.path.dirname(input_path)
    input_name = os.path.basename(input_path).split(sep='.')[0]

    nii = nib.load(input_path)
    dat = nii.get_fdata()
    xdim = dat.shape[0]

    if xdim % 2 == 1:
        xdim = xdim - 1

    # Split off left
    left_dat = np.zeros(dat.shape)

    left_dat[int(xdim / 2):, :, :] = dat[int(xdim / 2):, :, :]

    # Split off right
    right_dat = np.zeros(dat.shape)
    right_dat[:int(xdim / 2), :, :] = dat[:int(xdim / 2), :, :]

    # Save
    nib.Nifti1Image(left_dat, nii.affine, nii.header).to_filename(os.path.join(input_dir, input_name + '_lh.nii.gz'))
    nib.Nifti1Image(right_dat, nii.affine, nii.header).to_filename(os.path.join(input_dir, input_name + '_rh.nii.gz'))

