addpath(genpath('/home/tonglab/david/repos/mrTools-4.8.0.0'));
addpath(genpath('/home/tonglab/david/repos/analyzePRF'));
funcs = {'/mnt/NVMe2_1TB/p013_retinotopyTongLab/derivatives/pRF/sub-M015/ses-3T1/concatenated_before_prf/timeseries.nii.gz'};
mask = '/mnt/NVMe2_1TB/p013_retinotopyTongLab/derivatives/ROIs/sub-M015/ses-3T1/mask_analyzed.nii.gz';
outdir = '/mnt/NVMe2_1TB/p013_retinotopyTongLab/derivatives/pRF/sub-M015/ses-3T1/concatenated_before_prf';
analyzePRF_call(funcs, mask, outdir, 2, 1)