"""
Requires docker, as installing neuropythy natively causes dependency conflicts with other scripts
"""
import os
import os.path as op
import glob
import json

def get_wang_atlas(subject):

    fs_dir = os.getenv('SUBJECTS_DIR')

    neuropythy_params = {"freesurfer_subject_paths": fs_dir,
                         "data_cache_root": "~/Temp/npythy_cache",
                         "hcp_subject_paths": "/Volumes/server/Projects/HCP/subjects",
                         "hcp_auto_download": True,
                         "hcp_credentials": "~/.hcp-passwd"}
    npythyrc_path = f'{fs_dir}/.npythyrc'
    json.dump(neuropythy_params, open(npythyrc_path, 'w+'))

    # Get atlas as volume

    if len(glob.glob(f"{fs_dir}/{subject}/surf/??.wang15_mplbl.mgz")) < 2:
        os.system(f'docker run --rm ' \
                  f'--mount type=bind,src={fs_dir},dst=/subjects ' \
                  f'--env "NPYTHYRC=/subjects/.npythyrc" ' \
                  f'nben/neuropythy ' \
                  f'atlas --verbose {subject}')
    
    for hemi in ['lh', 'rh']:
        
        # Convert to labels for all regions
        vol_file = f"{fs_dir}/{subject}/surf/{hemi}.wang15_mplbl.mgz"
        outpath = f"{fs_dir}/{subject}/surf/{hemi}.wang15_mplbl.label"
        os.system(f"mri_cor2label --i {vol_file} --stat --l {outpath} --surf {subject} {hemi} inflated")


        roiname_array = ("V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "VO1", "VO2", "PHC1", "PHC2", \
                         "TO2", "TO1", "LO2", "LO1", "V3B", "V3A", "IPS0", "IPS1", "IPS2", "IPS3", "IPS4", \
                         "IPS5", "SPL1", "FEF")
        for r, roiname in enumerate(roiname_array):

            # Convert to label
            label = f"{fs_dir}/{subject}/surf/{hemi}.wang15_mplbl.{roiname}.label"
            if not op.isfile(label):
                os.system(f"mri_cor2label --i {vol_file} --id {r + 1} --l {outpath} --surf {subject} {hemi} inflated")

            """
            # convert to nifti in anatomical space
            nifti = f"{fs_dir}/{subject}/mri/{hemi}.wang15_mplbl.{roiname}.anat.nii"
            if not op.isfile(nifti):
                os.system(f"mri_label2vol --label {label} ---temp {fs_dir}/{subject}/mri/T1.nii' --identity --o {nifti} --subject {subject} --hemi {hemi}")
            """

if __name__ == "__main__":
    get_wang_atlas(f'sub-F016')


