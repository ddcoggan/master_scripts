# /usr/bin/python
# Created by David Coggan on 2022 11 14
import os
import glob
fsdir = f'/media/tonglab/DAVID/spikiness_animacy/data/fMRI/freesurfer'

def deface_post_seg(subject):
    images_with_faces = [
        'nu', 'orig', 'orig_nu', 'rawavg', 'T1', 'orig/001'
    ]
    subjdir = f'{fsdir}/{subject}'
    for image in images_with_faces:
        for filetype in ['mgz','nii','nii.gz']:
            file = f"{subjdir}/mri/{image}.{filetype}"
            if os.path.isfile(file):
                print(f'defacing {file}')
                os.system(f'mideface --i {file} --o {file}')

if __name__ == "__main__":
    subjects = [os.path.basename(x) for x in glob.glob(f"{fsdir}/????")]
    for subject in subjects:
        deface_post_seg(subject)
