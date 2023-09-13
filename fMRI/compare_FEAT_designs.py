import nibabel as nib
import os
from tqdm import tqdm

def compare_FEAT_designs(design1, design2):

    d1 = open(design1, 'r').readlines()
    d2 = open(design2, 'r').readlines()

    print(f'lines only in first design')
    for line in d1:
        if not line.startswith('#') and line not in d2:
            print(line)

    print(f'lines only in second design')
    for line in d2:
        if not line.startswith('#') and line not in d1:
            print(line)

if __name__ == "__main__":
    design1 = '/mnt/HDD2_16TB/projects/p022_occlusion/in_vivo/fMRI/exp1/derivatives/FEAT/sub-M020/ses-1/task-occlusion/run-1.feat/design.fsf'
    design2 = ('/mnt/HDD2_16TB/projects/p022_occlusion/in_vivo/fMRI/exp1'
               '/derivatives/FEAT/sub-M020/ses-1/task-occlusion/run-1.feat'
               '/design2.fsf')
    compare_FEAT_designs(design1, design2)


