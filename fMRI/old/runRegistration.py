#!/usr/bin/python
'''
runs all preprocessing and directory set up
requires a configured 'experiment.py' file and raw data unpacked into individual/subject/session/rawData
'''

import sys
import os
import os.path as op
import glob
import pickle
import itertools
import numpy as np
from scipy.io import loadmat
from argparse import Namespace
import matplotlib.pyplot as plt
import datetime
import matlab.engine
import shutil
import gzip

sys.path.append(f'{os.path.expanduser("~")}/david/masterScripts/fMRI')

def runRegistration(experiment, overwrite=False):

    for subject in ['M001','M015','M017']:#experiment['sessInfo']:
        for s, session in enumerate(experiment['sessInfo'][subject]):
    
            # dir info
            sessInfo = experiment['sessInfo'][subject][session]
            sessDir = f"{experiment['general']['dataDir']}/individual/{subject}/{session}"
            xformDir = f"{sessDir}/reg/transforms"
            os.makedirs(xformDir, exist_ok=True)
            
            ### T1 high res space to standard spaces ###
            fsSubjDir = os.popen('echo $SUBJECTS_DIR').read()[:-1]
            fsDir = sorted(glob.glob(f"{fsSubjDir}/{subject}*"))[-1]
            subjectFS = os.path.basename(fsDir)

            # register to MNI 1mm
            regFile = f'{fsDir}/mri/transforms/reg.mni152.1mm.lta'
            if not os.path.isfile(regFile): # no option to overwrite as these are reliable and slow script down
                print(f'Registering MNI 1mm brain to high res T1 space...')
                os.system(f'mni152reg --s {subjectFS} --1')
            linkFile = f"{xformDir}/MNI1mm_to_T1HighRes.lta"
            if not os.path.isfile(linkFile) or overwrite:
                os.system(f"ln -sf {regFile} {linkFile}")

            # register to MNI 2mm
            regFile = f'{fsDir}/mri/transforms/reg.mni152.2mm.lta'
            if not os.path.isfile(regFile): # no option to overwrite as these are reliable and slow script down
                print(f'Registering MNI 2mm brain to high res T1 space...')
                os.system(f'mni152reg --s {subjectFS}')
            linkFile = f"{xformDir}/MNI2mm_to_T1HighRes.lta"
            if not os.path.isfile(linkFile) or overwrite:
                os.system(f"ln -sf {regFile} {linkFile}")


            ### Functional spaces to T1 high res space ###

            # get list of unique spaces to register to
            regTypes = []
            scans = sessInfo['funcScans']
            for scan in scans:
                regType = experiment['design'][scan]['params']['regType']
                if regType not in regTypes:
                    regTypes.append(regType)

            # register each regType to anatomical
            for regType in ['funcHighRes']:#regTypes:
                regDir = f'{sessDir}/reg/{regType}'
                refFunc = f'{regDir}/refFunc.nii.gz'

                if not os.path.isfile(refFunc) or overwrite:

                    print('creating reference func image...')
                    os.makedirs(regDir, exist_ok=True)

                    # collate all runs with this reg type and use median as reference scan
                    runsAll = [] # unique idx across all runs of this reg type
                    runsScan = [] # unique idxs within each scan type
                    scansAll = [] # list of scan names
                    for scan in scans:
                        thisRegType = experiment['design'][scan]['params']['regType']
                        if thisRegType == regType:
                            runs = sessInfo['funcScans'][scan]
                            runsAll += (runs)
                            runsScan += list(np.arange(len(runs)))
                            scansAll += [scan] * len(runs)
                    medianIdx = np.argsort(runsAll)[len(runsAll)//2]
                    refRun = runsScan[medianIdx]
                    refScan = scansAll[medianIdx]

                    # use preprocessed version if possible
                    preproc = experiment['design'][refScan]['params']['preproc']
                    if preproc:
                        suffix = f"_{preproc}"
                    else:
                        suffix = ''

                    # motion correct timeseries
                    inPath = glob.glob(f'{sessDir}/functional/{refScan}/run{refRun+1:02}/preprocessing/timeseries_magnitude{suffix}.nii*')[0]
                    outPath = f"{regDir}/motionCorrected_timeseries.nii.gz"

                    os.system(f'mcflirt -in {inPath} -out {outPath}')

                    # get mean volume across time
                    inPath = outPath
                    outPath = f'{regDir}/refFunc.nii.gz'
                    os.system(f'fslmaths {inPath} -Tmean {outPath}')

                    # extract brain
                    inPath = outPath
                    os.system(f'bet {inPath} {refFunc[:-7]}_brain.nii.gz')

                    # register to anat
                    # WARNING - commented out by default to prevent accidental overwriting of manually corrected registrations
                    # uncomment to run registration
                    regPath = f"{xformDir}/{regType}_to_T1HighRes.lta"
                    #os.system(f'bbregister --s {subjectFS} --mov {refFunc} --int {sessDir}/reg/funcWholeBrain/refFunc.nii.gz --reg {regPath} --bold')

                    runCommand=1 # place debug stop here and check results using last line of output before continuing to next subject

                    # register to MNI
                    MNI2mmReg = f'{fsDir}/mri/transforms/reg.mni152.2mm.lta'
                    os.system(f'mri_concatenate_lta -invert2 {MNI2mmReg} {regPath} {xformDir}/MNI2mm_to_{regType}.lta')
                    

if __name__ == "__main__":
    os.chdir('/mnt/NVMe1_1TB/projects/p023_laminarFMRI')
    from v7.analysis.scripts.experiment import experiment
    overwrite = True
    runRegistration(experiment, overwrite)
