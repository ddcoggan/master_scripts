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
from b0process import b0process

def preprocess(experiment):

    for subject in ['F019']:#experiment['sessInfo']:
        for s, session in enumerate(experiment['sessInfo'][subject]):
    
            # dir info
            sessInfo = experiment['sessInfo'][subject][session]
            sessID = sessInfo['sessID']
            sessDir = f"{experiment['general']['dataDir']}/individual/{subject}/{session}"
            rawDir = os.path.join(sessDir, 'rawData')
            T1HighResDir = os.path.join(sessDir, 'reg', 'T1HighRes')
            os.makedirs(T1HighResDir, exist_ok=True)


            ### T1 HIGH RES PREPROCESSING ###
            
            # run segmentation if a new subject or if requested
            newSeg = experiment['general']['newSeg']
            fsSubjDir = os.popen('echo $SUBJECTS_DIR').read()[:-1]
            fsDirSearch = sorted(glob.glob(f"{fsSubjDir}/{subject}*"))
            if len(fsDirSearch) == 0:
                fsDir = f"{fsSubjDir}/{subject}"
                runSeg = True
            elif newSeg and os.path.basename(fsDirSearch[-1]) != f'{subject}_{session}':
                fsDir = f"{fsSubjDir}/{subject}_{session}"
                runSeg = True
            else:
                fsDir = fsDirSearch[-1]
                runSeg = False
            subjectFS = os.path.basename(fsDir)
    
            if runSeg:
                T1HighResScan = sessInfo['T1HighResScan']
                T1HighResSearch = glob.glob(f'{rawDir}/Tong_{sessID}.{T1HighResScan:02}*.T1_3D_TFE_iso0.70.01.nii')
                if len(T1HighResSearch) == 0:
                    T1HighResSearch = glob.glob(f'{rawDir}/Tong_{sessID}.{T1HighResScan:02}*.nii')
                assert len(T1HighResSearch) == 1
                T1HighResRaw = T1HighResSearch[0]
    
                # run bias correction
                T1HighResCorrected = f'{os.path.dirname(T1HighResRaw)}/m{os.path.basename(T1HighResRaw)}'
                if not os.path.isfile(T1HighResCorrected):
                    print('Running bias correction...')
                    eng = matlab.engine.start_matlab()
                    eng.correctBiasT1(T1HighResRaw, nargout=0)
                    eng.quit()

                # check voxel size. If sub 1mm, run high-res segmentation, else run standard segmentation
                T1info = os.popen(f'fslinfo {T1HighResCorrected}').readlines()
                pixDims = [float(x.split()[1]) for x in T1info[6:9]]

                print('Running surface segmentation...')
                if np.prod(pixDims) < .95:
                    os.system(f'recon-all -hires -subject {subjectFS} -all -i {T1HighResCorrected} -expert /home/tonglab/david/masterScripts/fMRI/reconAllHighRes.opt')
                else:
                    os.system(f'recon-all -subject {subjectFS} -all -i {T1HighResCorrected}')

            # convert mgz to nifti
            T1HighResMGZ = f'{fsDir}/mri/T1.mgz'
            T1HighResNII = f'{fsDir}/mri/T1HighRes.nii'
            if os.path.isfile(T1HighResMGZ) and not os.path.isfile(T1HighResNII):
                os.system(f'mri_convert {T1HighResMGZ} {T1HighResNII}')
    
            # extract brain
            T1HighResNIIbrain = f'{T1HighResNII[:-4]}_brain.nii.gz'
            if not os.path.isfile(T1HighResNIIbrain):
                os.system(f'bet {T1HighResNII} {T1HighResNIIbrain}')
    
            # put links to files in session dir for convenience
            outT1HighRes = f'{T1HighResDir}/T1HighRes.nii'
            if not os.path.isfile(outT1HighRes):
                os.system(f'ln -s {T1HighResNII} {outT1HighRes}')
            outT1HighResBrain = f'{T1HighResDir}/T1HighRes_brain.nii.gz'
            if not os.path.isfile(outT1HighResBrain):
                os.system(f'ln -s {T1HighResNIIbrain} {outT1HighResBrain}')


            ### FIELD MAP PREPROCESSING ###
            
            # b0 field map
            if sessInfo['b0Scan']:  # only run if b0 map exists
    
                b0dir = f'{sessDir}/b0'
                os.makedirs(b0dir, exist_ok=True)
                b0processed = f'{b0dir}/b0_processed.nii.gz'
                if not os.path.isfile(b0processed):
    
                    magSearch = glob.glob(f'{rawDir}/*{sessID}.{sessInfo["b0Scan"]:02}*e1*.nii')
                    assert len(magSearch) == 1
                    magFile = magSearch[0]
    
                    realSearch = glob.glob(f'{rawDir}/*{sessID}.{sessInfo["b0Scan"]:02}*e2*.nii')
                    assert len(realSearch) == 1
                    realFile = realSearch[0]
    
                    b0process(b0dir, magFile, realFile)
    
            # funcNoEPI
            if sessInfo['funcNoEPIscan']:
                FNEdir = f'{rawDir}/funcNoEPI'
                os.makedirs(FNEdir, exist_ok=True)
                FNEsearch = glob.glob(f'{rawDir}/*{sessID}.{sessInfo["funcNoEPIscan"]:02}*func_noEPI*.nii')
                assert len(FNEsearch) == 1
                FNE = f'{FNEdir}/funcNoEPI.nii'
                if not os.path.isfile(FNE):
                    FNEdir = f'{sessDir}/funcNoEPI'
                    os.makedirs(FNEdir, exist_ok=True)
                    os.system(f'ln -s {op.abspath(FNEin)} {FNE}')


            ### FUNCTIONAL PREPROCESSING ###

            scans = sessInfo['funcScans']
            for scan in scans:
                runs = sessInfo['funcScans'][scan]
                for r, run in enumerate(runs):

                    print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | Preprocessing | '
                          f'Subject: {subject} | Session: {session} | Scan: {scan} | Run: {r + 1}')
                    funcDir = os.path.join(sessDir, 'functional', scan, f'run{r + 1:02}')
                    os.makedirs(os.path.join(funcDir, 'preprocessing'), exist_ok=True)

                    # make links to raw data
                    for dataType, suffix in zip(['magnitude', 'real', 'imaginary'],
                                                ['', '_real', '_imaginary']):
                        foundFiles = glob.glob(f'{rawDir}/*{sessID}.{run:02}.*.01{suffix}.nii')
                        assert (len(foundFiles) < 2)
                        if len(foundFiles) == 1:
                            inFile = foundFiles[0]
                            outFile = f'{funcDir}/preprocessing/timeseries_{dataType}_withNoise.nii'
                            if not os.path.isfile(outFile):
                                os.system(f'ln -s {os.getcwd()}/{inFile} {outFile}')

                    # run preprocessing
                    preproc = experiment['design'][scan]['params']['preproc']

                    if 'NORDIC' in preproc:

                        complex = f'{funcDir}/preprocessing/timeseries_complex_withNoise.nii.gz'
                        mag = f'{funcDir}/preprocessing/timeseries_magnitude_withNoise.nii'
                        phase = f'{funcDir}/preprocessing/timeseries_phase_withNoise.nii.gz'
                        real = f'{funcDir}/preprocessing/timeseries_real_withNoise.nii'
                        imag = f'{funcDir}/preprocessing/timeseries_imaginary_withNoise.nii'

                        # complex
                        if not os.path.isfile(complex):
                            print('calculating complex from real and imaginary...')
                            os.system(f'fslcomplex -complex {real} {imag} {complex}')

                        # phase
                        if not os.path.isfile(phase):
                            print('calculating phase from complex...')
                            os.system(f'fslcomplex -realphase {complex} {phase}')
                            os.system(f'fslcpgeom {imag} {phase}')  # get the original geometry header info back

                        # run NORDIC preprocessing
                        NORDIC = f'{funcDir}/preprocessing/timeseries_magnitude_NORDIC_withNoise.nii'
                        if not os.path.isfile(NORDIC):
                            print('running NORDIC...')
                            arg = {'noise_volume_last': 1,
                                   'phase_filter_width': 10.,  # must be float
                                   'use_magn_for_gfactor': 1}  # WARNING: to disable, remove key from dict, do not set to zero (see script)
                            eng = matlab.engine.start_matlab()
                            eng.NIFTI_NORDIC(mag, phase, NORDIC[:-4], arg, nargout=0)
                            eng.quit()

                        # trim mag and NORDIC data and change TR in header
                        nFuncVols = int(os.popen(f'fslnvols {mag}').read()[0:-1]) - 1
                        for dataType in ['magnitude', 'magnitude_NORDIC']:
                            inFile = f'{funcDir}/preprocessing/timeseries_{dataType}_withNoise'
                            outFile = f'{funcDir}/preprocessing/timeseries_{dataType}'
                            if len(glob.glob(f'{outFile}.nii*')) == 0:
                                print(f'trimming noise volume from {dataType} timeseries...')
                                os.system(f'fslroi {inFile} {outFile} 0 {nFuncVols}')
                                os.system(f'fslmerge -tr {outFile} {outFile} 4.217')
                        outFile = NORDIC

                    if 'topup' in preproc:

                        # link to opposite PE image
                        foundFiles = glob.glob(f'{rawDir}/*{sessID}.{run + 1:02}*.nii')
                        assert (len(foundFiles) == 1)
                        inFile = foundFiles[0]
                        oppPE = f'{funcDir}/preprocessing/oppPE.nii'
                        if not os.path.exists(oppPE):
                            os.system(f'ln -s {op.abspath(inFile)} {oppPE}')

                        # apply Top Up
                        inFile = outFile
                        outFile = f"{inFile.split('.')[0]}_topup.nii.gz"  # set final filename for top up output
                        if not os.path.isfile(outFile):
                            print('Running Top Up...')
                            runTopUp(timeseries, oppPE, 90, 270)

                    if 'b0' in preproc:

                        inFile = outFile
                        outFile = f"{inFile.split('.')[0]}_b0.nii.gz"  # set final filename for b0 output

                        if op.isfile(inFile) and not op.isfile(outFile):
                            print(f'b0 correcting scan {withWithout} topup')


if __name__ == "__main__":
    os.chdir('/mnt/NVMe1_1TB/projects/p023_laminarFMRI')
    from v7.analysis.scripts.experiment import experiment
    preprocess(experiment)
