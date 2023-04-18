import os
import datetime
import matlab.engine
import sys

def run_NORDIC(subjects):

    for subject in subjects:
        for s, session in enumerate(subjects[subject]):
            funcdir = f"sub-{subject}/ses-{s + 1}/func"
            funcscans = sorted(glob.glob(f"sub-{subject}/ses-{s + 1}/func/*part-mag_bold.nii"))

            for funcscan in funcscans:

                print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | NORDIC preprocessing | '
                      f'Subject: {subject} | Session: {session} | Scan: {funcscan}')

                mag = funcscan
                phase = mag.replace('part-mag', 'part-phase')
                real = mag.replace('part-mag', 'part-real')
                imag = mag.replace('part-mag', 'part-imag')
                complex = mag.replace('part-mag', 'part-complex')

                # calculate phase
                if not op.isfile(phase):
                    # complex
                    print('calculating complex from real and imaginary...')
                    os.system(f'fslcomplex -complex {real} {imag} {complex}')

                    # phase
                    print('calculating phase from complex...')
                    os.system(f'fslcomplex -realphase {complex} {phase}')
                    os.system(f'fslcpgeom {imag} {phase}')  # get the original geometry header info back

                    # delete complex
                    os.remove(complex)

                # run NORDIC preprocessing
                outdir = f"derivatives/NORDIC/{funcdir}"
                os.makedirs(outdir, exist_ok=True)
                outpath = f'{outdir}/{os.path.basename(mag)}'
                if not os.path.isfile(outpath):
                    print('running NORDIC preprocessing...')
                    arg = {'noise_volume_last': 1,
                           'phase_filter_width': 10.,  # must be float
                           'use_magn_for_gfactor': 1}  # WARNING: to disable, remove key from dict, do not set to zero (see script)
                    eng = matlab.engine.start_matlab()
                    eng.NIFTI_NORDIC(mag, phase, outpath, arg, nargout=0)
                    eng.quit()

                    # trim mag and NORDIC data and change TR in header
                    num_func_vols = int(os.popen(f'fslnvols {mag}').read()[0:-1])
                    print(f'trimming noise volume from preprocessed timeseries...')
                    os.system(f'fslroi {outpath} {outpath} 0 {num_func_vols - 1}')
                    os.system(f'fslmerge -tr {outpath} {outpath} 4.217')

            # make links to anat and fmap data
            otherdirs = glob.glob(f"sub-{subject}/ses-{s + 1}/*")
            otherdirs.remove(funcdir)
            for otherdir in otherdirs:
                outdir = f"derivatives/NORDIC/{otherdir}"
                if not op.exists(outdir):
                    os.system(f"ln -s {op.abspath(otherdir)} {outdir}")


if __name__ == "__main__":
    run_NORDIC('M015')

