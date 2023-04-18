# splits database into train, val and test subsets

import os, glob
import os.path as op
import shutil
import numpy as np
indir = '/home/tonglab/Datasets/ShapeNetCore.v2_orig'
outdir = '/home/tonglab/Datasets/ShapeNetCore.v2'
ratios =  [.7,.1,.2] # train, val, test
classdirs = sorted(glob.glob(f'{indir}/????????'))

for subset in ['train','val','test']:
    for c, classdir in enumerate(classdirs):
        classdir_out = f'{outdir}/{subset}/{op.basename(classdir)}'
        os.makedirs(classdir_out, exist_ok=True)

for c, classdir in enumerate(classdirs):
    objects = sorted(glob.glob(f'{classdir}/*'))
    np.random.shuffle(objects)
    num_objects = len(objects)
    for o, obj in enumerate(objects):
        print(f"class {c+1}/{len(classdirs)}, object {o+1}/{num_objects}")
        if o < int(ratios[0]*num_objects):
            dest = f'{outdir}/train/{op.basename(classdir)}/{op.basename(obj)}'
            shutil.copytree(obj, dest)
        elif o < int(np.sum(ratios[:2])*num_objects):
            dest = f'{outdir}/val/{op.basename(classdir)}/{op.basename(obj)}'
            shutil.copytree(obj, dest)
        else:
            dest = f'{outdir}/test/{op.basename(classdir)}/{op.basename(obj)}'
            shutil.copytree(obj, dest)

