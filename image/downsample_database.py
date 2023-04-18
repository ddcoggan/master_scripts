# downsamples image database by ensuring that the smallest dimension of each image is no larger than x pixels.

import os, glob
from PIL import Image

def minDimCap(imagePath, outPath, minSize=512):

    image = Image.open(imagePath).convert('RGB')
    oldImSize = image.size
    oldMinSize = min(oldImSize)
    rescaleFactor = minSize/oldMinSize
    if rescaleFactor < 1:
        newImSize = [int(oldImSize[0]*rescaleFactor), int(oldImSize[1]*rescaleFactor)]
        outImage = image.resize(newImSize)
    else:
        outImage = image
    outImage.save(outPath)
    
if __name__ == "__main__":
    
    outDir = '/home/tonglab/Datasets/ILSVRC2012_512max'
    for trainval in ['train','val']:
        classDirs = sorted(glob.glob(f'/home/tonglab/Datasets/ILSVRC2012/{trainval}/*'))
        for c, classDir in enumerate(classDirs):
            classDirOut = f'{outDir}/{trainval}/{os.path.basename(classDir)}'
            os.makedirs(classDirOut, exist_ok=True)
            images = sorted(glob.glob(f'{classDir}/?*.?*'))
            for i, image in enumerate(images):
                imageOut = f'{classDirOut}/{os.path.basename(image)}'
                if not os.path.isfile(imageOut):
                    print(f'{trainval}, class {c+1}/{len(classDirs)}, image {i+1}/{len(images)}')
                    minDimCap(image, imageOut, minSize=512)

