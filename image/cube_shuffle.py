import os
import glob
from PIL import Image
import random
import math
import numpy as np
import datetime

gridSize = 8 # x by x grid
nCubes = gridSize**2
datasetDir = '/home/dave/Datasets/imagenet16'
newDatasetDir = datasetDir + '_cubeShuffle'
imagePaths = sorted(glob.glob(f'{datasetDir}/val/**/*.jpg'))
for i, imagePath in enumerate(imagePaths):
    print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | image {i+1}/{len(imagePaths)} {os.path.basename(imagePath)}')
    image = np.array(Image.open(imagePath).convert('RGB'))
    H, W, C = image.shape
    cubeSize = [H/gridSize, W/gridSize]
    cubeImage = Image.new(mode='RGB', size=(W, H), color=(127,127,127))
    cubeOrder = np.random.permutation(nCubes)
    for c, cube in enumerate(cubeOrder):
        origRow = math.floor(cube/gridSize)
        origCol = cube % gridSize
        rowStart = int(origRow*cubeSize[0])
        rowStop = int((origRow+1)*cubeSize[0])
        colStart = int(origCol * cubeSize[1])
        colStop = int((origCol + 1) * cubeSize[1])
        origCube = Image.fromarray(image[rowStart:rowStop+1, colStart:colStop+1, :])
        pasteRow = int(math.floor(c/gridSize) * cubeSize[1])
        pasteCol = int(c % gridSize * cubeSize[0])
        cubeImage.paste(origCube, (pasteRow, pasteCol))
    outDir = f'{newDatasetDir}/val/{imagePath.split("/")[-2]}'
    os.makedirs(outDir, exist_ok=True)
    outPath = f'{outDir}/{os.path.basename(imagePath)}'
    cubeImage.save(outPath)
