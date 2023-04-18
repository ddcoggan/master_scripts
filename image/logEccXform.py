import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
import glob
import os
import datetime

inDir = '/home/dave/Datasets/imagenet16'
outDir = '/home/dave/Datasets/imagenet16_logEcc'
os.makedirs(outDir, exist_ok=True)

Xout, Yout = (256,256)
SF = 4 # scaling factor scales the image up and down either size of log ecc transform to make the images smoother in the center
Xbig,Ybig = Xout*SF,Yout*SF
newCoords = np.empty(shape=(Xbig,Ybig,2))

for x in range(Xbig):
    for y in range(Ybig):

        # get eccentricity in pixel units, retaining direction
        eX = x - Xbig / 2
        eY = y - Ybig / 2
        ecc = np.sqrt((np.abs(eX) + 2) ** 2 + (np.abs(eY) + 2) ** 2)

        # log transform eccentricity
        logEcc = math.log(ecc) * 32 * SF

        # get transformed X,Y coords
        xNew = eX * ecc / logEcc + Xbig / 2
        yNew = eY * ecc / logEcc + Ybig / 2

        # clip range
        xNew = int(max(xNew, 0))
        xNew = int(min(xNew, Xbig - 1))
        yNew = int(max(yNew, 0))
        yNew = int(min(yNew, Ybig - 1))

        newCoords[x,y,0] = xNew
        newCoords[x,y,1] = yNew

P = 200
xformPlot = np.zeros(P)
for p in range(P):
    xformPlot[p] = math.log(p+1)

plt.plot(list(range(1, P+1)), xformPlot)
plt.xlabel('eccentricity')
plt.ylabel('log eccentricity')
plt.savefig(f'{outDir}/mapping.jpg')
plt.close()


def logEcc(imagePath, Xbig,Ybig, SF, newCoords, outImagePath): # input image

    imOrig = Image.open(imagePath).convert('RGB')
    imBig = imOrig.resize((Xbig,Ybig))
    imNew = np.zeros((Ybig, Xbig, 3)) # first 2 dims flipped for arrays
    imBigA = np.array(imBig)
    for x in range(Xbig):
        for y in range(Ybig):
            xNew, yNew = newCoords[x,y,:]
            imNew[y, x, :] = imBigA[int(yNew), int(xNew), :]
    outImage = Image.fromarray(imNew.astype('uint8'), 'RGB').resize((int(Xbig/SF),int(Ybig/SF))).rotate(0)
    outImage.resize((Xout,Yout))
    outImage.save(outImagePath)



for trainVal in ['train','val']:

    catDirs = sorted(glob.glob(os.path.join(inDir, trainVal, '*')))
    for c, catDir in enumerate(catDirs):

        outCatDir = os.path.join(outDir, trainVal, os.path.basename(catDir))
        os.makedirs(outCatDir, exist_ok=True)
        imagePaths = sorted(glob.glob(os.path.join(catDir, '*')))

        for i, imagePath in enumerate(imagePaths):

            print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")} | trainVal: {trainVal} | category: {c+1}/{len(catDirs)} | image: {i+1}/{len(imagePaths)}')
            outImagePath = os.path.join(outCatDir, os.path.basename(imagePath))
            logEcc(imagePath, Xbig, Ybig, SF, newCoords, outImagePath)




