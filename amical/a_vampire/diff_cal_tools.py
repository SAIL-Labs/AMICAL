import os
import amical
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy import io
from matplotlib.colors import PowerNorm
import pylab as plt
from scipy import ndimage
import math
import h5py as h5py
from astropy.io import fits

plt.ion()


def makeVampDarks(dataPath, filePref, nSubFiles, saveFilePref, camera1dark_path, camera2dark_path):

    fileExtn = '.fits'

    HAFileformat = False
    startFileNum = 0
    useMedian = False  # Take the median of all files, not the mean
    useMedianWithinCube = False  # Take the median of all frames, not the mean
    showSEMMap = False
    saveData = True

    plt.figure()
    if HAFileformat:
        # Filter arrangement, listed as [cam1, cam2]
        state1 = ('cont', 'Ha')
        state2 = ('Ha', 'cont')

        # Read a FITS file to get sizes
        curFilenumStr = '%d' % startFileNum
        curCamStr = '_cam1'
        curStateStr = 'state1'
        curFilename = dataPath + filePref + curStateStr + fileSuf + curFilenumStr + curCamStr + fileExtn
        print(curFilename)
        hdulist = fits.open(curFilename)
        curHDU = hdulist[0]
        curCube = np.transpose(curHDU.data)
        nFrms = curCube.shape[2]
        dim = curCube.shape[0]

        # Indexes are [:, :, Set+State, Channel (camera)]
        allSummedIms = np.zeros([dim, dim, nSets * 2, 2])

        curSetState = 0
        for f in range(0, nSets):
            curFileNum = f
            curFilenumStr = '%d' % curFileNum

            for s in range(0, 2):
                curState = s
                if curState == 0:
                    curStateStr = 'state1'
                else:
                    curStateStr = 'state2'

                for c in range(0, 2):
                    curChan = c
                    if curChan == 0:
                        curCamStr = '_cam1'
                    else:
                        curCamStr = '_cam2'

                    curFilename = dataPath + filePref + curStateStr + fileSuf + curFilenumStr \
                                  + curCamStr + fileExtn
                    #print('Reading file %s' % curFilename)
                    hdulist = fits.open(curFilename)
                    curHDU = hdulist[0]
                    curCube = np.transpose(curHDU.data)
                    goodframes = curCube[:, :, 2:nFrms]  # Discard 1st 2 frames
                    curDark = np.mean(goodframes, axis=2)
                    allSummedIms[:, :, curSetState, curChan] = curDark

                    plt.clf()
                    plt.imshow(curDark)
                    plt.colorbar()
                    plt.pause(0.001)

                curSetState = curSetState + 1


    else:
        # Read a FITS file to get sizes
        curFilenumStr = '%d' % startFileNum
        curCamStr = '_cam1'
        curFilename = dataPath + filePref + curFilenumStr + curCamStr + fileExtn
        hdulist = fits.open(curFilename)
        curHDU = hdulist[0]
        curCube = np.transpose(curHDU.data)
        nFrms = curCube.shape[2]
        dim = curCube.shape[0]

        allSummedIms = np.zeros([dim, dim, nSubFiles // 2, 2])
        curSet = 0
        for f in range(startFileNum, nSubFiles // 2 + startFileNum):
            curFileNum = f

            for c in range(0, 2):
                curChan = c

                # Generate current filename
                if curChan == 0:
                    curCamStr = '_cam1'
                else:
                    curCamStr = '_cam2'

                curFilenumStr = '%d' % curFileNum
                curFilename = dataPath + filePref + curFilenumStr + curCamStr + fileExtn

                #print('Reading file %s' % curFilename)
                hdulist = fits.open(curFilename)
                curHDU = hdulist[0]
                curSuperCube = np.transpose(curHDU.data)
                goodframes = curSuperCube[:, :, 2:nFrms]  # Discard 1st 2 frames
                if useMedianWithinCube:
                    curDark = np.median(goodframes, axis=2)
                else:
                    curDark = np.mean(goodframes, axis=2)
                allSummedIms[:, :, curSet, curChan] = curDark

                if curChan == 1:
                    curSet = curSet + 1

    if useMedian:
        finalDarks = np.median(allSummedIms, axis=2)
    else:
        finalDarks = np.mean(allSummedIms, axis=2)
    sigmaDarks = np.std(allSummedIms, axis=2) / np.sqrt(nFrms - 1)

    if showSEMMap:
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(finalDarks[:, :, 0])
        plt.colorbar()
        plt.title('Camera 1 Dark')
        plt.subplot(2, 2, 2)
        plt.imshow(finalDarks[:, :, 1])
        plt.colorbar()
        plt.title('Camera 2 Dark')

        plt.subplot(2, 2, 3)
        plt.imshow(sigmaDarks[:, :, 0])
        plt.colorbar()
        plt.title('Camera 1 SEM map')
        plt.subplot(2, 2, 4)
        plt.imshow(sigmaDarks[:, :, 1])
        plt.colorbar()
        plt.title('Camera 2 SEM map')

    else:
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(finalDarks[:, :, 0])
        plt.colorbar()
        plt.title('Camera 1 Dark')
        plt.subplot(1, 2, 2)
        plt.imshow(finalDarks[:, :, 1])
        plt.colorbar()
        plt.title('Camera 2 Dark')

    med_ch1 = np.median(finalDarks[:, :, 0])
    mean_ch1 = np.mean(finalDarks[:, :, 0])
    med_ch2 = np.median(finalDarks[:, :, 1])
    mean_ch2 = np.mean(finalDarks[:, :, 1])

    # print(' ')
    # print('Channel 1: median = %f, mean = %f' % (med_ch1, mean_ch1))
    # print('Channel 2: median = %f, mean = %f' % (med_ch2, mean_ch2))
    # print(' ')

    sdDarks = np.std(allSummedIms, axis=2) / np.sqrt(nFrms - 1)
    avSd1 = np.mean(sdDarks[:, :, 0])
    avSd2 = np.mean(sdDarks[:, :, 1])
    # print('Average pixel s.d. for camera 1: %f' % avSd1)
    # print('Average pixel s.d. for camera 2: %f' % avSd2)

    if saveData:

        saveFilename = saveFilePref + '/' + filePref
        print(saveFilename)
        np.savez(saveFilename, allSummedDarks=allSummedIms, finalDarks=finalDarks, sigmaDarks=sigmaDarks)

        npzfile = np.load(saveFilename + '.npz')
        camera1dark = npzfile.f.finalDarks[:, :, 0]
        hdu1 = fits.PrimaryHDU(camera1dark)
        hdu1_l = fits.HDUList([hdu1])
        hdu1_l.writeto(camera1dark_path)

        camera2dark = npzfile.f.finalDarks[:, :, 1]
        hdu2 = fits.PrimaryHDU(camera2dark)
        hdu2_l = fits.HDUList([hdu2])
        hdu2_l.writeto(camera2dark_path)


    return
