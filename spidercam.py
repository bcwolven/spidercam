#!/usr/local/bin/python3

import os
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Convolution method?
# This was slow at best, and crashed on the high-res Moon pic for an
# authentically sized "spider FWHM."
# import scipy.ndimage as spnd
# self.imgSpdr[:,:,channel] = spnd.convolve(imgChan[:,:,channel],spiderPSF,
#                                           mode='nearest')

# This method is much faster, with no crashing for a realistic kernel size
import scipy.signal as spsg

thisCode = os.path.realpath(__file__)
projRoot = os.path.dirname(thisCode)


class arachnivision():
    """spidercam, spidercam, sees the sky like a spider can."""
    def __init__(self):
        self.imgOrig = None
        self.imgDimX = 0
        self.imgDimY = 0
        self.numChan = 0
        self.numFWHM = 5.       # Size of 2-D kernel in FWHM (+/- numFWHM/2)
        self.peopleAngRes = 0.
        self.spiderAngRes = 0.
        self.sourceScale  = 0.
        self.spiderVisRespFile = \
            "Habronattus_pyrrithrix_Photoreceptor_absorbance.csv"

    def _setupFigure(self,figID):
        # Is this scaling because of a matplotlib convention or did I just
        # happen to use a 100 DPI image for testing? TBD - todo
        self.figwid = 0.01*self.imgDimX
        self.figrat = self.imgDimY/self.imgDimX
        plt.figure(figID,figsize=(self.figwid,self.figwid*self.figrat))
        plt.subplots_adjust(left=0.000,right=1.000,top=1.000,bottom=0.000)
        plt.axes().set_xticks([])
        plt.axes().set_yticks([])

    def _makeGaussian(self,size,fwhm=3,center=None):
        """Make a square gaussian kernel (borrowed from stack overflow)
        - Size is the length of a side of the square.
        - Fwhm is full-width-half-maximum, which can be thought of as an
          effective radius.
        NOTE1 - kernel now normalized - was previously scaled so range was 0->1
        NOTE2 - There's probably a package function for this somewhere already
        """
        # x = np.arange(0, size, 1, float)
        x = np.arange(0,size,1,dtype=np.longdouble)
        y = x[:,np.newaxis]
        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]
        kernel = np.exp(-4.*np.log(2.) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
        kernel /= np.sum(kernel)
        return kernel

    def setSourcePlateScale(self,degreesPerPixel):
        """Set value for degrees per pixel in the input/source image"""
        self.sourceScale = degreesPerPixel

    def setPeopleAngularResolution(self,fwhmInDegrees):
        """Set value for FWHM of PSF in degrees, assumed to be Gaussian"""
        self.peopleAngRes = fwhmInDegrees

    def setSpiderAngularResolution(self,fwhmInDegrees):
        """Set value for FWHM of PSF in degrees, assumed to be Gaussian"""
        self.spiderAngRes = fwhmInDegrees

    def loadSpiderData(self):
        csvFile = os.path.join(projRoot,self.spiderVisRespFile)

# Reads data but indexing of resulting 2-D array does not work as expected?
#       import csv
#       with open(csvFile,'rU') as csviter:
#           csvRows = csv.reader(csviter)
#           respData = []
#           for rr,row in enumerate(csvRows):
#               if rr == 0:
#                   columnHeaders = row
#               else:
#                   respData.append([float(xx) for xx in row])
#       respData = np.array(respData)
#       print(len(columnHeaders),len(respData),np.shape(respData))
#       print(respData[0])
#       print(respData[-1])
#       print(respData[0][0],respData[0][1],respData[0][2],respData[0][3])
#       print([respData[0:10]][0])
#       respData = np.reshape(respData)
#       import sys
#       sys.exit()

        respData = np.genfromtxt(csvFile,dtype=float,delimiter=',',names=True)
        colmName = respData.dtype.names
        print("Read file: %s" % self.spiderVisRespFile)
        print("Extracted columns:")
        for header in colmName:
            print(header)
        plt.figure('spiderVisResp')
        plt.axes().set_title(self.spiderVisRespFile)
        plt.axes().set_xlabel('Wavelength (nm)')
        plt.axes().set_ylabel('Normalized Photoreceptor Absorbance')
        plt.grid(True)
        plt.plot(respData[colmName[0]][:],respData[colmName[1]][:],color='b',
                 label=colmName[1])
        plt.plot(respData[colmName[0]][:],respData[colmName[2]][:],color='g',
                 label=colmName[2])
        plt.plot(respData[colmName[0]][:],respData[colmName[3]][:],color='r',
                 label=colmName[3])
        plt.legend(loc='lower center',fontsize=6)
        plt.savefig(os.path.join(projRoot,"photoreceptor-absorbance.png"))
        # plt.clf()
        # plt.cla()

    def loadSourceImage(self,srcImg):
        """Load source image and set dimensions. Assuming color channels are in
        last dimension at the moment."""
        self.srcImg  = srcImg      # File basename, without full path
        self.imgOrig = mpimg.imread(os.path.join(projRoot,srcImg))
        imgDims      = np.shape(self.imgOrig)
        self.imgDimX = imgDims[1]  # Yeah, this isn't IDL, deal with it
        self.imgDimY = imgDims[0]  # Yeah, this still isn't IDL, deal with it
        self.numChan = imgDims[2]
        print("Loaded source image: %s" % self.srcImg)

    def sourceToEyeball(self):
        """Take a source image and 1) convolve with 0.02º FWHM Gaussian PSF to
        estimate what people would see with the naked eye, and 2) convolve with
        0.07º FWHM Gaussian PSF and modify the color balance to try and
        replicate what a jumping spider might see if it gazed up at the night
        sky."""

        imgChan = self.imgOrig.astype('float64')/255.  # Rescale 0-255 -> 0.-1.
        self.imgPepl = np.empty_like(imgChan)    # Store convolved version here
        self.imgSpdr = np.empty_like(imgChan)    # Store convolved version here

        # Make a 2-D Gaussian kernel for people and spider eye PSFs.
        # FwHM and corresponding kernel size are image dependent, set by angular
        # resolution of the particular critter's visual system and the plate
        # scale (degrees per pixel here) of the image. The plate scale and
        # visual angular reolutions are assumed to be the
        # same in both dimensions at present.

        peopleFWHM = self.peopleAngRes/self.sourceScale
        peopleSize = np.int(self.numFWHM*peopleFWHM)  # Extend kernel to N FWHM
        peoplePSF  = self._makeGaussian(peopleSize,fwhm=peopleFWHM)
#       peoplePSF /= np.sum(peoplePSF)            # Normalize kernel... or else

        spiderFWHM = self.spiderAngRes/self.sourceScale
        spiderSize = np.int(self.numFWHM*spiderFWHM)  # Extend kernel to N FWHM
        spiderPSF  = self._makeGaussian(spiderSize,fwhm=spiderFWHM)
#       spiderPSF /= np.sum(spiderPSF)            # Normalize kernel... or else

        # Do people-eye convolution, using original color channel weighting.
        for channel in range(self.numChan):
            self.imgPepl[:,:,channel] = spsg.fftconvolve(imgChan[:,:,channel],
                                                      peoplePSF,mode='same')

        # Tweak color balance for spider version - just an utter SWAG right now.
        # Eventually this ought to be its own method, relying on the spectral
        # information of the source image and the spectral response of the
        # critter visual system.
        imgChan[:,:,0] *= 0.85  # Red
        imgChan[:,:,1] *= 1.00  # Green
        imgChan[:,:,2] *= 0.85  # Blue

        # Do spider eye convolution, using modified color channel weighting.
        for channel in range(self.numChan):
            self.imgSpdr[:,:,channel] = spsg.fftconvolve(imgChan[:,:,channel],
                                                      spiderPSF,mode='same')

    def saveSourceImage(self):
        self._setupFigure('source')
        plt.imshow(jumper.imgOrig)
        print("Saving unaltered version.")
        plt.savefig(os.path.join(projRoot,"source-"+self.srcImg))

    def savePeopleImage(self):
        self._setupFigure('people')
        plt.imshow(jumper.imgPepl)
        print("Saving people/naked eye version.")
        plt.savefig(os.path.join(projRoot,"people-"+self.srcImg))

    def saveSpiderImage(self):
        self._setupFigure('spider')
        plt.imshow(jumper.imgSpdr)
        print("Saving spider-eyes-ed version.")
        plt.savefig(os.path.join(projRoot,"spider-"+self.srcImg))


if __name__ == "__main__":

    # Use argparse to... parse args
    parser = argparse.ArgumentParser(description="Simulate what a jumping "
                                     "spider might see when they look at an "
                                     "object in the night sky.")
    parser.add_argument("-i","--image",required=False,
                        default="20141008tleBaldridge001.jpg",
                        help="Source image")
#                       default="beletskYairglow_pano.jpg",
    # 2250 pixels for moon diameter of ~0.5 degrees.
    parser.add_argument("-d","--plate-scale",required=False,type=float,
                        default=2.222e-04,help="Plate scale of source image - "
                        "For default image is 2.222e-04 degrees/pixel")
    parser.add_argument("-p","--people-resolution",required=False,type=float,
                        default=0.007,help="Resolution to use for human eye - "
                        "default is foveal resolution of 0.007 degrees")
    parser.add_argument("-s","--spider-resolution",required=False,type=float,
                        default=0.070,help="Resolution to use for spider eye - "
                        "default is resolution of 0.07 degrees")

    # Process arguments - no screening for valid inputs done here beyond what
    # argparse does internally right now.
    args   = parser.parse_args()
    srcImg = args.image  # relative path from directory containing this file

    # Create instance of class to load and manipulate image.
    jumper = arachnivision()

    # Set plate scale (degrees/pixel) of the source image.
    jumper.setSourcePlateScale(args.plate_scale)

    # Set the visual angular resolution of the two critters in question -
    # "People" and "Spider" ony at the moment. Perhaps make more general later?
    jumper.setPeopleAngularResolution(args.people_resolution)
    jumper.setSpiderAngularResolution(args.spider_resolution)

    # Load spider photoreceptor absorbance curves
    jumper.loadSpiderData()

    # Load source image
    jumper.loadSourceImage(srcImg)

    # Save copy of original with "source" stuck at front of name - have we done
    # any violence to it unintentionally in loading and saving? Sanity check...
    jumper.saveSourceImage()

    # Modify it to something resembling what spider would see?
    jumper.sourceToEyeball()

    # Save convolved version of original with "people" stuck at front of name.
    # This is identical to the original in terms of color balance, but uses a
    # people-vision-specific angular resolution.
    jumper.savePeopleImage()

    # Save "spider-eyes-ed" version with "spider" stuck at front of name. This
    # is different from the original in terms of both color balance and the fact
    # that it uses a spider-vision-specific angular resolution.
    jumper.saveSpiderImage()

# Miscellaneous discussion notes and source Tweet references:
#
# Jumping spider vision angular resolution quoted to be ~0.07 degrees. Wikipedia
# quotes value for typical human eye to be 0.02 degrees, so only about 3.5 times
# better!  But *foveal* resolution is closer to 0.007º or 10 times better.
# Perhaps Wikipedia value is for rods rather than cones? The foveal area of the
# retina sees a swath ~2º wide, located at center of retina.

# "So did some back of envelope calcs. The jumping spider in our avatar
# (Habronattus pyrrithrix) can def see the moon, maybe even craters…"
# https://twitter.com/MorehouseLab/status/872081983819612161

# "Moon diameter is 9.22x10^-3 radians, or ~0.53 deg visual angle. H.
# pyrrithrix can resolve objects up to 0.07 deg visual angle."
# https://twitter.com/MorehouseLab/status/872082579217887232
