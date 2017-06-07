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
        NOTE - is now normalized - was previously scaled so range was 0->1
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

    def loadSourceImage(self,srcImg):
        """Load source image and set dimensions. Assuming color channels are in
        last dimension at the moment."""
        self.imgOrig = mpimg.imread(os.path.join(projRoot,srcImg))
        imgDims      = np.shape(self.imgOrig)
        self.imgDimX = imgDims[1]  # Yeah, this isn't IDL, deal with it
        self.imgDimY = imgDims[0]  # Yeah, this still isn't IDL, deal with it
        self.numChan = imgDims[2]
        print("Loaded source image: %s" % srcImg)

    def sourceToEyeball(self):
        """Take a source image and 1) convolve with 0.02º FWHM Gaussian PSF to
        estimate what people would see with the naked eye, and 2) convolve with
        0.07º FWHM Gaussian PSF and modify the color balance to try and
        replicate what a jumping spider might see if it gazed up at the night
        sky."""

        # Adjust weight of color channels - just guessing at the moment
        imgChan = self.imgOrig.astype('float64')/255.  # Scale 0-255 -> 0.-1.
        self.imgPepl = np.empty_like(imgChan)  # Store convolved version here
        self.imgSpdr = np.empty_like(imgChan)  # Store convolved version here

        # Make a 2-D Gaussian kernel for people and spider eye PSFs.
        # FwHM and required size will be image dependent!!
        # Spider angular resolution quoted to be ~0.07
        # degrees. Wikipedia quotes value for human eye to be 0.02 degrees, so
        # only about 3.5 times better!

        # "So did some back of envelope calcs. The jumping spider in our avatar
        # (Habronattus pyrrithrix) can def see the moon, maybe even craters…"
        # https://twitter.com/MorehouseLab/status/872081983819612161

        # "Moon diameter is 9.22x10^-3 radians, or ~0.53 deg visual angle. H.
        # pyrrithrix can resolve objects up to 0.07 deg visual angle."
        # https://twitter.com/MorehouseLab/status/872082579217887232

        # 2250 pixels for moon diameter = 0.5 degrees. 0.02 degrees = 90 pixels
        peopleFWHM = 90.
        peopleFWHM = 1.5
        peopleSize = np.int(5*peopleFWHM)
        peoplePSF  = self._makeGaussian(peopleSize,fwhm=peopleFWHM)
        peoplePSF /= np.sum(peoplePSF)

        # 2250 pixels for moon diameter = 0.5 degrees. 0.07 degrees = 315 pixels
        spiderFWHM = peopleFWHM*3.5
        spiderSize = np.int(5*spiderFWHM)
        spiderPSF  = self._makeGaussian(spiderSize,fwhm=spiderFWHM)
        spiderPSF /= np.sum(spiderPSF)

        # Do people eye convolution
        for channel in range(self.numChan):
            self.imgPepl[:,:,channel] = spsg.fftconvolve(imgChan[:,:,channel],
                                                      peoplePSF,mode='same')

        # Tweak color balance for spider version - just an utter SWAG right now.
        imgChan[:,:,0] *= 0.85  # Red
        imgChan[:,:,1] *= 1.00  # Green
        imgChan[:,:,2] *= 0.85  # Blue

        # Do spider eye convolution
        for channel in range(self.numChan):
            self.imgSpdr[:,:,channel] = spsg.fftconvolve(imgChan[:,:,channel],
                                                      spiderPSF,mode='same')

    def saveSourceImage(self,figName):
        self._setupFigure(0)
        plt.imshow(jumper.imgOrig)
        print("Saving unaltered version.")
        plt.savefig(os.path.join(projRoot,figName))

    def savePeopleImage(self,figName):
        self._setupFigure(1)
        plt.imshow(jumper.imgPepl)
        print("Saving people/naked eye version.")
        plt.savefig(os.path.join(projRoot,figName))

    def saveSpiderImage(self,figName):
        self._setupFigure(2)
        plt.imshow(jumper.imgSpdr)
        print("Saving spider-eyes-ed version.")
        plt.savefig(os.path.join(projRoot,figName))


if __name__ == "__main__":

    # Use argparse to... parse args
    parser = argparse.ArgumentParser(description="Simulate what a jumping "
                                     "spider might see when they look at an "
                                     "object in the night sky.")
    parser.add_argument("-i","--image",required=False,
                        default="beletskYairglow_pano.jpg",
                        help="Source image")
#                       default="20141008tleBaldridge001.jpg",
    # Process arguments
    args   = parser.parse_args()
    srcImg = args.image  # relative path from directory containing this file

    # Create instance of class
    jumper = arachnivision()

    # Load source image
    jumper.loadSourceImage(srcImg)

    # Modify it to something resembling what spider would see?
    jumper.sourceToEyeball()

    jumper.saveSourceImage("source-"+srcImg)

    # Save convolved version of original with "people" stuck at front of name.
    # This is identical to the original in terms of color balance, but uses an
    # angular resolution of 0.02 degrees for typical human vision as opposed to
    # the 0.07 degrees quoted for a certain spider's visual angular resolution.
    jumper.savePeopleImage("people-"+srcImg)

    # Save "spider-eyes-ed" version with "spider" stuck at front of name.
    jumper.saveSpiderImage("spider-"+srcImg)
