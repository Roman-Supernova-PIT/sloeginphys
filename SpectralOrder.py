import sys
import numpy as np
from importlib.resources import files, as_file
from scipy import interpolate
from astropy.io import fits
from scipy.interpolate import UnivariateSpline
from .myUtils import *
from .band import Band


DISPTYPES = ["prism", "grism"]


class SpectralOrder:

    def __init__(self, sca, disptype, wavelengths=0, order=1, wavbinsub=1):

        if(sca<1 or sca>18):
            print("Error")

        if(disptype not in DISPTYPES):
            print("Error")

        self.wavstep = 0.001

        if(wavelengths == 0):
            wavelengths = np.arange(0.75, 1.855 + self.wavstep, self.wavstep)


        wavbin = [wavelengths[0]]

        dldr = self.dispersion(np.array([4088/2, 4088/2]).T, np.array(wavelengths, dtype=np.float64))/wavbinsub

        k = UnivariateSpline(wavelengths, dldr, k=1, s=0)
        wi = wavbin[0]
        i = 1
        while(wi<wavelengths[-1]):
            wn = wavbin[i-1] + k(wavbin[i-1]).round(3)
            if(wavelengths[-1] - wn < self.wavstep):
                wn = wavelengths[-1] + self.wavstep
            wavbin.append(wn)
            wi = wavbin[i]
            i+=1

        wavbin = np.array(wavbin)

        self.wavavg = (wavbin[:-1] + wavbin[1:]) / 2
        self.bin_index = np.array([np.argmax(w < wavbin) - 1 for w in wavelengths])
        self.bin_n = np.max(self.bin_index) + 1


        self.sca = sca
        self.disptype = disptype
        self.wavelengths = wavelengths
        self.order = order

        self.collecting_area = 3.757e4

        self._loadDispCoeffs()
        self._loadSensitivity()
        #self._loadPSF()


        exps = [np.arange(n) for n in np.arange(1,10)]
        self.xe = np.array([])
        self.ye = np.array([])
        for e in exps:
            self.xe = np.append(self.xe, e[::-1])
            self.ye = np.append(self.ye, e)

        return

    def _loadDispCoeffs(self):
        """
        Method to load the dispersion coefficients.

        """
        # placeholder function until I know how the coefficients are stored...
        # Example coefficients from JWST
        try:
            base = files('roman_snpit.core.config.data')
            prefix = "roman_"+self.disptype+"_"+"sca"+str(self.sca)
            with as_file(base.joinpath(prefix+"_dx.npy")) as f:
                self.A = np.load(f)
            with as_file(base.joinpath(prefix+"_dy.npy")) as f:
                self.B = np.load(f)
            with as_file(base.joinpath(prefix+"_dl.npy")) as f:
                self.C = np.load(f)


            if(self.disptype == "prism"):
                with as_file(base.joinpath(prefix+"_ts.npy")) as f:
                    self.D = np.load(f)

        except Exception as e:
            print(e)



        return


    def _loadSensitivity(self):
        """
        Method to load the sensitivity curve.

        """

        try:
            base = files('roman_snpit.core.config.data')
            prefix = "Roman_effarea_20210614.txt"
            with as_file(base.joinpath(prefix)) as f:
                eff_area = np.loadtxt(f, skiprows=1)
                wav = eff_area[:, 0]
                # Wave     R062      Z087      Y106      J129      H158      F184      W146      K213      SNPrism   Grism_1stOrder  Grism_0thOrder
                # this is the current format of the effective area file

            if(self.disptype == "prism"):
                tran = np.array(eff_area[:, 9] * 1e4/self.collecting_area)

            band = Band(wav, tran, unit='micron')

            self.sensitivity = band(self.wavelengths * 1e4)

        except Exception as e:
            print(e)


        return


    def _loadPSF(self):

        psfFilename = "/Users/mgriggio/roman_snpit/ePSFcube.fits"

        with fits.open(psfFilename) as psffile:
            self.psfFunc = interpolate.interp1d(np.arange(0.65, 1.95, 0.05), psffile[0].data, kind='linear', axis=0, assume_sorted=True, copy=False)

        return

###############################
### this is the function for
### polynomials
###############################

    # def dispersion(self, positions, wavelengths=0):
    #     """
    #     Method to compute the dispersion in A/pix at some undispersed
    #     position and wavelength.  This is given by the derivative of the
    #     wavelength solution as a function of position along the spectral
    #     trace.

    #     .. math::
    #        \frac{d\\lambda}{dr} = \frac{d\\lambda/dt}{\\sqrt{(dx/dt)^2 + (dy/dt)^2}}
    #     where :math:`t` is parametric value given by:
    #     .. math::
    #        t = DISPL^{-1}(\\lambda)

    #     and :math:`DISPL` comes from the grismconf file.

    #     Parameters
    #     ----------
    #     x0 : int or float
    #        The undispersed x-coordinate

    #     y0 : int or float
    #        The undispersed y-coordinate

    #     wavelength : int, float, or None, optional
    #        The wavelength (in A) to compute the dispersion.  If set to
    #        `None`, then the bandpass-averaged wavelength is used.

    #     Returns
    #     -------
    #     dldr : float
    #        The instaneous dispersion in A/pix.

    #     """

    #     # compute the dispersion using:
    #     # t = h^-1(lambda)
    #     # x(t) = a+ b*t + c*t^2 + ...
    #     # x'(t) = b+ 2*c*t + ...
    #     # y'(t) = r+ 2*s*t + ...
    #     # l'(t) = u+ 2*v*t + ...
    #     # r'(t) = sqrt(dxdt**2 + dydt**2)
    #     # dl/dr = l'(t)/r'(t)


    #     if(wavelengths == 0):
    #         wavelengths = self.wavelengths

    #     dxCoeffs = evalAll2D(positions, self.A, self.xe, self.ye)
    #     dyCoeffs = evalAll2D(positions, self.B, self.xe, self.ye)
    #     dispCoeffs = evalAll2D(positions, self.C, self.xe, self.ye)

    #     der1Coeffs = (dispCoeffs*np.arange(dispCoeffs.shape[1]))[:,1:]
    #     der2Coeffs = (der1Coeffs*np.arange(der1Coeffs.shape[1]))[:,1:]

    #     tinv = invert_single(wavelengths, dispCoeffs, der1Coeffs, der2Coeffs)

    #     t0 = tinv
    #     dtidt = 1

    #     if(self.disptype == "prism"):
    #         # tinv = 1/(t-t*)
    #         tstar = evalAll2D(positions, self.D, self.xe, self.ye)
    #         t0 = tstar + 1/tinv
    #         dtidt = - tinv**2


    #     if(np.any(t0 > 0) or np.any(t0 < 1)):
    #         print("Error!")
    #         sys.exit(-1)
    #         # should raise an exception here and do something,
    #         # since we should never get to this point

    #     dxdt = evalN(t0, (dxCoeffs*np.arange(dxCoeffs.shape[1]))[:,1:])
    #     dydt = evalN(t0, (dyCoeffs*np.arange(dyCoeffs.shape[1]))[:,1:])
    #     dldt = dtidt * evalN(tinv, der1Coeffs)

    #     dr = np.sqrt(dxdt**2 + dydt**2)

    #     if(np.any(dr == 0)):
    #         sys.exit(-1)
    #         we should not get here, exception?


    #     return dldt / np.sqrt(dxdt**2 + dydt**2)

#################################

###############################
### this is the function for
### polynomials
###############################

    # def deltas(self, positions, wavelengths=0):
    #     """
    #      Method to compute the offsets  with respect to the undispersed
    #      position.  NOTE: the final WFSS position would be given by
    #      adding back the undispersed positions.

    #      Parameters
    #      ----------
    #      x0 : float, `np.ndarray`, or int
    #         The undispersed x-position.

    #      y0 : float, `np.ndarray`, or int
    #         The undispersed y-position.

    #      wav : `np.ndarray`
    #         The wavelength (in A).

    #      Returns
    #      -------
    #      dx : float or `np.ndarray`
    #         The x-coordinate along the trace with respect to the undispersed
    #         position.

    #      dy : float or `np.ndarray`
    #         The y-coordinate along the trace with respect to the undispersed
    #         position.

    #      Notes
    #      -----
    #      The undispersed positions (`x0` and `y0`) must be of the same
    #      shape, hence either both scalars or `np.ndarray`s with the same
    #      shape.  The dtype of the variables does not matter.  If these
    #      variables are arrays, then the output will be a two-dimensional
    #      array of shape (len(wav),len(x0)).  If they are scalars, then
    #      the output will be a one-dimensional array with shape (len(wav),).

    #      """

    #     # invert dl/dr to get t
    #     # evaluate dx-dy at x,y,t

    #     if(wavelengths == 0):
    #         wavelengths = self.wavelengths
    #         #else check if it is an array

    #     dxCoeffs = evalAll2D(positions, self.A, self.xe, self.ye)
    #     dyCoeffs = evalAll2D(positions, self.B, self.xe, self.ye)

    #     dispCoeffs = evalAll2D(positions, self.C, self.xe, self.ye)

    #     der1Coeffs = (dispCoeffs*np.arange(dispCoeffs.shape[1]))[:,1:]
    #     der2Coeffs = (der1Coeffs*np.arange(der1Coeffs.shape[1]))[:,1:]

    #     tinv = invert(wavelengths, dispCoeffs, der1Coeffs, der2Coeffs)

    #     t0 = tinv

    #     if(self.disptype == "prism"):
    #         # tinv = 1/(t-t*)
    #         tstar = evalAll2D(positions, self.D, self.xe, self.ye)
    #         t0 = tstar + 1/tinv

    #     if(np.any(t0 > 0) or np.any(t0 < 1)):
    #         print("Error!")
    #         sys.exit(-1)


    #     return evalN(t0, dxCoeffs), evalN(t0, dyCoeffs)

#################################



################################
### Using this at the moment to
### reproduce roman_imsim
### simulations
################################

    def dispersion(self, positions, wavelengths=0):

        w = wavelengths
        def dydw(w):
            return (304.99*w**2 - 778.241*w + 1164.08)/(-w**2 + 3.89343*w - 1.15066)**2


        return 1/dydw( w )

################################

################################
### Using this at the moment to
### reproduce roman_imsim
### simulations
################################

    def deltas(self, positions, wavelengths=0):
        if(wavelengths == 0):
            wavelengths = self.wavelengths

        w = wavelengths
        dy = (-81.993865 + 138.367237*(w - 1.0) + 19.348549*(w - 1.0)**2)/(1.0 + 1.086447*(w - 1.0) + -0.573797*(w - 1.0)**2)

        return np.zeros((positions.shape[0],w.shape[0])), np.array([dy for i in range(positions.shape[0])])

###############################


    @property
    def name(self):
        """
        The name of the order
        """

        return str(self.order)

    def __str__(self):
        return f'Spectral order: {self.disptype}, {self.order}'
