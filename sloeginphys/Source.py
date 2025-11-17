import numpy as np
import pandas as pd
from importlib.resources import files, as_file
from .sed import SED
from .band import Band

class Source:

    def __init__(self, segMapID, segMapPixPos, segMapSubPos, id=0):


        self.segMapID = segMapID
        self.segMapPixPos = segMapPixPos
        self.segMapSubPos = segMapSubPos

        return

    def addSED(self, filt, filename, index=None):

        # this is a placeholder function

        collecting_area = 3.757e4

        base = files('roman_snpit.core.config.data')
        prefix = "Roman_effarea_20210614.txt"
        with as_file(base.joinpath(prefix)) as f:
            eff_area = np.loadtxt(f, skiprows=1)
            wav = eff_area[:, 0]
            # Wave     R062      Z087      Y106      J129      H158      F184      W146      K213      SNPrism   Grism_1stOrder  Grism_0thOrder
            # this is the current format of the effective area file

        if(filt == "H158"):
            tran = np.array(eff_area[:, 5] * 1e4/collecting_area)
        elif(filt == "SNPrism"):
            tran=np.array(eff_area[:, 9] * 1e4/collecting_area)

        self.sed = SED.from_file(filename)

        self.band = Band(wav, tran, unit='micron')

        self.mag = 23.77

        # self.mag = 26.021664968822403

        self.sed.normalize(self.band, self.mag, abmag=True)

        if (index==None):
            self.seds = [self.sed]*self.segMapSubPos.shape[0]
        else:
            self.seds[index]=self.sed

        return

    def addPixSED(self, filt, nwav):

        # this is a palceholder function

        data = np.loadtxt("result.txt").T
        wav = data[0].reshape(-1, nwav)
        flux = data[1].reshape(-1, nwav)

        self.seds = []

        for i in range(wav.shape[0]):
            wavu = wav[i]
            fu = flux[i]

            sed = SED(wavu*10**4, fu)

            self.seds.append(sed)

        return


    def addWCSPos(self, WCSPos):

        self.WCSPos = WCSPos

        return

    def addWCSSubPos(self, WCSSubPos):

        self.WCSSubPos = WCSSubPos

        return

    def addWCSSubPosCen(self, WCSSubPosCen):

        self.WCSSubPosCen =  WCSSubPosCen

        return


    def compute_weights(self, img, epsilon=1e-9):

        x = self.segMapPixPos[:,0]
        y = self.segMapPixPos[:,1]

        w = img[y-1, x-1]
        v = 0.5 * (w + np.sqrt(w * w + epsilon**2))

        # normalize
        v /= np.sum(v)

        self.weights = v

        return v
