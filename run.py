from .fit import fitter
from snappl.config import Config
from snappl.logger import SNLogger
import numpy as np

#Load configuration
config=Config.get(config_file, reread=True, prefix="spectroscopy.sloeginphys")
log=log=SNLogger()

#Get values
direct_image=config.value("dir_im")
sn_image=config.value("sn_im")
ra=config.value("ra")
dec=config.value("dec")
theta=config.value("theta")
config_file=config.value("config")

#Check variables
assert isinstance(ra, float), "RA must be a float"
assert isinstance(dec, float), "DEC must be a float"
assert isinstance(direct_image, str), "Direct image must be a string"
assert isinstance(sn_image, str), "SN image must be a string"
assert isinstance(theta, list) or isinstance(theta, np.ndarray), "theta must be a list or array" 
assert isinstance(config_file, str), "Configuration file must be a string"
if not (os.path.isfile(config_file)):
    log.error(config_file+" does not exist")
    return

#Run the fit
f=fitter(direct_image, sn_image, local=False, ra=None, dec=None)
f.make_map()
f.fit(theta, config_file)