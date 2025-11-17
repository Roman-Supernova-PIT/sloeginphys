import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import FlatLambdaCDM
from .WFSSImageCollection import WFSSImageSimulator
from .WFSSImageCollection import WFSSImageSimulator_nohdr
from .myUtils import *
from .bc03utils import make_spec
from .fit_utils import overlap
from scipy.optimize import least_squares, minimize
from synphot import SpectralElement, Observation, SourceSpectrum
from snappl.logger import SNLogger
from snappl.config import Config
from snappl.image import Image
from snappl.segmap import SegmentationMap
from snappl.dbclient import SNPITDBClient
from pypolyclip import clip_multi
import os
import glob
import time
import datetime
os.environ["SPS_HOME"] = "/global/u1/a/aisaacs/FSPS/"
try:
    import fsps
except:
    print("pyFSPS not installed or SPS_HOME directory not configured. Set SPS_HOME if you want to use FSPS. Otherwise, only BC03 may be used in fitting.")
#try:
#    import emcee
#except:
#    print("emcee not installed. Cannot use Bayesian statistics in fitting.")

class fitter:
    def __init__(self, direct_image, local=False, segmap=None, ra=None, dec=None):
        """Initialize the object
        Parameters:
            direct_image (string) - path to the fits file where the direct image for the final desired galaxy fit is
        Outputs:
            None; creates object and defines quantities that will be useful later"""
        #Create log
        self.log=SNLogger(midformat="Fitter")
        assert isinstance(local, bool), "local must be boolean"
        #Saves RA and DEC if provided and if valid
        if(ra!=None and dec!=None):
            assert isinstance(ra, float), "RA must be float"
            assert isinstance(dec, float), "DEC must be float"
            self.ra=ra
            self.dec=dec
        elif((ra!=None and dec==None) or (ra==None and dec!=None)):
            self.log.error("Both RA and DEC must be provided")
            return
        else:
            self.ra=ra
            self.dec=dec
        if(local==True):
            #Retrieve files
            self.dir_im=direct_image
            self.seg_map=segmap   
            #Check direct image
            try:
                dir_im_temp=fits.open(self.dir_im)
            except:
                self.log.error("Direct image is invalid")
                return
            #Check segmentation map
            try:
                seg_map_temp=fits.open(self.seg_map)
            except:
                self.log.error("Segmentation map is invalid")
                return
            #Retrieve data from direct image
            self.dir_im_data=dir_im_temp[1].data
            self.dir_im_hdr=dir_im_temp[1].header
            self.dir_im_band=self.dir_im_hdr["FILTER"]
            if (self.dir_im_band not in ["R062", "Z087", "Y106", "J129", "W146", "H158", "F184", "K213", "F062", "F087", "F106", "F129", "F146", "F158", "F213", "062", "087", "106", "129", "146", "158", "184", "213"]):
                self.log.error("Filter "+self.dir_im_band+" is not a valid filter")
                return
            dir_im_temp.close()
            #Retrieve data from the segmentation map
            self.seg_map_hdr=seg_map_temp[0].header
            self.seg_map_data=seg_map_temp[0].data
            self.seg_map_data_orig=seg_map_temp[0].data
            seg_map_temp.close()
            self.ref_wcs=WCS(self.seg_map_hdr)
        else:
            #Check direct image
            assert isinstance(direct_image, str) or isinstance(direct_image, pathlib.Path), "direct_image must be either a filepath to the direct image or the UUID of the direct image"
            self.dir_im=Images.find_images(filepath=direct_image)
            if(self.dir_im==None):
                assert isinstance(direct_image, str), str(direct_image)+" is not a valid UUID. UUID must be a string"
                self.dir_im=Image.get_image(direct_image)
            if(self.dir_im==None):
                self.log.error("direct_image is not a valid image or is not present in the database")
                return
            self.dir_im_data=dir_im.data
            self.dir_im_band=dir_im.band
            #Retrieve corresponding segmentation map
            self.seg_map=SegmentationMap.find_segmaps(l2image_id=self.dir_im.id)
            #Retrieve data from the segmentation map
            self.seg_map_data=self.seg_map.data
            self.seg_map_data_orig=self.seg_map.data
            self.ref_wcs=self.dir_im.get_wcs().get_astropy_wcs()
        #Get number of pixels and their coordinates
        self.pixPos=np.array(np.transpose((np.where(self.seg_map_data!=0))))
        self.numPix=self.pixPos.shape[0]
        #Define a simulator object for the desired output
        self.simulator=WFSSImageSimulator_nohdr(self.dir_im_data, self.seg_map_data, self.ref_wcs, self.dir_im_band)
        return

    def pick_object(self, ID):
        """Function to allow the user to select which object to fit
        Parameters:
            id (int) - the number in the original segmentation map that corresponds to the desired object
        Returns:
            None; rewrites variables used later and recreates the simulator"""
        assert isinstance(ID, int)
        #Find where the object with the given id is located, using the preserved copy of the segmentation map
        new_coord=np.where(self.seg_map_data_orig==ID)
        #Ensures at least one pixel matches the ID
        if(len(new_coord)==0):
            self.log.error("ID not present in segmentation map")
            return
        #Generate a grid of zeros
        new_map=np.zeros((self.seg_map.height, self.seg_map.width))
        #Assign 1 to the desired coordinates
        new_map[new_coord]=1
        #Redefine the seg_map_data variable, preserving the original map in case another object is used later
        self.seg_map_data=new_map
        #Get number of pixels and their coordinates
        self.pixPos=np.array(np.transpose((np.where(self.seg_map_data!=0))))
        self.numPix=self.pixPos.shape[0]
        #Recreate the simulator with the new segmentation map
        self.simulator=WFSSImageSimulator_nohdr(self.dir_im_data, self.seg_map_data, self.ref_wcs, self.dir_im.band)
        return

    def get_ID_xy(self, x, y):
        """Retrieves the ID of an object given the x and y coordinates"""
        assert isinstance(x, int), "x must be an integer"
        assert isinstance(y, int), "y must be an integer"
        try:
            loc=self.seg_map_data_orig[y, x]
            return loc
        except:
            self.log.error("x/y coordinate not present in the segmentation map")
            return

    def get_ID_ad(self, ra, dec):
        """Retrieves the ID of an object given the RA and DEC"""
        assert isinstance(ra, float), "ra must be a float"
        assert isinstance(dec, float), "dec must be a float"
        coord=SkyCoord(str(ra)+" "+str(dec), unit=(u.deg, u.deg))
        x_temp, y_temp=self.ref_wcs.world_to_pixel(coord)
        x=round(float(x_temp))
        y=round(float(y_temp))
        try:
            loc=self.seg_map_data_orig[y, x]
            return loc
        except:
            self.log.error("RA/DEC coordinate not present in the segmentation map")
            return

    def make_map(self, ra=None, dec=None, overwrite=False):
        """High-level function to use the given coordinates to prepare the segmentation map for fitting
        Parameters:
            ra, dec (float) - right ascension and declination of the object if not given at initialization
        Returns:
            None; rewrites variables used later and recreates the simulator"""
        if(ra==None and dec==None):
            if(self.ra==None or self.dec==None):
                self.log.error("RA and DEC must both be provided")
                return
            else:
                ID=self.get_ID_ad(self.ra, self.dec)
                self.pick_object(ID)
        else:
            if(ra==None or dec==None):
                self.log.error("RA and DEC must both be provided")
                return
            else:
                assert isinstance(ra, float), "RA must be float"
                assert isinstance(dec, float), "DEC must be float"
                ID=self.get_ID_ad(ra, dec)
                self.pick_object(ID)
                if(overwrite==True):
                    self.ra=ra
                    self.dec=dec


    def fit(self, theta, config_file, spec_data, phot_data):
        """This function fits a simulated spectrum to provided data.
        Parameters:
            spec_data (string or list) - if string: path to spectroscopic data in the form of standard Roman fits files. Must end with "/". If list: list of complete paths to files for analysis. For both: Header must include WCS and exposure time
            phot_data (string or list) - if string: path to photometric data in the form of standard Roman fits files. Must end with "/". If list: list of complete paths to files for analysis. For both: Header must include WCS, exposure time, and filter name
            theta (array) - initial guess. Must be size appropriate to match other inputs
            config_file (string) - The file containing the configuration parameters. Contains the following:
                z (float) - redshift, NOT metallicity. REQUIRED
                working_dir (string) - directory where all files created by this function will be saved
                filter_path (string) - path to the location of the Roman filter functions. Only required if photometry is true
                method (string) - scipy optimization method to use. Default is scipy least_squares with trf method. Accepts any method used by scipy minimize or scipy least_squares except custom methods
                use_bayes (bool) - choice to use Bayesian analysis to get a more detailed picture. Better fit with stronger uncertainty at cost of more time
                niter (int) - number of iterations to run. Only used in Bayesian analysis. Default is 10^5
                nwalkers (int) - number of walkers to use. Only used in Bayesian analysis. Default is 8
                sim_code (string) - choice of simulation to code. Must be BC03 or FSPS
                one_sed (bool) - if True, all pixels in the galaxy will be set to the same SED. If False, each pixel has an individual SED
                sps_home (string) - path to SPS files. Required if using FSPS, unless SPS_HOME is set elsewhere
                verbose (bool) - if True, program will provide updates on its current process and times for the lengthier process
                buffer (int) - buffer around the segmentation map to test for pixel overlaps. Default is 1. It is unlikely this parameter will need to be adjusted
                bc03_params - fixed parameters that will not be iterated over if using BC03. No defaults set
                    Includes: working directory, ised directory, library name, metallicity, IMF choice, dust parameters, SFH choice, gas recycling choice if SFH is 1 or -1, and file names if SFH is 6. See bc03utils and make_csp_file documentation for more details
                fsps_params - Fixed parameters that will not be iterated over for FSPS
                    Note: FSPS contains a vast number of parameters, many of which can be iterated over. fsps_params contains those that cannot be iterated over. See pyFSPS documetation (https://dfm.io/python-fsps/current/stellarpop_api/#fsps.StellarPopulation) for more details on available parameters. If any parameter here is not specified, the default value is used
                fsps_optional - Optional fixed parameters for FSPS
                    Note: Any FSPS parameters defined here will not be fit for. Use string "default" to set to the default but not fit. See FSPS documentation (https://dfm.io/python-fsps/current/stellarpop_api/#fsps.StellarPopulation) for more details on available parameters
                bounds - bounds of the fit. First option is a boolean to choose if using bounds or not. If not provided or use_bounds is False, bounds age to the age of the universe to prevent runtime errors
                output - various choices of what outputs are desired. Note that a reference image must be provided if any of the subtracted options are set to True
            Return options:
            Best fit parameters from the frequentist optimization
            Chi^2 test statistic from frequentist optimization
            Best fit image, either as direct array or saved as fits file
            Subtracted image, either as direct array or saved as fits file
            If use_bayes is true, the sample chain
            If use_bayes is true, the median for each parameter, serving as the best fit for the Bayesian method
        Notes:
            If use_bayes is true, the returned image is created using the median value of the parameters from the chain
            Reccomended methods are trf, lm, and Nelder-Mead. Current default method is trf
            Reference times are provided in the comments as well as in other documentation. If the program exceeds these significantly, there may be something wrong
            BC03 and FSPS behave slightly differently. Due to the smaller number of parameters in BC03, all relevant paramters will be fit automatically. Due to the high number in FSPS, fit parameters must be selected
            BC03:
                For SFH 1 or SFH -1, ensure epsilon is provided if recycling is true
                Cannot iterate over file names if using SFH 6 (see bc03utils documentation)
                SFH 7 is not currently supported
                Command line output for BC03 will be redirected into a log file in the working directory called "bc03_logfile.txt". In the unlikely event that the simulation fails, check this file for further information
            FSPS:
                SFH 2 and 3 not currently supported. SFH 2 is not supported by pyFSPS so is unlikely to become supported
                Any parameters that will not be used by the IMF, SFH, dust, and other selections will not be fit for. Please ensure the fixed parameters are compatible with the desired fittable parameters
        Written by Ann Isaacs (isaac413@umn.edu). Please contact me with any issues/bugs/suggestions
        """
        # Get config file
        config=Config.get(config_file, reread=True)
        prefix="spectroscopy.sloeginphys."
        #Check that universal parameters are valid
        verbose=config.value(prefix+"verbose")
        if(verbose!=None):
            assert isinstance(verbose, bool), "verbose must be boolean"
        else:
            verbose=False
        if(verbose==True):
            big_start=time.time()
            self.log.info("Beginning fit")
            self.log.info("Checking input parameters")
        # Extract parameters to make my life easier
        param_prefix="spectroscopy.sloeginphys.params."
        z=config.value(prefix+"temp.z")
        sn_im=config.value(prefix+"temp.SN_image")
        working_dir=config.value(prefix+"paths.working_dir")
        local=config.value(prefix+"run.local")
        filter_path=config.value(prefix+"paths.filter_path")
        provenance_tag=config.value(prefix+"temp.provenance_tag")
        spec_process=config.value(prefix+"temp.spec_process")
        phot_process=config.value(prefix+"temp.phot_process")
        mjd_min=config.value(param_prefix+"mjd_min")
        mjd_max=config.value(param_prefix+"mjd_max")
        sim_code=config.value(param_prefix+"sim_code")
        one_sed=config.value(param_prefix+"one_sed")
        method=config.value(param_prefix+"method")
        use_bayes=config.value(param_prefix+"use_bayes")
        niter=config.value(param_prefix+"niter")
        nwalkers=config.value(param_prefix+"nwalkers")
        buffer=config.value(param_prefix+"buffer")
        h0=config.value(param_prefix+"H0")
        omegaM=config.value(param_prefix+"OmegaM")
        return_fit=config.value(prefix+"output.return_fit")
        save_fit=config.value(prefix+"output.save_fit")
        return_image=config.value(prefix+"output.return_image")
        save_image=config.value(prefix+"output.save_image")
        image_name=config.value(prefix+"output.image_name")
        return_subtracted=config.value(prefix+"output.return_subtracted")
        save_subtracted=config.value(prefix+"output.save_subtracted")
        subtracted_name=config.value(prefix+"output.subtracted_name")
        if(z==None):
            self.log.error("Redshift must be provided")
            return
        else:
            assert isinstance(z, float) or isinstance(z, int), "z must be float or int"
            assert z>0, "z must be greater than zero"
        if(local!=None):
            assert isinstance(local, bool), "local must be a boolean"
        else:
            local=False
        if(sn_im==None):
            self.log.error("A 2-D spectrum containing the supernova must be provided")
        else:
            if(local==True):
                assert isinstance(sn_im, str), "SN_image must be a string"
                try:
                    test_im=fits.open(sn_im)
                    self.sn_data=test_im[1].data
                    self.sn_wcs=WCS(test_im[1].header)
                except:
                    self.log.error(str(sn_im)+" is not a valid fits file")
                    return
            else:
                assert isinstance(sn_im, str) or isinstance(ref_image, pathlib.Path), "SN_image must be a string and a valid UUID"
                test_im=Images.find_images(filepath=sn_im)
                #If the path doesn't work, try the list element as a UUID
                if(test_im==None):
                    assert isinstance(sn_im, str), str(sn_im)+" is not a valid UUID"
                    test_im=Images.get_image(sn_im)
                if(test_im==None):
                    self.log.error(str(sn_im)+" does not correspond to a valid image")
                    return
                else:
                    self.sn_data=self.test_im.data
                    self.sn_size=self.test_im.size
                    self.sn_wcs=self.test_im.get_wcs().get_astropy_wcs()
        if(working_dir==None):
            self.log.error("A working directory must be provided")
            return
        else:
            assert isinstance(working_dir, str), "working_dir must be a string"
            if(not os.path.isdir(working_dir)):
                self.log.error("working_dir is not an existing directory. Please create the directory and try again")
                return
        if(provenance_tag!=None):
            assert isinstance(provenance_tag, str), "provenance_tag must be a string"
            #Note: later add something to ensure the provenance is valid
        if(spec_process!=None):
            assert isinstance(spec_process, str), "spec_process must be a string"
            #Note: later add something to ensure process is valid
        if(phot_process!=None):
            assert isinstance(phot_process, str), "phot_process must be a string"
            #Note: later add something to ensure process is valid
        if(mjd_min!=None):
            assert isinstance(mjd_min, float), "min_mjd must be a float"
        if(mjd_max!=None):
            assert isinstance(mjd_max, float), "mjd_max must be a float"
        if(sim_code==None):
            self.log.error("Simulation code choice must be provided")
            return
        else:
            assert sim_code=="BC03" or sim_code=="FSPS", "sim_code must be a string, and must be either BC03 or FSPS"
        if(one_sed==None):
            one_sed=False
        else:
            assert isinstance(one_sed, bool), "one_sed must be a boolean"
        if(method!=None):
            assert isinstance(method, str), "method must be a string"
        if(use_bayes==None):
            use_bayes=False
        else:
            assert isinstance(use_bayes, bool), "use_bayes must be boolean. use_bayes=True not currently supported"
            if(use_bayes==True):
                self.log.error("Bayesian statistics not currently supported")
                use_bayes=False
        if(niter!=None):
            assert isinstance(niter, float) or isinstance(niter, int), "niter must be int or float"
            niter=int(niter)
        else:
            niter=int(10**5)
        if(nwalkers!=None):
            assert isinstance(nwalkers, float) or isinstance(nwalkers, int), "nwalkers must be int or float"
            nwalkers=int(nwalkers)
        else:
            nwalkers=8
        if(buffer==None):
            buffer=1
        else:
            assert isinstance(buffer, int), "buffer must be int"
        if(h0==None):
            h0=70
        else:
            assert isinstance(h0, float) or isinstance(h0, int), "H0 (Hubble constant) must be int or float"
        if(omegaM==None):
            omegaM=0.3
        else:
            assert isinstance(omegaM, float) or isinstance(omegaM, int), "OmegaM (Density of matter) must be int or float"
        if(return_fit!=None):
            assert isinstance(return_fit, bool), "return_fit must be boolean"
        else:
            return_fit=True
        if(save_fit!=None):
            assert isinstance(save_fit, bool), "save_fit must be boolean"
        else:
            #Temporary, until we have a file format
            save_fit=False
        if(return_image!=None):
            assert isinstance(return_image, bool), "return_image must be boolean"
        else:
            return_image=True
        if(save_image!=None):
            assert isinstance(save_image, bool), "save_image must be boolean"
        else:
            save_image=False
        if(return_subtracted!=None):
            assert isinstance(return_subtracted, bool), "return_subtracted must be boolean"
        else:
            return_subtracted=True
        if(save_subtracted!=None):
            assert isinstance(save_subtracted, bool), "save_subtracted must be boolean"
        else:
            save_subtracted=False
        if(image_name!=None):
            assert isinstance(image_name, str), "image_name must be a string"
        if(subtracted_name!=None):
            assert isinstance(subtracted_name, str), "subtracted_name must be a string"
        if (method!=None):
            if method in ["Newton-CG", "dogleg", "trust-ncg", "trust-exact", "trust-krylov"]:
                self.log.error("Requires callable Jacobian. Not currently supported")
                return
            elif method not in ["trf", "dogbox", "lm", "Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "COBYQA", "SLSQP", "trust-constr"]:
                self.log.error("Invalid method")
                return
        #Check that input parameters for the specific sim_code choice are valid and set up a parameter dictionary for use in fitting
        if sim_code == "BC03":
            # Retrieve fixed parameters and check that all needed parameters have been provided
            ised_dir = config.value(param_prefix+"bc03_params.ised_dir")
            lib = config.value(param_prefix+"bc03_params.library")
            metallicity = config.value(param_prefix+"bc03_params.metallicity")
            imf = config.value(param_prefix+"bc03_params.imf")
            sfh = config.value(param_prefix+"bc03_params.sfh")
            dust = config.value(param_prefix+"bc03_params.dust")
            recyc = config.value(param_prefix+"bc03_params.recyc")
            file_names = config.value(param_prefix+"bc03_params.file_names")
            # Check that universal parameters are present and valid
            if (ised_dir == None or (not isinstance(ised_dir, str))):
                self.log.error( "Please provide ised directory as a string")
                return
            if(not os.path.isdir(ised_dir)):
                self.log.error("ised_dir is not an existing directory")
                return
            if (lib == None or metallicity == None or imf == None or (not isinstance(lib, str)) or (not isinstance(metallicity, str)) or (not isinstance(imf, str))):
                self.log.error("Please provide library, metallicity, and IMF choice as strings")
                return
            if(not os.path.isfile(ised_dir+"bc2003_lr_"+lib+"_"+metallicity+"_"+imf+"_ssp.ised")):
                self.log.error("bc2003_lr_"+lib+"_"+metallicity+"_"+imf+"_ssp.ised is not present in the ised directory")
                return
            if sfh == None or (not isinstance(sfh, int)):
                self.log.error("Please provide SFH choice as an integer")
                return
            if dust == None or (not isinstance(dust, bool)):
                self.log.error("Please provide dust as a boolean")
                return
            # Check that SFH is valid and supported
            if sfh == 7:
                self.log.error("SFH not supported")
                return
            elif sfh not in [0, 1, -1, 2, 3, 4, 5, 6, 7]:
                self.log.error("Invalid SFH")
                return
            # Check that parameter-dependent values are valid
            if sfh == 1 or sfh == -1:
                if (recyc == None or not(isinstance(recyc, bool))):
                    self.log.error("If SFH is 1 or -1, please provide gas recycling choice as a boolean")
                    return
            if sfh == 6:
                if(one_sed==False):
                    if (file_names == None or not(isinstance(file_names, list))):
                        self.log.error("If SFH is 6, please provide file names as a list of strings")
                        return
                    for f in file_names:
                        if(not os.path.isfile(f)):
                            self.log.error("File "+f+" does not exist")
                            return
                else:
                    if(file_names==None or not(isinstance(file_names, str))):
                        self.log.error("If SFH is 6, please provide a file name as a string")
                        return
                    if(not os.path.isfile(file_names)):
                        self.log.error("File "+f+" does not exist")
                        return
            # Combine parameters to make the csp_params list used later
            csp_params = [lib, metallicity, imf, dust, sfh]
            # Check theta length
            if ((one_sed==False) and (len(theta)) % self.numPix != 0):
                self.log.error("Paramater length incorrect")
                return
            if(one_sed==False):
                plength = len(theta) / self.numPix
            else:
                plength = len(theta)
            if dust == False:
                if (sfh == 0 or sfh == 6) and plength != 1:
                    self.log.error("Parameter length incorret")
                    return
                elif sfh == 1 or sfh == -1:
                    if recyc == False and plength != 3:
                        self.log.error("Parameter length incorret")
                        return
                    elif recyc == True and plength != 4:
                        self.log.error("Parameter length incorret")
                        return
                elif (sfh == 2) and plength != 2:
                    self.log.error("Parameter length incorret")
                    return
                elif (sfh == 3 or sfh == 4 or sfh == 5) and plength != 3:
                    self.log.error("Parameter length incorret")
                    return
            else:
                if (sfh == 0 or sfh == 6) and plength != 3:
                    self.log.error("Parameter length incorret")
                    return
                elif sfh == 1 or sfh == -1:
                    if recyc == False and plength != 5:
                        self.log.error("Parameter length incorret")
                        return
                    elif recyc == True and plength != 6:
                        self.log.error("Parameter length incorret")
                        return
                elif (sfh == 2) and plength != 4:
                    self.log.error("Parameter length incorret")
                    return
                elif (sfh == 3 or sfh == 4 or sfh == 5) and plength != 5:
                    self.log.error("Parameter length incorret")
                    return
            # If SFH is 6, checks that file length matches number of pixels
            if sfh == 6:
                if(one_sed==False):
                    if len(file_names)!=self.numPix:
                        self.log.error("Incorrect length of file name list")
                        return
            #Creates parameter dictionary for later use and a dictionary to pass to bc03utils
            param_dict={}
            if (sfh==0 or sfh==6):
                if(dust==True):
                    param_dict[0]="mu"
                    param_dict[1]="tauV"
                    param_dict[2]="age"
                else:
                    param_dict[0]="age"
            elif(np.abs(sfh)==1):
                param_dict={0: "tau"}
                if(recyc==True):
                    param_dict[1]="epsilon"
                    param_dict[2]="Tcut"
                    if(dust==True):
                        param_dict[3]="mu"
                        param_dict[4]="tauV"
                        param_dict[5]="age"
                    else:
                        param_dict[3]="age"
                else:
                    param_dict[1]="Tcut"
                    if(dust==True):
                        param_dict[2]="mu"
                        param_dict[3]="tauV"
                        param_dict[4]="age"
                    else:
                        param_dict[2]="age"
            elif(sfh==2):
                param_dict={0: "tau"}
                if(dust==True):
                    param_dict[1]="mu"
                    param_dict[2]="tauV"
                    param_dict[3]="age"
                else:
                    param_dict[1]="age"
            elif(sfh==3):
                param_dict={0: "sfr", 1: "Tcut"}
                if(dust==True):
                    param_dict[2]="mu"
                    param_dict[3]="tauV"
                    param_dict[4]="age"
                else:
                    param_dict[2]="age"
            elif(sfh==4 or sfh==5):
                param_dict={0: "tau", 1: "Tcut"}
                if(dust==True):
                    param_dict[2]="mu"
                    param_dict[3]="tauV"
                    param_dict[4]="age"
                else:
                    param_dict[2]="age"
            else: #This should be impossible as this is checked above but you never know
                self.log.error("Invalid SFH")
                return
        elif sim_code == "FSPS":
            # Sets SPS to the provided path. This can also be set above in the imports section
            if (config.value(param_prefix+"sps_home")) != None:
                assert isinstance(config.value(param_prefix+"sps_home"), str), "sps_home must be a string"
                if not os.path.isdir(config.value(param_prefix+"sps_home")):
                    self.log.error(config.value(param_prefix+"sps_home")+" does not exist")
                    return
                os.environ["SPS_HOME"] = config.value(prefix+"paths.sps_home")
            #Check parameters that must be selected at initialization
            if config.value(param_prefix+"fsps_params.zcontinuous") != None:
                assert config.value(param_prefix+"fsps_params.zcontinuous") in [0, 1, 2, 3], ("zcontinuous must be an integer and must be 0, 1, 2, or 3")
                zcont=config.value(param_prefix+"fsps_params.zcontinuous")
            else:
                zcont=0
            # Initialize stellar population object with the fixed parameters, checking to ensure the values are valid. Any parameters not specified in the configuration file will not be altered from the default
            sp=fsps.StellarPopulation(compute_vega_mags=False, vactoair_flag=False, zcontinuous=zcont)
            if config.value(param_prefix+"fsps_params.imf_type") != None:
                assert config.value(param_prefix+"fsps_params.imf_type") in [0, 1, 2, 3, 4, 5], ("imf_type must be an integer and must be 0, 1, 2, 3, 4, or 5")
                sp.params["imf_type"] = config.value(param_prefix+"fsps_params.imf_type")
            if config.value(param_prefix+"fsps_params.sfh") != None:
                assert config.value(param_prefix+"fsps_params.sfh") in [0, 1, 4, 5], ("sfh must be an integer and must be 0, 1, 4, or 5. Options 2 and 3 are not currently supported")
                sp.params["sfh"] = config.value(param_prefix+"fsps_params.sfh")
            if config.value(param_prefix+"fsps_params.dust_type") != None:
                assert config.value(param_prefix+"fsps_params.dust_type") in [0, 1, 2, 3, 4, 5, 6], ("dust_type must be an integer and must be 0, 1, 2, 3, 4, 5, or 6")
                sp.params["dust_type"] = config.value(param_prefix+"fsps_params.dust_type")
            if config.value(param_prefix+"fsps_params.tpagb_norm_type") != None:
                assert config.value(param_prefix+"fsps_params.tpagb_norm_type") in [0, 1, 2], ("tpagb_norm_type must be an integer and must be 0, 1, or 2")
                sp.params["tpagb_norm_type"] = config.value(param_prefix+"fsps_params.tpagb_norm_type")
            if config.value(param_prefix+"fsps_params.wgp1") != None:
                assert config.value(param_prefix+"fsps_params.wgp1") in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], "wgp1 must be an integer and must be 1-18"
                sp.params["wgp1"] = config.value(param_prefix+"fsps_params.wgp1")
            if config.value(param_prefix+"fsps_params.wgp2") != None:
                assert config.value(param_prefix+"fsps_params.wgp2") in [1, 2, 3, 4, 5, 6], ("wgp2 must be an integer and must be 1, 2, 3, 4, 5, or 6")
                sp.params["wgp2"] = config.value(param_prefix+"fsps_params.wgp2")
            if config.value(param_prefix+"fsps_params.wgp3") != None:
                assert config.value(param_prefix+"fsps_params.wgp3") in [0, 1], ("wgp3 must be an integer and must be 0 or 1")
                sp.params["wgp3"] = config.value(param_prefix+"fsps_params.wgp3")
            if config.value(param_prefix+"fsps_params.zmet") != 0:
                assert isinstance(config.value(param_prefix+"fsps_params.zmet"), int), ("zmet must be an integer")
                sp.params["zmet"] = config.value(param_prefix+"fsps_optional.zmet")
            if config.value(param_prefix+"fsps_params.use_wr_spectra") != None:
                assert config.value(param_prefix+"fsps_params.use_wr_spectra") in [0, 1], ("use_wr_spectra must be an integer and must be 0 or 1")
                sp.params["use_wr_spectra"] = config.value(param_prefix+"fsps_params.use_wr_spectra")
            if config.value(param_prefix+"fsps_params.add_xrb_emission") != None:
                assert config.value(param_prefix+"fsps_params.add_xrb_emission") in [0, 1], ("add_xrb_emission must be an integer and must be 0 or 1")
                sp.params["add_xrb_emission"] = config.value(param_prefix+"fsps_params.add_xrb_emission")
            if config.value(param_prefix+"fsps_params.add_agb_dust_model") != None:
                assert isinstance(config.value(param_prefix+"fsps_params.add_agb_dust_model"), bool), ("add_agb_dust_model must be a boolean")
                sp.params["add_agb_dust_model"] = config.value(param_prefix+"fsps_params.add_agb_dust_model")
            if config.value(param_prefix+"fsps_params.add_dust_emission") != None:
                assert isinstance(config.value(param_prefix+"fsps_params.add_dust_emission"), bool), ("add_dust_emission must be a boolean")
                sp.params["add_dust_emission"] = config.value(param_prefix+"fsps_params.add_dust_emission")
            if config.value(param_prefix+"fsps_params.add_igm_absorption") != None:
                assert isinstance(config.value(param_prefix+"fsps_params.add_igm_absorption"), bool), ("add_igm_absorption must be a boolean")
                sp.params["add_igm_absorption"] = config.value(param_prefix+"fsps_params.add_igm_absorption")
            if config.value(param_prefix+"fsps_params.add_neb_emission") != None:
                assert isinstance(config.value(param_prefix+"fsps_params.add_neb_emission"), bool), ("add_neb_emission must be a boolean")
                sp.params["add_neb_emission"] = config.value(param_prefix+"fsps_params.add_neb_emission")
            if config.value(param_prefix+"fsps_params.add_neb_continuum") != None:
                assert isinstance(config.value(param_prefix+"fsps_params.add_neb_continuum"), bool), ("add_neb_continuum must be a boolean")
                sp.params["add_neb_continuum"] = config.value(param_prefix+"fsps_params.add_neb_continuum")
            if config.value(param_prefix+"fsps_params.add_stellar_remnants") != None:
                assert isinstance(config.value(param_prefix+"fsps_params.add_stellar_remnants"), bool), ("add_stellar_remnants must be a boolean")
                sp.params["add_stellar_remnants"] = config.value(param_prefix+"fsps_params.add_stellar_remnants")
            if config.value(param_prefix+"fsps_params.compute_light_ages") != None:
                assert isinstance(config.value(param_prefix+"fsps_params.compute_light_ages"), bool), ("compute_light_ages must be a boolean")
                sp.params["compute_light_ages"] = config.value(param_prefix+"fsps_params.compute_light_ages")
            if config.value(param_prefix+"fsps_params.nebemlineinspec") != None:
                assert isinstance(config.value(param_prefix+"fsps_params.nebemlineinspec"), bool), ("nebemlineinspec must be a boolean")
                sp.params["nebemlineinspec"] = config.value(param_prefix+"fsps_params.nebemlineinspec")
            if config.value(param_prefix+"fsps_params.smooth_velocity") != None:
                assert isinstance(config.value(param_prefix+"fsps_params.smooth_velocity"), bool), ("smooth_velocity must be a boolean")
                sp.params["smooth_velocity"] = config.value(param_prefix+"fsps_params.smooth_velocity")
            if config.value(param_prefix+"fsps_params.smooth_lsf") != None:
                assert isinstance(config.value(param_prefix+"fsps_params.smooth_lsf"), bool), ("smooth_lsf must be a boolean")
                sp.params["smooth_lsf"] = config.value(param_prefix+"fsps_params.smooth_lsf")
            if config.value(param_prefix+"fsps_params.cloudy_dust") != None:
                assert isinstance(config.value(param_prefix+"fsps_params.cloudy_dust"), bool), ("cloudy_dust must be a boolean")
                sp.params["cloudy_dust"] = config.value(param_prefix+"fsps_params.cloudy_dust")
            if config.value(param_prefix+"fsps_params.sigma_smooth") != None:
                assert isinstance(config.value(param_prefix+"fsps_params.sigma_smooth"), float), ("sigma_smooth must be a float")
                sp.params["sigma_smooth"] = config.value(param_prefix+"fsps_params.sigma_smooth")
            if config.value(param_prefix+"fsps_params.min_wave_smooth") != None:
                assert isinstance(config.value(param_prefix+"fsps_params.min_wave_smooth"), float), ("min_wave_smooth must be a float")
                sp.params["min_wave_smooth"] = config.value(param_prefix+"fsps_params.min_wave_smooth")
            if config.value(param_prefix+"fsps_params.max_wave_smooth") != None:
                assert isinstance(config.value(param_prefix+"fsps_params.max_wave_smooth"), float), ("max_wave_smooth must be a float")
                sp.params["max_wave_smooth"] = config.value(param_prefix+"fsps_params.max_wave_smooth")
            # Assign the location of each parameter using a dictionary. Kind of an odd way to do this, but it assigns each used parameter an index that can be used later in fitting. Also sets any fixed values. If a parameter is set to "default", does nothing
            # If you know a better way to do this PLEASE let me know (isaac413@umn.edu)
            param_dict = {}
            i = 0
            # Age
            if config.value(param_prefix+"fsps_optional.tage") == None:
                param_dict[i] = "tage"
                i += 1
            else:
                assert isinstance(config.value(param_prefix+"fsps_optional.tage"), float), "tage must be a float"
                assert config.value(param_prefix+"fsps_optional.tage")>0, "tage must be greater than zero"
                sp.params["tage"] = config.value(param_prefix+"fsps_optional.tage")
            # Metallicity parameters
            if config.value(param_prefix+"fsps_params.zcontinuous") != 0:
                if config.value(param_prefix+"fsps_optional.logzsol") == None:
                    param_dict[i] = "logzsol"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.logzsol") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.logzsol"), float), ("logzsol must be a float")
                    sp.params["logzsol"] = config.value(param_prefix+"fsps_optional.logzsol")
                if config.value(param_prefix+"fsps_params.zcontinuous") == 2:
                    if config.value(param_prefix+"fsps_optional.pmetals") == None:
                        param_dict[i] = "pmetals"
                        i += 1
                    elif config.value(param_prefix+"fsps_optional.pmetals") != "default":
                        assert isinstance(config.value(param_prefix+"fsps_optional.pmetals"), float), ("pmetals must be a float")
                        sp.params["pmetals"] = config.value(param_prefix+"fsps_optional.pmetals")
            if config.value(param_prefix+"fsps_params.add_neb_emission") == True:
                if config.value(param_prefix+"fsps_optional.gas_logu") == None:
                    param_dict[i] = "gas_logu"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.gas_logu") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.gas_logu"), float), ("gas_logu must be a float")
                    sp.params["gas_logu"] = config.value(param_prefix+"fsps_optional.gas_logu")
                if config.value(param_prefix+"fsps_optional.gas_logz") == None:
                    param_dict[i] = "gas_logz"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.gas_logz") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.gas_logz"), float), ("gas_logz must be a float")
                    sp.params["gas_logz"] = config.value(param_prefix+"fsps_optional.gas_logz")
            # IMF parameters
            if config.value(param_prefix+"fsps_params.imf_type") == 2:
                if config.value(param_prefix+"fsps_optional.imf1") == None:
                    param_dict[i] = "imf1"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.imf1") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.imf1"), float), ("imf1 must be a float")
                    sp.params["imf1"] = config.value(param_prefix+"fsps_optional.imf1")
                if config.value(param_prefix+"fsps_optional.imf2") == None:
                    param_dict[i] = "imf2"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.imf2") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.imf2"), float), ("imf2 must be a float")
                    sp.params["imf2"] = config.value(param_prefix+"fsps_optional.imf2")
                if config.value(param_prefix+"fsps_optional.imf3") == None:
                    param_dict[i] = "imf3"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.imf3") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.imf3"), float), ("imf3 must be a float")
                    sp.params["imf3"] = config.value(param_prefix+"fsps_optional.imf3")
            elif config.value(param_prefix+"fsps_params.imf_type") == 3:
                if config.value(param_prefix+"fsps_optional.vdmc") == None:
                    param_dict[i] = "vdmc"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.vdmc") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.vdmc"), float), ("vdmc must be a float")
                    sp.params["vdmc"] = config.value(param_prefix+"fsps_optional.vdmc")
            elif config.value(param_prefix+"fsps_params.imf_type") == 4:
                if config.value(param_prefix+"fsps_optional.mdave") == None:
                    param_dict[i] = "mdave"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.mdave") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.mdave"), float), ("mdave must be a float")
                    sp.params["mdave"] = config.value(param_prefix+"fsps_optional.mdave")
            if config.value(param_prefix+"fsps_optional.imf_upper_limit") == None:
                param_dict[i] = "imf_upper_limit"
                i += 1
            elif config.value(param_prefix+"fsps_optional.imf_upper_limit") != "default":
                assert isinstance(config.value(param_prefix+"fsps_optional.imf_upper_limit"), float), ("imf_upper_limit must be a float")
                sp.params["imf_upper_limit"] = config.value(param_prefix+"fsps_optional.imf_upper_limit")
            if config.value(param_prefix+"fsps_optional.imf_lower_limit") == None:
                param_dict[i] = "imf_lower_limit"
                i += 1
            elif config.value(param_prefix+"fsps_optional.imf_lower_limit") != "default":
                assert isinstance(config.value(param_prefix+"fsps_optional.imf_lower_limit"), float), ("imf_lower_limit must be a float")
                sp.params["imf_lower_limit"] = config.value(param_prefix+"fsps_optional.imf_lower_limit" )
            if config.value(param_prefix+"fsps_optional.masscut") == None:
                param_dict[i] = "masscut"
                i += 1
            elif config.value(param_prefix+"fsps_optional.masscut") != "default":
                assert isinstance(config.value(param_prefix+"fsps_optional.masscut"), float), ("masscut must be a float")
                sp.params["masscut"] = config.value(param_prefix+"fsps_optional.masscut")
            # SFH parameters
            if sp.params["sfh"] > 0:
                if config.value(param_prefix+"fsps_optional.sf_start") == None:
                    param_dict[i] = "sf_start"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.sf_start") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.sf_start"), float), ("sf_start must be a float")
                    sp.params["sf_start"] = config.value(param_prefix+"fsps_optional.sf_start")
                if config.value(param_prefix+"fsps_optional.sf_trunc") == None:
                    param_dict[i] = "sf_trunc"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.sf_trunc") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.sf_trunc"), float), ("sf_trunc must be a float")
                    sp.params["sf_trunc"] = config.value(param_prefix+"fsps_optional.sf_trunc")
                if config.value(param_prefix+"fsps_params.sfh") in [1, 4]:
                    if config.value(param_prefix+"fsps_optional.fburst") == None:
                        param_dict[i] = "fburst"
                        i += 1
                    elif config.value(param_prefix+"fsps_optional.fburst") != "default":
                        assert isinstance(config.value(param_prefix+"fsps_optional.fburst"), float), ("fburst must be a float")
                        sp.params["fburst"] = config.value(param_prefix+"fsps_optional.fburst")
                    if config.value(param_prefix+"fsps_optional.tburst") == None:
                        param_dict[i] = "tburst"
                        i += 1
                    elif config.value(param_prefix+"fsps_optional.tburst") != "default":
                        assert isinstance(config.value(param_prefix+"fsps_optional.tburst"), float), ("tburst must be a float")
                        sp.params["tburst"] = config.value(param_prefix+"fsps_optional.tburst")
                    if config.value(param_prefix+"fsps_optional.tau") == None:
                        param_dict[i] = "tau"
                        i += 1
                    elif config.value(param_prefix+"fsps_optional.tau") != "default":
                        assert isinstance(config.value(param_prefix+"fsps_optional.tau"), float), ("tau must be a float")
                        sp.params["tau"] = config.value(param_prefix+"fsps_optional.tau")
                    if config.value(param_prefix+"fsps_optional.const") == None:
                        param_dict[i] = "const"
                        i += 1
                    elif config.value(param_prefix+"fsps_optional.const") != "default":
                        assert isinstance(config.value(param_prefix+"fsps_optional.const"), float), ("const must be a float")
                        sp.params["const"] = config.value(param_prefix+"fsps_optional.const")
                elif config.value(param_prefix+"fsps_params.sfh") == 5:
                    if config.value(param_prefix+"fsps_optional.sf_slope") == None:
                        param_dict[i] = "sf_slope"
                        i += 1
                    elif config.value(param_prefix+"fsps_optional.sf_slope") != "default":
                        assert isinstance(config.value(param_prefix+"fsps_optional.sf_slope"), float), ("sf_slope must be a float")
                        sp.params["sf_slope"] = config.value(param_prefix+"fsps_optional.sf_slope")
            # Dust parameters
            if config.value(param_prefix+"fsps_optional.frac_nodust") == None:
                param_dict[i] = "frac_nodust"
                i += 1
            elif config.value(param_prefix+"fsps_optional.frac_nodust") != "default":
                assert isinstance(config.value(param_prefix+"fsps_optional.frac_nodust"), float), ("frac_nodust must be a float")
                sp.params["frac_nodust"] = config.value(param_prefix+"fsps_optional.frac_nodust")
            if config.value(param_prefix+"fsps_params.dust_type") in [0, 4]:
                if config.value(param_prefix+"fsps_optional.dust_index") == None:
                    param_dict[i] = "dust_index"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.dust_index") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.dust_index"), float), ("dust_index must be a float")
                    sp.params["dust_index"] = config.value(param_prefix+"fsps_optional.dust_index")
            else:
                if config.value(param_prefix+"fsps_params.dust_type") == 1:
                    if config.value(param_prefix+"fsps_optional.uvb") == None:
                        param_dict[i] = "uvb"
                        i += 1
                    elif config.value(param_prefix+"fsps_optional.uvb") != "default":
                        assert isinstance(config.value(param_prefix+"fsps_optional.uvb"), float), ("uvb must be a float")
                        sp.params["uvb"] = config.value(param_prefix+"fsps_optional.uvb")
                    if config.value(param_prefix+"fsps_optional.mwr") == None:
                        param_dict[i] = "mwr"
                        i += 1
                    elif config.value(param_prefix+"fsps_optional.mwr") != "default":
                        assert isinstance(config.value(param_prefix+"fsps_optional.mwr"), float), ("mwr must be a float")
                        sp.params["mwr"] = config.value(param_prefix+"fsps_optional.mwr")
                if config.value(param_prefix+"fsps_optional.dust_tesc") == None:
                    param_dict[i] = "dust_tesc"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.dust_tesc") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.dust_tesc"), float), ("dust_tesc must be a float")
                    sp.params["dust_tesc"] = config.value(param_prefix+"fsps_optional.dust_tesc")
                if config.value(param_prefix+"fsps_params.dust_type") not in [2, 3]:
                    if config.value(param_prefix+"fsps_optional.dust1") == None:
                        param_dict[i] = "dust1"
                        i += 1
                    elif config.value(param_prefix+"fsps_optional.dust1") != "default":
                        assert isinstance(config.value(param_prefix+"fsps_optional.dust1"), float), ("dust1 must be a float")
                        sp.params["dust1"] = config.value(param_prefix+"fsps_optional.dust1")
                if config.value(param_prefix+"fsps_params.dust_type") != 3:
                    if config.value(param_prefix+"fsps_optional.dust2") == None:
                        param_dict[i] = "dust2"
                        i += 1
                    elif config.value(param_prefix+"fsps_optional.dust2") != "default":
                        assert isinstance(config.value(param_prefix+"fsps_optional.dust2"), float), ("dust2 must be a float")
                        sp.params["dust2"] = config.value(param_prefix+"fsps_optional.dust2")
                if config.value(param_prefix+"fsps_optional.dust3") == None:
                    param_dict[i] = "dust3"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.dust3") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.dust3"), float), ( "dust3 must be a float")
                    sp.params["dust3"] = config.value(param_prefix+"fsps_optional.dust3")
                if config.value(param_prefix+"fsps_optional.frac_obrun") == None:
                    param_dict[i] = "frac_obrun"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.frac_obrun") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.frac_obrun"), float), ("frac_obrun must be a float")
                    sp.params["frac_obrun"] = config.value(param_prefix+"fsps_optional.frac_obrun")
                if config.value(param_prefix+"fsps_optional.dust1_index") == None:
                    param_dict[i] = "dust1_index"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.dust1_index") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.dust1_index"), float), ("dust1_index must be a float")
                    sp.params["dust1_index"] = config.value(param_prefix+"fsps_optional.dust1_index")
            if config.value(param_prefix+"fsps_params.add_dust_emission") == True:
                if config.value(param_prefix+"fsps_optional.duste_gamma") == None:
                    param_dict[i] = "duste_gamma"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.duste_gamma") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.duste_gamma"), float), ("duste_gamma must be a float")
                    sp.params["duste_gamma"] = config.value(param_prefix+"fsps_optional.duste_gamma")
                if config.value(param_prefix+"fsps_optional.duste_umin") == None:
                    param_dict[i] = "duste_umin"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.duste_umin") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.duste_umin"), float), ("duste_umin must be a float")
                    sp.params["duste_umin"] = config.value(param_prefix+"fsps_optional.duste_umin")
                if config.value(param_prefix+"fsps_optional.duste_qpah") == None:
                    param_dict[i] = "duste_qpah"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.duste_qpah") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.duste_qpah"), float), ( "duste_qpah must be a float")
                    sp.params["duste_qpah"] = config.value(param_prefix+"fsps_optional.duste_qpah")
            if config.value(param_prefix+"fsps_params.add_agb_dust_model") == True:
                if config.value(param_prefix+"fsps_optional.agb_dust") == None:
                    param_dict[i] = "agb_dust"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.agb_dust") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.agb_dust"), float), ("agb_dust must be a float")
                    sp.params["agb_dust"] = config.value(param_prefix+"fsps_optional.agb_dust")
            # Misc parameters
            if config.value(param_prefix+"fsps_optional.fagn") == None:
                param_dict[i] = "fagn"
                i += 1
            elif config.value(param_prefix+"fsps_optional.fagn") != "default":
                assert isinstance(config.value(param_prefix+"fsps_optional.fagn"), float), ("fagn must be a float" )
                sp.params["fagn"] = config.value(param_prefix+"fsps_optional.fagn")
            if config.value(param_prefix+"fsps_optional.agn_tau") == None:
                param_dict[i] = "agn_tau"
                i += 1
            elif config.value(param_prefix+"fsps_optional.agn_tau") != "default":
                assert isinstance(config.value(param_prefix+"fsps_optional.agn_tau"), float), ("agn_tau must be a float")
                sp.params["agn_tau"] = config.value(param_prefix+"fsps_optional.agn_tau")
            if config.value(param_prefix+"fsps_optional.logt_wmb_hot") == None:
                param_dict[i] = "logt_wmb_hot"
                i += 1
            elif config.value(param_prefix+"fsps_optional.logt_wmb_hot") != "default":
                assert isinstance(config.value(param_prefix+"fsps_optional.logt_wmb_hot"), float), ("logt_wmb_hot must be a float")
                sp.params["logt_wmb_hot"] = config.value(param_prefix+"fsps_optional.logt_wmb_hot")
            if config.value(param_prefix+"fsps_optional.redgb") == None:
                param_dict[i] = "redgb"
                i += 1
            elif config.value(param_prefix+"fsps_optional.redgb") != "default":
                assert isinstance(config.value(param_prefix+"fsps_optional.redgb"), float), ("redgb must be a float")
                sp.params["redgb"] = config.value(param_prefix+"fsps_optional.redgb")
            if config.value(param_prefix+"fsps_optional.agb") == None:
                param_dict[i] = "agb"
                i += 1
            elif config.value(param_prefix+"fsps_optional.agb") != "default":
                assert isinstance(config.value(param_prefix+"fsps_optional.agb"), float), ("agb must be a float")
                sp.params["agb"] = config.value(param_prefix+"fsps_optional.agb")
            if config.value(param_prefix+"fsps_optional.fcstar") == None:
                param_dict[i] = "fcstar"
                i += 1
            elif config.value(param_prefix+"fsps_optional.fcstar") != "default":
                assert isinstance(config.value(param_prefix+"fsps_optional.fcstar"), float), ("fcstar must be a float")
                sp.params["fcstar"] = config.value(param_prefix+"fsps_optional.fcstar")
            if config.value(param_prefix+"fsps_optional.sbss") == None:
                param_dict[i] = "sbss"
                i += 1
            elif config.value(param_prefix+"fsps_optional.sbss") != "default":
                assert isinstance(config.value(param_prefix+"fsps_optional.sbss"), float), ("sbss must be a float")
                sp.params["sbss"] = config.value(param_prefix+"fsps_optional.sbss")
            if config.value(param_prefix+"fsps_optional.fbhb") == None:
                param_dict[i] = "fbhb"
                i += 1
            elif config.value(param_prefix+"fsps_optional.fbhb") != "default":
                assert isinstance(config.value(param_prefix+"fsps_optional.fbhb"), float), ("fbhb must be a float")
                sp.params["fbhb"] = config.value(param_prefix+"fsps_optional.fbhb")
            if config.value(param_prefix+"fsps_optional.pagb") == None:
                param_dict[i] = "pagb"
                i += 1
            elif config.value(param_prefix+"fsps_optional.pagb") != "default":
                assert isinstance(config.value(param_prefix+"fsps_optional.pagb"), float), ("pagb must be a float")
                sp.params["pagb"] = config.value(param_prefix+"fsps_optional.pagb")
            if (config.value(param_prefix+"fsps_optional.frac_xrb") == None):  # This parameter covers x-ray emission so it may be irrelevant
                param_dict[i] = "frac_xrb"
                i += 1
            elif config.value(param_prefix+"fsps_optional.frac_xrb") != "default":
                assert isinstance(config.value(param_prefix+"fsps_optional.frac_xrb"), float), ("frac_xrb must be a float")
                sp.params["frac_xrb"] = config.value(param_prefix+"fsps_optional.frac_xrb")
            if config.value(param_prefix+"fsps_params.tpagb_norm_type") == 1:
                if (config.value(param_prefix+"fsps_optional.dell") == None):
                    param_dict[i] = "dell"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.dell") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.dell"), float), ("dell must be a float")
                    sp.params["dell"] = config.value(param_prefix+"fsps_optional.dell")
                if config.value(param_prefix+"fsps_optional.delt") == None:
                    param_dict[i] = "delt"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.delt") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.delt"), float), ("delt must be a float")
                    sp.params["delt"] = config.value(param_prefix+"fsps_optional.delt")
            if config.value(param_prefix+"fsps_params.add_igm_absorption") == True:
                if config.value(param_prefix+"fsps_optional.igm_factor") == None:
                    param_dict[i] = "igm_factor"
                    i += 1
                elif config.value(param_prefix+"fsps_optional.igm_factor") != "default":
                    assert isinstance(config.value(param_prefix+"fsps_optional.igm_factor"), float), ("igm_factor must be a float")
                    sp.params["igm_factor"] = config.value(param_prefix+"fsps_optional.igm_factor")
        else:
            self.log.error("Invalid simulation code choice. Please select either BC03 (GALAXEV) or FSPS")
            return
        #Check that any bounds provided are valid if bounds are used
        use_bounds=config.value(param_prefix+"bounds.use_bounds")
        if(use_bounds!=None):
            assert isinstance(use_bounds, bool), "use_bounds must be boolean"
        else:
            use_bounds=False
        if(use_bounds==True):
            for j in param_dict.keys():
                if(config.value(param_prefix+"bounds."+param_dict[j])!=None):
                    assert isinstance(config.value(param_prefix+"bounds."+param_dict[j]), list), "Bounds on "+param_dict[j]+" must be a list"
                    btest=config.value(param_prefix+"bounds."+param_dict[j])
                    if(len(btest)!=2):
                        self.log.error("Bound on "+param_dict[j]+" must contain exactly two elements of the form [lower bound, upper bound]")
                        return
                    assert isinstance(config.value(param_prefix+"bounds."+param_dict[j])[0], float) or isinstance(config.value(param_prefix+"bounds."+param_dict[j])[0], int), "Lower bound on "+param_dict[j]+" must be a float or int"
                    assert isinstance(config.value(param_prefix+"bounds."+param_dict[j])[1], float) or isinstance(config.value(param_prefix+"bounds."+param_dict[j])[1], int), "Upper bound on "+param_dict[j]+" must be a float or int"
                    assert config.value(param_prefix+"bounds."+param_dict[j])[0]<config.value(param_prefix+"bounds."+param_dict[j])[1], "Lower bound on "+param_dict[j]+" must be less than upper bound"
        #Length of parameter vector for each individual pixel
        plength=len(param_dict)
        #Check theta length
        if(one_sed==True):
            if(plength!=len(theta)):
                self.log.error("Theta size incorrect. Please provide an initial guess for each parameter to be fit. ")
                return
        else:
            if(len(theta)%plength!=0):
                self.log.error("Theta size incorrect. Please provide an initial guess for each parameter to be fit for each pixel. ")
                return
        #Define the cosmology used later to get distances
        cosmo=FlatLambdaCDM(H0=h0, Om0=omegaM)
        #Open images and store the data and headers in lists. This speeds up the code by only opening the fits files/accessing the database once
        # Should take very little time per image, but may depend on where you pull the files from
        if(verbose==True):
            self.log.info("Retrieving image and bandpass files")
        #Get the list of images
        if(local==True):
            #Get the list of images
            if(isinstance(spec_data, str)):
                specs=glob.glob(spec_data+"*.fits")
            elif(isinstance(spec_data, list)):
                specs=spec_data
            elif(spec_data==None):
                specs=[]
            else:
                self.log.error("Invalid entry for spectroscopy data. Please provide either a path or a list of files, or set to None if not using spectroscopy")
                return
            if(isinstance(phot_data, str)):
                phots=glob.glob(phot_data+"*.fits")
            elif(isinstance(phot_data, list)):
                phots=phot_data
            elif(phot_data==None):
                phots=[]
            else:
                self.log.error("Invalid entry for photometry data. Please provide either a path or a list of files, or set to None if not using photometry")
                return
            #Check that at least some data is provided
            if(len(specs)==0 and len(phots)==0):
                self.log.error("Please provide a path to photometric and/or spectroscopic data or a list of files")
                return
            #Open the fits files and put the data into lists
            # Any fits files that do not open properly will be assumed to be invalid and will be skipped, but will NOT stop the program (unless it is the only file)
            spec_data_list=[]
            spec_err_list=[]
            if(len(specs)>0):
                for i in range(0, len(specs)):
                    try:
                        temp=fits.open(specs[i])
                        spec_data_list.append(temp[1].data)
                        spec_err_list.append(temp[2].data)
                        temp.close()
                    except:
                        self.log.error("Fits file "+str(specs[i])+" is invalid and will not be included in the fit. If this is the only file in the fit, the fit will fail")
            phot_data_list=[]
            phot_hdr_list=[]
            phot_err_list=[]
            if(len(phots)>0):
                for i in range(0, len(phots)):
                    try:
                        temp=fits.open(phots[i])
                        phot_data_list.append(temp[1].data)
                        phot_hdr_list.append(temp[1].header)
                        filt=temp[1].header["FILTER"]
                        phot_err_list.append(temp[2].data)
                        temp.close()
                        if (filt not in ["R062", "Z087", "Y106", "J129", "W146", "H158", "F184", "K213", "F062", "F087", "F106", "F129", "F146", "F158", "F213", "062", "087", "106", "129", "146", "158", "184", "213"]):
                            self.log.error("Filter "+str(filt)+" in file "+str(phots[i])+" is not valid. Valid filters are R062, Z087, Y106, J129, W146, H158, F184, and K213")
                            return
                    except:
                        self.log.error("Fits file "+str(phots[i])+" is invalid and will not be included in the fit. If this is the only file in the fit, the fit will fail")
            if(len(phot_data_list)==0 and len(spec_data_list)==0):
                self.log.error("No valid files entered. Please enter at least one valid fits file")
                return
        else:
            #If no spectrum data provided, automatically pull all images containing the RA and DEC that fit the parameters specificied in the configuration file
            if(spec_data==None):
                if(self.ra!=None and self.dec!=None):
                    if (min_mjd==None or max_mjd==None):
                        self.log.error("For automatic data retrieval, please provide a date range")
                        return
                    specs=Images.find_images(provenance_tag=provenance_tag, process=spec_process, ra=self.ra, dec=self.dec, mjd_min=mjd_min, mjd_max=mjd_max)
                else:
                    self.log.error("RA and DEC must be provided to automatically retrieve images")
                    return
            #If specific data is provided as a list, retrieve those images only
            elif(isinstance(spec_data, list)):
                specs=[]
                for i in range(0, len(spec_data)):
                    #Checks that the elements are valid
                    assert isinstance(spec_data[i], str) or isinstance(spec_data[i], pathlib.Path), "Each element of spec_data must be a string or a path"
                    #First assumes the list element is a path
                    test_im=Images.find_images(filepath=spec_data[i])
                    #If the path doesn't work, try the list element as a UUID
                    if(test_im==None):
                        assert isinstance(spec_data[i], str), str(spec_data[i])+" is not a valid UUID"
                        test_im=Images.get_image(spec_data[i])
                    if(test_im==None):
                        self.log.error("List element "+str(spec_data[i])+" does not correspond to a valid image and will not be used in fitting")
                    else:
                        specs.append(test_im)
            else:
                self.log.error("spec_data must be a list if provided")
            #If no photometry data provided, automatically pull all images containing the RA and DEC that fit the parameters specificied in the configuration file
            if(phot_data==None):
                if(self.ra!=None and self.dec!=None):
                    if (min_mjd==None or max_mjd==None):
                        self.log.error("For automatic data retrieval, please provide a date range")
                        return
                    phots=Images.find_images(provenance_tag=provenance_tag, process=phot_process, ra=self.ra, dec=self.dec, mjd_min=mjd_min, mjd_max=mjd_max)
                else:
                    self.log.error("RA and DEC must be provided to automatically retrieve images")
                    return
            #If specific data is provided as a list, retrieve those images only
            elif(isinstance(phot_data, list)):
                phots=[]
                for i in range(0, len(phot_data)):
                    #Checks that the elements are valid
                    assert isinstance(phot_data[i], str) or isinstance(phot_data[i], pathlib.Path), "Each element of phot_data must be a string or a path"
                    #First assumes the list element is a path
                    test_im=Images.find_images(filepath=phot_data[i])
                    #If the path doesn't work, try the list element as a UUID
                    if(test_im==None):
                        assert isinstance(phot_data[i], str), str(phot_data[i])+" is not a valid UUID"
                        test_im=Images.get_image(phot_data[i])
                    if(test_im==None):
                        self.log.error("List element "+str(phot_data[i])+" does not correspond to a valid image and will not be used in fitting")
                    else:
                        phots.append(test_im)
            else:
                self.log.error("phot_data must be a list if provided")
            #Check that at least some data is provided
            if(len(specs)==0 and len(phots)==0):
                self.log.error("Please provide either spectroscopic or photometric data by inputting either RA/DEC coodrinates or a list of filepaths and/or UUIDs")
                return
        #Get the data and put it into lists. Prevents us from retrieving these over and over again. 
        # Any files that do not open properly will be assumed to be invalid and will be skipped, but will NOT stop the program (unless it is the only file)
        spec_data_list=[]
        spec_err_list=[]
        spec_size_list=[]
        spec_WCS_list=[]
        if(len(specs)>0): 
            if(local==True):
                for i in range(0, len(specs)):
                    try:
                        temp=fits.open(specs[i])
                        temp_hdr=temp[1].header
                        spec_data_list.append(temp[1].data)
                        spec_err_list.append(temp[2].data)
                        spec_size_list.append([temp_hdr["NAXIS1"], temp_hdr["NAXIS2"]])
                        spec_WCS_list.append(WCS(temp_hdr))
                        temp.close()
                    except:
                        self.log.error("Fits file "+str(specs[i])+" is invalid and will not be included in the fit. If this is the only file in the fit, the fit will fail")
            else:
                spec_data_list.append(specs[i].data)
                spec_err_list.append(specs[i].noise)
                spec_size_list.append(specs[i].image_shape)
                spec_WCS_list.append(specs[i].get_wcs().get_astropy_wcs())
        phot_data_list=[]
        phot_err_list=[]
        phot_filt_list=[]
        phot_size_list=[]
        phot_WCS_list=[]
        if(len(phots)>0):
            for i in range(0, len(phots)):
                if(local==True):
                    for i in range(0, len(phots)):
                        try:
                            temp=fits.open(phots[i])
                            temp_hdr=temp[1].header
                            phot_data_list.append(temp[1].data)
                            phot_err_list.append(temp[2].data)
                            phot_filt_list.append(temp_hdr["FILTER"])
                            phot_size_list.append([temp_hdr(["NAXIS1"]), temp_hdr["NAXIS2"]])
                            phot_wcs_list.append(WCS(temp_hdr))
                            temp.close()
                            if (temp_hdr["FILTER"] not in ["R062", "Z087", "Y106", "J129", "W146", "H158", "F184", "K213", "F062", "F087", "F106", "F129", "F146", "F158", "F213", "062", "087", "106", "129", "146", "158", "184", "213"]):
                                self.log.error("Filter "+str(filter)+" in file "+str(phots[i])+" is not valid. Valid filters are R062, Z087, Y106, J129, W146, H158, F184, and K213")
                                return
                        except:
                            self.log.error("Fits file "+str(phots[i])+" is invalid and will not be included in the fit. If this is the only file in the fit, the fit will fail")
                else:
                    phot_data_list.append(phots[i].data)
                    phot_err_list.append(phots[i].noise)
                    phot_filt_list.append(phots[i].band)
                    phot_size_list.append(phots[i].image_shape)
                    phot_WCS_list.append(phots[i].get_wcs().get_astropy_wcs())
        # Load in bandpasses if photometry is used
        # Note: requires all 8 filter functions to be provided.
        if len(phots)!=0:
            if filter_path == None:
                self.log.error("Please provide path to Roman filter functions")
                return
            else:
                assert isinstance(filter_path, str), "filter_path must be a string"
            try:
                bp_062 = SpectralElement.from_file(filter_path + "Roman_WFI.F062.dat")
            except:
                self.log.error("Filter 062 is not in the filter file")
                return
            try:
                bp_087 = SpectralElement.from_file(filter_path + "Roman_WFI.F087.dat")
            except:
                self.log.error("Filter 087 is not in the filter file")
                return
            try:
                bp_106 = SpectralElement.from_file(filter_path + "Roman_WFI.F106.dat")
            except:
                self.log.error("Filter 106 is not in the filter file")
                return
            try:
                bp_129 = SpectralElement.from_file(filter_path + "Roman_WFI.F129.dat")
            except:
                self.log.error("Filter 129 is not in the filter file")
                return
            try:
                bp_146 = SpectralElement.from_file(filter_path + "Roman_WFI.F146.dat")
            except:
                self.log.error("Filter 146 is not in the filter file")
                return
            try:
                bp_158 = SpectralElement.from_file(filter_path + "Roman_WFI.F158.dat")
            except:
                self.log.error("Filter 158 is not in the filter file")
                return
            try:
                bp_184 = SpectralElement.from_file(filter_path + "Roman_WFI.F184.dat")
            except:
                self.log.error("Filter 184 is not in the filter file")
                return
            try:
                bp_213 = SpectralElement.from_file(filter_path + "Roman_WFI.F213.dat")
            except:
                self.log.error("Filter 213 is not in the filter file")
                return
            #if local==True:
            #    if filter_path == None:
            #        self.log.error("Please provide path to Roman filter functions")
            #        return
            #    else:
            #        assert isinstance(filter_path, str), "filter_path must be a string"
            #    try:
            #        bp_062 = SpectralElement.from_file(filter_path + "Roman_WFI.F062.dat")
            #    except:
            #        self.log.error("Filter 062 is not in the filter file")
            #        return
            #    try:
            #        bp_087 = SpectralElement.from_file(filter_path + "Roman_WFI.F087.dat")
            #    except:
            #        self.log.error("Filter 087 is not in the filter file")
            #        return
            #    try:
            #        bp_106 = SpectralElement.from_file(filter_path + "Roman_WFI.F106.dat")
            #    except:
            #        self.log.error("Filter 106 is not in the filter file")
            #        return
            #    try:
            #        bp_129 = SpectralElement.from_file(filter_path + "Roman_WFI.F129.dat")
            #    except:
            #        self.log.error("Filter 129 is not in the filter file")
            #        return
            #    try:
            #        bp_146 = SpectralElement.from_file(filter_path + "Roman_WFI.F146.dat")
            #    except:
            #        self.log.error("Filter 146 is not in the filter file")
            #        return
            #    try:
            #        bp_158 = SpectralElement.from_file(filter_path + "Roman_WFI.F158.dat")
            #    except:
            #        self.log.error("Filter 158 is not in the filter file")
            #        return
            #    try:
            #        bp_184 = SpectralElement.from_file(filter_path + "Roman_WFI.F184.dat")
            #    except:
            #        self.log.error("Filter 184 is not in the filter file")
            #        return
            #    try:
            #        bp_213 = SpectralElement.from_file(filter_path + "Roman_WFI.F213.dat")
            #    except:
            #        self.log.error("Filter 213 is not in the filter file")
            #        return
            #else:
            # Temporary, replace when filters are in the database somewhere
        #Overlap the pixel grids with pypolyclip to deal with mismatched WCS. Calculated here so they are only calculated once
        #Also generate simulator objects for spectroscopy, again so this only has to be done once
        #Should take on the order of seconds to minutes per image, depending on the number of pixels present in the segmentation map, a little longer for spectroscopic images to create the simulator
        if(verbose==True):
            start=time.time()
            self.log.info("Calculating pixel overlaps")
        #Set the size of the pixel grid the other images will be superimposed on
        naxis=(self.seg_map_hdr["NAXIS1"], self.seg_map_hdr["NAXIS2"])
        #Now we do a clever trick to avoid having to polyclip everything, which takes hours
        # We first convert the segmentation map pixels into the data image, then make a box around those pixels. We only clip those pixels, then only include in the simulator ones that overlap with the segmentation map.
        # Should take on the order of seconds to minutes, depending on the number of pixels in the segmentation map
        spec_pixel_list=[]
        spec_sim_list=[]
        for i in range(0, len(spec_data_list)):
            p, s=overlap(ref_wcs, spec_wcs_list[i], spec_size_list[i][0], spec_size_list[i][1], pixPos, buffer, True, spec_data_list[i])
            spec_pix_list.append(p)
            spec_sim_list.append(s)
        phot_pixel_list=[]
        #Photometry does not require the full simulator, but we still need the segmentation map, so we save that on its own
        phot_map_list=[]
        for i in range(0, len(phot_data_list)):
            p, s=overlap(ref_wcs, spec_wcs_list[i], spec_size_list[i][0], spec_size_list[i][1], pixPos, buffer, False, spec_data_list[i])
            phot_map_list.append(s)
            phot_pixel_list.append(p)
        if(verbose==True):
            self.log.info("Time to overlap the pixels: "+str(datetime.timedelta(seconds=(time.time()-start))))
        #Define a log normal function for use later. Note that using the log normal here avoids underflow/overflow errors (mostly underflow)
        def log_normal(x, mu, sigma):
            return np.log((1/np.sqrt(2*np.pi*sigma**2)))-((x-mu)**2)/(2*sigma**2)
        #Define prior
        def log_prior(theta):
            #Lower bound on age. Must be slightly greater than zero to prevent errors in simulation. Can be set lower or higher if desired
            low_age=1e-6
            #Return zero if not using bounds. Bounds age, preventing it from being less than zero, to prevent errors in the modeling code
            if(use_bounds==False):
                if(one_sed==False):
                    for i in range(0, self.numPix):
                        p = theta[int(i * plength):int((i + 1) * plength)]
                        for j in param_dict.keys():
                            if(param_dict[j]=="age" or param_dict[j]=="tage"):
                                if(p[j]<low_age):
                                    return -np.inf
                else:
                    for j in param_dict.keys():
                        if(param_dict[j]=="age" or param_dict[j]=="tage"):
                            if(theta[j]<low_age):
                                return -np.inf
                return 0.0
            else:
                if(one_sed==False):
                    for i in range(0, self.numPix):
                        p = theta[int(i * plength):int((i + 1) * plength)]
                        for j in param_dict.keys():
                            #If a bound is provided, return -infinity if a parameter for any pixel is outside the given range
                            if(config.value(param_prefix+"bounds."+param_dict[j])!=None):
                                lower=float(config.value(param_prefix+"bounds."+param_dict[j])[0])
                                upper=float(config.value(param_prefix+"bounds."+param_dict[j])[1])
                                #If for some reason the user has set a limit on age less than zero, changes that to zero
                                if(param_dict[j]=="age" or param_dict[j]=="tage"):
                                    if(lower<low_age):
                                        lower=0
                                if(p[j]<lower or p[j]>upper):
                                    return -np.inf
                            #Even if age bound is not provided, prevent age from being less than zero to prevent errors in modeling code
                            elif(param_dict[j]=="age" or param_dict[j]=="tage"):
                                if(p[j]<low_age):
                                    return -np.inf
                else:
                    for j in param_dict.keys():
                        #If a bound is provided, return -infinity if a parameter for any pixel is outside the given range
                        if(config.value(param_prefix+"bounds."+param_dict[j])!=None):
                            if(theta[j]<float(config.value(param_prefix+"bounds."+param_dict[j])[0]) or theta[j]>float(config.value(param_prefix+"bounds."+param_dict[j])[1])):
                                return -np.inf
                        #Even if age bound is not provided, prevent age from being less than zero to prevent errors in modeling code
                        elif(param_dict[j]=="age" or param_dict[j]=="tage"):
                            if(theta[j]<low_age):
                                return -np.inf
                #If after checking every parameter in every pixel, they're all within bounds, return 0.0
                return 0.0
        # Define log likelihood. Makes spectra with the input parameters, simulates an image with those spectra, and then compares that with the provided data
        # Define the probability function
        def prob_makespec(theta):
            #Simulate the spectra for the underlying pixel grid, using the original segmentation map
            if(one_sed==True):
                params=theta[int(0 * plength) : int((0 + 1) * plength)]
                if(sim_code=="BC03"):
                    age_params = [params[-1]]
                    sfh_params = list(params[:-1])
                    spec_name = "test"
                    csp_name = working_dir + "param.txt"
                    # Make the spectra
                    if dust == False:
                        if sfh == 1 or sfh == -1:
                            make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, recyc=recyc, delete_in=True, full_name=working_dir+"test.txt")
                        elif sfh == 6:
                            make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, [file_names], age_params, delete_in=True, full_name=working_dir+"test.txt")
                        else:
                            make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, delete_in=True, full_name=working_dir+"test.txt")
                    else:
                        dust_params = list(sfh_params[-2:])
                        sfh_params = list(sfh_params[:-2])
                        if sfh == 1 or sfh == -1:
                            make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, dust_params=dust_params, recyc=recyc,delete_in=True, full_name=working_dir+"test.txt")
                        elif sfh == 6:
                            make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, [file_names], age_params, dust_params=dust_params, delete_in=True, full_name=working_dir+"test.txt")
                        else:
                            make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, dust_params=dust_params, delete_in=True, full_name=working_dir+"test.txt")
                elif(sim_code=="FSPS"):
                    for j in range(0, len(param_dict)):
                        sp.params[param_dict[j]] = params[j]
                    #This is kind of a cheat. tage is a parameter in sp but must also be input in making the spectrum, so here we set it with all the others, then pull it out for actually making the spectrum
                    tage=sp.params["tage"]
                    #Make the spectrum
                    spec = sp.get_spectrum(tage=tage, peraa=True)
                    spec = np.transpose(spec)
                    np.savetxt(working_dir+"test.txt", spec)
            else:
                for i in range(0, len(self.pixPos)):
                    params = theta[int(i * plength) : int((i + 1) * plength)]
                    #Simulate the spectra using the pixels from the segmentation map
                    if(sim_code=="BC03"):
                        age_params = [params[-1]]
                        sfh_params = list(params[:-1])
                        spec_name = "test_" + str(i)
                        csp_name = working_dir + "param_" + str(i) + ".txt"
                        full_name=working_dir+str(self.pixPos[i][1])+"_"+str(self.pixPos[i][0])+".txt"
                        #Make the spectra
                        if dust == False:
                            if sfh == 1 or sfh == -1:
                                make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, recyc=recyc, delete_in=True, full_name=full_name)
                            elif sfh == 6:
                                make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, [file_names[i]], age_params, delete_in=True, full_name=full_name)
                            else:
                                make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, delete_in=True, full_name=full_name)
                        else:
                            dust_params = list(sfh_params[-2:])
                            sfh_params = list(sfh_params[:-2])
                            if sfh == 1 or sfh == -1:
                                make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, dust_params=dust_params, recyc=recyc,delete_in=True, full_name=full_name)
                            elif sfh == 6:
                                make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, [file_names[i]], age_params, dust_params=dust_params, delete_in=True, full_name=full_name)
                            else:
                                make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, dust_params=dust_params, delete_in=True, full_name=full_name)
                    elif(sim_code=="FSPS"):
                        for j in range(0, len(param_dict)):
                            sp.params[param_dict[j]] = params[j]
                        #This is kind of a cheat. tage is a parameter in sp but must also be input in making the spectrum, so here we set it with all the others, then pull it out for actually making the spectrum
                        tage=sp.params["tage"]
                        # Make the spectrum
                        spec = sp.get_spectrum(tage=tage, peraa=True)
                        spec = np.transpose(spec)
                        np.savetxt(working_dir+str(self.pixPos[i][0])+"_"+str(self.pixPos[i][1])+".txt", spec)
                    #It should be impossible to get to this point but better safe than sorry
                    else:
                        self.log.error("Invalid simulation code choice. Please select either FSPS or BC03 (GALAXEV)")
                        return
            #Sum likelihoods for the spectroscopy data
            sumlikely=0
            for k in range(0, len(specs)):
                #Get the data
                data=spec_data_list[k]
                error=spec_err_list[k]
                pix=spec_pixel_list[k]
                #Retrieve the simulator object we made earlier
                test_sim=spec_sim_list[k]
                test_pixPos=np.array(np.transpose(np.where(test_sim.segMapData!=0)))
                #Multiply the spectra and add them to the simulator
                for q in range(0, len(test_pixPos)):
                    #Get spectra and multiply
                    #Parameters of the pixel
                    y=test_pixPos[q][0]
                    x=test_pixPos[q][1]
                    pix_params=[]
                    #Get the polyclip parameters
                    for t in range(0, len(pix)):
                        test_x=pix[t][0]
                        test_y=pix[t][1]
                        if(test_x==x and test_y==y):
                            pix_params=pix[t]
                    xc=pix_params[2]
                    yc=pix_params[3]
                    area=pix_params[4]
                    #Load in the first spectrum to get wavelength
                    if(one_sed==True):
                        first_spec=np.loadtxt(working_dir+"test.txt")
                    else:
                        first_spec=np.loadtxt(working_dir+str(xc[0])+"_"+str(yc[0])+".txt")
                    wave=np.array(first_spec[:, 0])
                    #Convert to cgs units
                    # Input units: L_solar/Å
                    # Dimensional analysis: (L_solar/Å)*(erg*s^-1/L_solar)*(1/cm^2)=erg/s/cm^2/Å
                    #Get luminosity distance from redshift
                    dist=cosmo.luminosity_distance(z).value*3.08567758128e24 #Mpc to cm
                    temp_flux=first_spec[:, 1]*3.826e33*(1/(4*np.pi*dist**2))
                    total_flux=np.array(area[0]*temp_flux)
                    #Sum over all included pixels
                    for r in range(1, len(xc)):
                        if(one_sed==True):
                            temp_flux=first_spec[:, 1]
                        else:
                            spec_temp=np.loadtxt(working_dir+str(xc[r])+"_"+str(yc[r])+".txt")
                            temp_flux=spec_temp[:, 1]
                        flux=temp_flux*3.826e33*(1/(4*np.pi*dist**2))
                        total_flux=total_flux+(area[r]*flux)
                    #Add the spectrum to the simulator and save the file
                    out_data=np.transpose(np.array([wave, total_flux]))
                    np.savetxt(working_dir+str(x)+"_"+str(y)+"_data.txt", out_data)
                    test_sim.sourceColl[0].seds[q]=test_sim.sourceColl[0].seds[q].from_file(working_dir+str(x)+"_"+str(y)+"_data.txt")
                    test_sim.sourceColl[0].seds[q].redshift(z)
                # Do the simulation
                test = test_sim.simulate(0, return_sim=True, save_fits=False, redo_objpix=True)
                # Calculate log likelihood as the sum of normals
                mask = np.where(test != 0)
                test_seg=np.zeros((spec_size_list[k][1], spec_size_list[k][0]))
                test_seg[mask]=1
                is_in=np.abs(data*test_seg)
                if(np.max(is_in)<=0):
                    self.log.error("File "+str(specs[k])+" does not produce a simulation that overlaps with the desired output. If your fit fails, this is one likely cause. Check your segmentation map and fits file headers to ensure the fit object and the data match")
                for i in range(0, len(mask[0])):
                    sumlikely = sumlikely + log_normal(test[mask[0][i], mask[1][i]], data[mask[0][i], mask[1][i]], error[mask[0][i], mask[1][i]])
                    if(np.isfinite(sumlikely)!=True):
                        self.log.error("There is a likelihood value that is either infinite or NaN. If your fit fails, this is one likely cause. Check error file for unusually small errors")
            #Sum likelihoods for the photometry
            for k in range(0, len(phots)):
                data=phot_data_list[k]
                error=phot_err_list[k]
                filter=phot_filt_list[k]
                pix=phot_pixel_list[k]
                smap=phot_map_list[k]
                test_im=np.zeros(self.seg_map_data.shape)
                #Get the pixel positions
                test_pixPos=np.array(np.transpose(np.where(smap!=0)))
                #Multiply the spectra and calculate the synthetic photometry, then compare to the data
                for q in range(0, len(test_pixPos)):
                    #Get spectra and multiply
                    #Parameters of the pixel
                    y=test_pixPos[q][0]
                    x=test_pixPos[q][1]
                    pix_params=[]
                    #Get the polyclip parameters
                    for t in range(0, len(pix)):
                        test_x=pix[t][0]
                        test_y=pix[t][1]
                        if(test_x==x and test_y==y):
                            pix_params=pix[t]
                    xc=pix_params[2]
                    yc=pix_params[3]
                    area=pix_params[4]
                    #Load in the first spectrum to get wavelength
                    if(one_sed==True):
                        first_spec=np.loadtxt(working_dir+"test.txt")
                    else:
                        first_spec=np.loadtxt(working_dir+str(xc[0])+"_"+str(yc[0])+".txt")
                    wave=np.array(first_spec[:, 0])
                    #Convert to CGS units
                    dist=cosmo.luminosity_distance(z).value*3.08567758128e24 #Mpc to cm
                    temp_flux=first_spec[:, 1]*3.826e33*(1/(4*np.pi*dist**2))
                    total_flux=np.array(area[0]*temp_flux)
                    #Sum over all included pixels
                    for r in range(1, len(xc)):
                        if(one_sed==True):
                            temp_flux=first_spec[:, 1]
                        else:
                            spec_temp=np.loadtxt(working_dir+str(xc[r])+"_"+str(yc[r])+".txt")
                            temp_flux=spec_temp[:, 1]
                        flux=temp_flux*3.826e33*(1/(4*np.pi*dist**2))
                        total_flux=total_flux+(area[r]*flux)
                    #Save the file
                    out_data=np.transpose(np.array([wave, total_flux]))
                    np.savetxt(working_dir+str(x)+"_"+str(y)+"_data.txt", out_data)
                    #Do the synthetic photometry
                    test_spec=SourceSpectrum.from_file(working_dir+str(x)+"_"+str(y)+"_data.txt")
                    test_spec.z=z
                    if(filter in ["R062", "062", "F062"]):
                        obs=Observation(test_spec, bp_062).effstim(flux_unit="flam").value
                    elif(filter in ["Z087", "087", "F087"]):
                        obs=Observation(test_spec, bp_087).effstim(flux_unit="flam").value
                    elif(filter in ["Y106", "106", "F106"]):
                        obs=Observation(test_spec, bp_106).effstim(flux_unit="flam").value
                    elif(filter in ["J129", "129", "F129"]):
                        obs=Observation(test_spec, bp_129).effstim(flux_unit="flam").value
                    elif(filter in ["W146", "146", "F146"]):
                        obs=Observation(test_spec, bp_146).effstim(flux_unit="flam").value
                    elif(filter in ["H158", "158", "F158"]):
                        obs=Observation(test_spec, bp_158).effstim(flux_unit="flam").value
                    elif(filter in ["F184", "184"]):
                        obs=Observation(test_spec, bp_184).effstim(flux_unit="flam").value
                    elif(filter in ["K213", "213", "F213"]):
                        obs=Observation(test_spec, bp_213).effstim(flux_unit="flam").value
                    else:
                        self.log.error(str(filter)+" is not a valid filter. Please check header of image "+str(phots[k])+" and ensure the filter is listed")
                        return
                    #Add the observation to the test image
                    test_im[y, x]=obs
                    #Calculate log likelihood
                    sumlikely=sumlikely+log_normal(obs, data[y, x], error[y, x])
            if spi == True:
                return -sumlikely
            else:
                return sumlikely
        #Define total log probability
        def log_prob(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            pm=prob_makespec(theta)
            if(pm==None):
                self.log.error("Error in log likelihood function. Check log for relevant error message")
                return
            return lp + pm
        #Run initial optimization with scipy least squares or other provided optimization method
        # This may take a lot of time, depending on how big the galaxy is, which simulation code you choose, and how many parameters you want to fit
        if(verbose==True):
            self.log.info("Performing fit")
            start=time.time()
        spi = True
        minimum = 0.0
        if method == None:
            #minimum = fmin(log_prob, theta)
            minimum = least_squares(log_prob, theta)
        elif method in ["trf", "dogbox", "lm"]:
            minimum = least_squares(log_prob, theta, method=method)
        elif method in ["Nelder-Mead", "Powell", "CG", "BFGS", "L-BFGS-B", "TNC", "COBYLA", "COBYQA", "SLSQP", "trust-constr"]:
            minimum = minimize(log_prob, theta, method=method)
        elif method in ["Newton-CG", "dogleg", "trust-ncg", "trust-exact", "trust-krylov"]:
            self.log.error("Requires callable Jacobian. Not currently supported")
            return
            #if jac == None:
            #    log.error("Callable Jacobian required for these methods")
            #    return
            #else:
            #    minimum = minimize(log_prob, theta, method=method, jac=jac)
        else:
            self.log.error("Invalid method")
            return
        if(minimum.success==False):
            self.log.error("Optimization failed. Please examine the object returned of this function, which is the entire output of the scipy minimization, to find the issue")
            return minimum
        if(verbose==True):
            self.log.info("Time to perform the fit: "+str(datetime.timedelta(seconds=(time.time()-start))))
        #Make the image with the best fit parameters and calculate the chi^2
        if(verbose==True):
            start=time.time()
            self.log.info("Simulating best-fit image")
        best_fit=minimum.x
        #Put together a map of the properties
        hdu0=fits.PrimaryHDU()
        hdul=fits.HDUList([hdu0])
        count=0
        for k in param_dict.keys():
            data=np.zeros((seg_map_hdr["NAXIS2"], seg_map_hdr["NAXIS1"]))
            for l in range(0, len(self.pixPos)):
                x=self.pixPos[l][1]
                y=self.pixPos[l][0]
                if(one_sed==True):
                    data[y, x]=best_fit[count]
                else:
                    data[y, x]=best_fit[int((l*plength)+count)]
            new_header=seg_map_hdr
            new_header["PROPERTY"]=k
            new_header["COMMENT"]="Image created using the script fit.py from the Roman SNPIT project"
            new_header["COMMENT"]="Best fit properties for each pixel"
            hdu=fits.ImageHDU(data, new_header)
            hdul.append(hdu)
            count=count+1
        #Convert the underlying pixels to the SN image
        sn_pixels, sn_sim=overlap(self.ref_wcs, self.sn_wcs, self.sn_size[0], self.sn_size[1], pixPos, buffer, True, self.sn_data)
        #Simulate the spectra for the pixels using the best fit parameters
        if(one_sed==True):
            params=best_fit[int(0 * plength) : int((0 + 1) * plength)]
            if(sim_code=="BC03"):
                age_params = [params[-1]]
                sfh_params = list(params[:-1])
                spec_name = "test"
                csp_name = ised_dir + "param.txt"
                # Make the spectra
                if dust == False:
                    if sfh == 1 or sfh == -1:
                        make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, recyc=recyc, delete_in=True, full_name=working_dir+"test.txt")
                    elif sfh == 6:
                        make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, [file_names[0]], age_params, delete_in=True, full_name=working_dir+"test.txt")
                    else:
                        make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, delete_in=True, full_name=working_dir+"test.txt")
                else:
                    dust_params = list(sfh_params[-2:])
                    sfh_params = list(sfh_params[:-2])
                    if sfh == 1 or sfh == -1:
                        make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, dust_params=dust_params, recyc=recyc,delete_in=True, full_name=working_dir+"test.txt")
                    elif sfh == 6:
                        make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, [file_names[i]], age_params, dust_params=dust_params, delete_in=True, full_name=working_dir+"test.txt")
                    else:
                        make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, dust_params=dust_params, delete_in=True, full_name=working_dir+"test.txt")
                #This is kind of a cheat and probably not best practice but I don't have a better idea
                temp=np.loadtxt(working_dir+"test.txt")
                dist=cosmo.luminosity_distance(z).value*3.08567758128e24 #Mpc to cm
                new_flux=temp[:, 1]*3.826e33*(1/(4*np.pi*dist**2))
                new_spec=np.transpose(np.array([temp[:, 0], new_flux]))
                np.savetxt(working_dir+"test.txt", new_spec)
            elif(sim_code=="FSPS"):
                for j in range(0, len(param_dict)):
                    sp.params[param_dict[j]] = params[j]
                #This is kind of a cheat. tage is a parameter in sp but must also be input in making the spectrum, so here we set it with all the others, then pull it out for actually making the spectrum
                tage=sp.params["tage"]
                # Make the spectrum
                spec = sp.get_spectrum(tage=tage, peraa=True)
                spec = np.transpose(spec)
                dist=cosmo.luminosity_distance(z).value*3.08567758128e24 #Mpc to cm
                new_flux=spec[:, 1]*3.826e33*(1/(4*np.pi*dist**2))
                new_spec=np.transpose(np.array([spec[:, 0], new_flux]))
                np.savetxt(working_dir+"test.txt", new_spec)
        else:
            for i in range(0, len(self.pixPos)):
                params = best_fit[int(i * plength) : int((i + 1) * plength)]
                #Simulate the spectra using the pixels from the segmentation map
                if(sim_code=="BC03"):
                    age_params = [params[-1]]
                    sfh_params = list(params[:-1])
                    spec_name = "test_" + str(i)
                    csp_name = ised_dir + "param_" + str(i) + ".txt"
                    full_name=working_dir+str(self.pixPos[i][1])+"_"+str(self.pixPos[i][0])+".txt"
                    #Make the spectra
                    if dust == False:
                        if sfh == 1 or sfh == -1:
                            make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, recyc=recyc, delete_in=True, full_name=full_name)
                        elif sfh == 6:
                            make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, [file_names[i]], age_params, delete_in=True, full_name=full_name)
                        else:
                            make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, delete_in=True, full_name=full_name)
                    else:
                        dust_params = list(sfh_params[-2:])
                        sfh_params = list(sfh_params[:-2])
                        if sfh == 1 or sfh == -1:
                            make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, dust_params=dust_params, recyc=recyc,delete_in=True, full_name=full_name)
                        elif sfh == 6:
                            make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, [file_names[i]], age_params, dust_params=dust_params, delete_in=True, full_name=full_name)
                        else:
                            make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, dust_params=dust_params, delete_in=True, full_name=full_name)
                    #This is definitely not the best way to do this but I don't have a better idea
                    temp=np.loadtxt(full_name)
                    dist=cosmo.luminosity_distance(z).value*3.08567758128e24 #Mpc to cm
                    new_flux=temp[:, 1]*3.826e33*(1/(4*np.pi*dist**2))
                    new_spec=np.transpose(np.array([temp[:, 0], new_flux]))
                    np.savetxt(full_name, new_spec)
                elif(sim_code=="FSPS"):
                    for j in range(0, len(param_dict)):
                        sp.params[param_dict[j]] = params[j]
                    #This is kind of a cheat. tage is a parameter in sp but must also be input in making the spectrum, so here we set it with all the others, then pull it out for actually making the spectrum
                    tage=sp.params["tage"]
                    # Make the spectrum
                    spec = sp.get_spectrum(tage=tage, peraa=True)
                    spec = np.transpose(spec)
                    dist=cosmo.luminosity_distance(z).value*3.08567758128e24 #Mpc to cm
                    new_flux=spec[:, 1]*3.826e33*(1/(4*np.pi*dist**2))
                    new_spec=np.transpose(np.array([spec[:, 0], new_flux]))
                    np.savetxt(working_dir+str(self.pixPos[i][0])+"_"+str(self.pixPos[i][1])+".txt", new_spec)
        #Add the spectra to the simulator object
        #Get the data
        data=spec_data_list[k]
        error=spec_err_list[k]
        pix=spec_pixel_list[k]
        #Multiply the spectra and add them to the simulator
        for q in range(0, len(sn_pixPos)):
            #Get spectra and multiply
            #Parameters of the pixel
            y=sn_pixPos[q][0]
            x=sn_pixPos[q][1]
            pix_params=[]
            #Get the polyclip parameters
            for t in range(0, len(pix)):
                test_x=pix[t][0]
                test_y=pix[t][1]
                if(test_x==x and test_y==y):
                    pix_params=pix[t]
            xc=pix_params[2]
            yc=pix_params[3]
            area=pix_params[4]
            #Load in the first spectrum to get wavelength
            if(one_sed==True):
                first_spec=np.loadtxt(working_dir+"test.txt")
            else:
                first_spec=np.loadtxt(working_dir+str(xc[0])+"_"+str(yc[0])+".txt")
            wave=np.array(first_spec[:, 0])
            #Convert to cgs units
            # Input units: L_solar/Å
            # Dimensional analysis: (L_solar/Å)*(erg*s^-1/L_solar)*(1/cm^2)=erg/s/cm^2/Å
            #Get luminosity distance from redshift
            dist=cosmo.luminosity_distance(z).value*3.08567758128e24 #Mpc to cm
            temp_flux=first_spec[:, 1]*3.826e33*(1/(4*np.pi*dist**2))
            total_flux=np.array(area[0]*temp_flux)
            #Sum over all included pixels
            for r in range(1, len(xc)):
                if(one_sed==True):
                    temp_flux=first_spec[:, 1]
                else:
                    spec_temp=np.loadtxt(working_dir+str(xc[r])+"_"+str(yc[r])+".txt")
                    temp_flux=spec_temp[:, 1]
                flux=temp_flux*3.826e33*(1/(4*np.pi*dist**2))
                total_flux=total_flux+(area[r]*flux)
            #Add the spectrum to the simulator and save the file
            out_data=np.transpose(np.array([wave, total_flux]))
            np.savetxt(working_dir+str(x)+"_"+str(y)+"_data.txt", out_data)
            test_sim.sourceColl[0].seds[q]=test_sim.sourceColl[0].seds[q].from_file(working_dir+str(x)+"_"+str(y)+"_data.txt")
            test_sim.sourceColl[0].seds[q].redshift(z)    
        for q in range(0, len(self.pixPos)):
            if(one_sed==True):
                self.simulator.sourceColl[0].seds[q]=self.simulator.sourceColl[0].seds[q].from_file(working_dir+"test.txt")
                self.simulator.sourceColl[0].seds[q].redshift(z)
            else:
                x=self.pixPos[q][1]
                y=self.pixPos[q][0]
                self.simulator.sourceColl[0].seds[q]=self.simulator.sourceColl[0].seds[q].from_file(working_dir+str(x)+"_"+str(y)+"_data.txt")
                self.simulator.sourceColl[0].seds[q].redshift(z)
        #Do the simulation
        best_im=self.simulator.simulate(0, return_sim=True, save_fits=False, redo_objpix=True)
        if(verbose==True):
            self.log.info("Time to simulate image: "+str(datetime.timedelta(seconds=(time.time()-start))))
        #Define chi^2
        def chi2():
            chi2=0
            for i in range(0, len(specs)):
                pix=spec_pixel_list[i]
                data=spec_data_list[i]
                err=spec_err_list[i]
                #Retrieve the simulator object
                test_sim=spec_sim_list[i]
                test_pixPos=np.array(np.transpose(np.where(test_sim.segMapData!=0)))
                #Multiply the spectra and add them to the simulator
                for q in range(0, len(test_pixPos)):
                    #Get spectra and multiply
                    #Parameters of the pixel
                    y=test_pixPos[q][0]
                    x=test_pixPos[q][1]
                    pix_params=[]
                    #Get the polyclip parameters
                    for t in range(0, len(pix)):
                        test_x=pix[t][0]
                        test_y=pix[t][1]
                        if(test_x==x and test_y==y):
                            pix_params=pix[t]
                    xc=pix_params[2]
                    yc=pix_params[3]
                    area=pix_params[4]
                    #Load in the first spectrum to get wavelength
                    if(one_sed==True):
                        first_spec=np.loadtxt(working_dir+"test.txt")
                    else:
                        first_spec=np.loadtxt(working_dir+str(xc[0])+"_"+str(yc[0])+".txt")
                    wave=np.array(first_spec[:, 0])
                    dist=cosmo.luminosity_distance(z).value*3.08567758128e24 #Mpc to cm
                    temp_flux=first_spec[:, 1]*3.826e33*(1/(4*np.pi*dist**2))
                    total_flux=np.array(area[0]*temp_flux)
                    #Sum over all included pixels
                    for r in range(1, len(xc)):
                        if(one_sed==True):
                            temp_flux=first_spec[:, 1]
                        else:
                            spec_temp=np.loadtxt(working_dir+str(xc[r])+"_"+str(yc[r])+".txt")
                            temp_flux=spec_temp[:, 1]
                        flux=temp_flux*3.826e33*(1/(4*np.pi*dist**2))
                        total_flux=total_flux+(area[r]*flux)
                    #Add the spectrum to the simulator and save the file
                    out_data=np.transpose(np.array([wave, total_flux]))
                    np.savetxt(working_dir+str(x)+"_"+str(y)+"_best_fit.txt", out_data)
                    test_sim.sourceColl[0].seds[q]=test_sim.sourceColl[0].seds[q].from_file(working_dir+str(x)+"_"+str(y)+"_data.txt")
                    test_sim.sourceColl[0].seds[q].redshift(z)
                # Do the simulation
                test = test_sim.simulate(0, return_sim=True, save_fits=False, redo_objpix=True)
                mask=np.where(test!=0)
                #Calculate chi2
                chi2=chi2+np.sum(((data[mask]-test[mask])/err[mask])**2)
            for i in range(0, len(phots)):
                data=phot_data_list[i]
                error=phot_err_list[i]
                filter=phot_filt_list[i]
                pix=phot_pixel_list[i]
                map=phot_map_list[i]
                #Get the pixel positions
                test_pixPos=np.array(np.transpose(np.where(map!=0)))
                #Multiply the spectra and calculate the synthetic photometry, then compare to the data
                for q in range(0, len(test_pixPos)):
                    #Get spectra and multiply
                    #Parameters of the pixel
                    y=test_pixPos[q][0]
                    x=test_pixPos[q][1]
                    pix_params=[]
                    #Get the polyclip parameters
                    for t in range(0, len(pix)):
                        test_x=pix[t][0]
                        test_y=pix[t][1]
                        if(test_x==x and test_y==y):
                            pix_params=pix[t]
                    xc=pix_params[2]
                    yc=pix_params[3]
                    area=pix_params[4]
                    #Load in the first spectrum to get wavelength
                    if(one_sed==True):
                        first_spec=np.loadtxt(working_dir+"test.txt")
                    else:
                        first_spec=np.loadtxt(working_dir+str(xc[0])+"_"+str(yc[0])+".txt")
                    wave=np.array(first_spec[:, 0])
                    dist=cosmo.luminosity_distance(z).value*3.08567758128e24 #Mpc to cm
                    temp_flux=first_spec[:, 1]*3.826e33*(1/(4*np.pi*dist**2))
                    total_flux=np.array(area[0]*temp_flux)
                    #Sum over all included pixels
                    for r in range(1, len(xc)):
                        if(one_sed==True):
                            temp_flux=first_spec[:, 1]
                        else:
                            spec_temp=np.loadtxt(working_dir+str(xc[r])+"_"+str(yc[r])+".txt")
                            temp_flux=spec_temp[:, 1]
                        flux=temp_flux*3.826e33*(1/(4*np.pi*dist**2))
                        total_flux=total_flux+(area[r]*flux)
                    #Do the synthetic photometry
                    test_spec=SourceSpectrum.from_file(working_dir+str(x)+"_"+str(y)+"_data.txt")
                    test_spec.z=z
                    if(filter in ["R062", "062", "F062"]):
                        obs=Observation(test_spec, bp_062).effstim(flux_unit="flam").value
                    elif(filter in ["Z087", "087", "F087"]):
                        obs=Observation(test_spec, bp_087).effstim(flux_unit="flam").value
                    elif(filter in ["Y106", "106", "F106"]):
                        obs=Observation(test_spec, bp_106).effstim(flux_unit="flam").value
                    elif(filter in ["J129", "129", "F129"]):
                        obs=Observation(test_spec, bp_129).effstim(flux_unit="flam").value
                    elif(filter in ["W146", "146", "F146"]):
                        obs=Observation(test_spec, bp_146).effstim(flux_unit="flam").value
                    elif(filter in ["H158", "158", "F158"]):
                        obs=Observation(test_spec, bp_158).effstim(flux_unit="flam").value
                    elif(filter in ["F184", "184"]):
                        obs=Observation(test_spec, bp_184).effstim(flux_unit="flam").value
                    elif(filter in ["K213", "213", "F213"]):
                        obs=Observation(test_spec, bp_213).effstim(flux_unit="flam").value
                    else:
                        self.log.error(str(filter)+" is not a valid filter. Please check header of image "+str(phots[k])+" and ensure the filter is listed")
                        return
                    chi2=chi2+((data[y, x]-obs)/error[y, x])**2
            return chi2
        if(verbose==True):
            self.log.info("Calculating chi^2")
        chi=chi2()
        #Organize the requested outputs
        if(use_bayes==False):
            return_list=[]
            if(return_fit==True):
                return_list.append(param_dict.keys())
                return_list.append(minimum.x)
                return_list.append(chi)
            if(save_fit==True):
                if(local==True):
                    hdul.writeto("best_fit_properties.fits", overwrite=True)
                else:
                    #Temporary, until we have a file format for this
                    self.log.error("Saving the fit to the database not currently supported")
                    return
            if(return_image==True):
                return_list.append(best_im)
            if(save_image==True):
                if(local==True):
                    im_header=self.dir_im_hdr
                    im_header["COMMENT"]="Image created using the script fit.py from the Roman SNPIT project"
                    hdu = fits.PrimaryHDU(best_im, im_header)
                    if image_name == None:
                        hdu.writeto("best_fit.fits", overwrite=True)
                    else:
                        hdu.writeto(image_name + ".fits", overwrite=True)
                else:
                    #Temporary, until we have a format for 2-D spectra
                    self.log.error("Saving the best fit image not currently supported")
                    #im_header=self.seg_map_hdr
                    #im_header["COMMENT"]="Image created using the script fit.py from the Roman SNPIT project"
                    #hdu = fits.PrimaryHDU(best_im, im_header)
                    #if image_name == None:
                    #    hdu.writeto("best_fit.fits", overwrite=True)
                    #else:
                    #    hdu.writeto(image_name + ".fits", overwrite=True)
            if(return_subtracted==True or save_subtracted==True):
                sub_im=Image.find_images(filepath=ref_im)
                if(sub_im==None):
                    assert isinstance(ref_im, str), str(ref_im)+" is not a valid UUID"
                    sub_im=Image.get_image(ref_im)
                if(sub_im==None):
                    self.log.error("The reference image is not valid. Image subtraction cannot be performed")
                else:
                    sub_data=sub_im.data
                    sub=sub_data-best_im
                    if(return_subtracted==True):
                        return_list.append(sub)
                    if(save_subtracted==True):
                        if(local==True):
                            im_header=self.dir_im_hdr
                            im_header["COMMENT"]="Image created using the script fit.py from the Roman SNPIT project"
                            im_header["COMMENT"]="Host-galaxy subtracted image"
                            hdu = fits.PrimaryHDU(sub, im_header)
                            if subtracted_name == None:
                                hdu.writeto("best_fit_subtracted.fits", overwrite=True)
                            else:
                                hdu.writeto(subtracted_name + ".fits", overwrite=True)
                        else:
                            #Temporary, until we have a format for 2-D spectra
                            self.log.error("Saving the subtracted image not currently supported")
                            #im_header=self.seg_map_hdr
                            #im_header["COMMENT"]="Image created using the script fit.py from the Roman SNPIT project"
                            #im_header["COMMENT"]="Host-galaxy subtracted image"
                            #hdu = fits.PrimaryHDU(sub, im_header)
                            #if subtracted_name == None:
                            #    hdu.writeto("best_fit_subtracted.fits", overwrite=True)
                            #else:
                            #    hdu.writeto(subtracted_name + ".fits", overwrite=True)
            if(verbose==True):
                self.log.info("Fit success!")
                self.log.info("Total time to run: "+str(datetime.timedelta(seconds=(time.time()-big_start))))
            return return_list
        # Perform Bayesian analysis
        #   Not currently functional
        else:
            self.log.error("Bayesian statistics not currently supported")
            return
            # Starts analysis at best fit from previous optimization, with some small perturbations
            guess = minimum.x
            start = guess + 1e-3 * np.random.randn(nwalkers, len(guess))
            # Run MCMC
            sampler = emcee.EnsembleSampler(nwalkers, len(guess), log_prob)
            sampler.run_mcmc(start, niter, progress=True)
            # Find the best fit parameters as the median of the chain
            flat_samples = sampler.get_chain(flat=True)
            nparams = flat_samples.shape[1]
            best_fit = np.zeros(nparams)
            for i in range(0, nparams):
                best_fit[i] = np.median(flat_samples[:, i])
            # Makes the best fit image using the best fit parameters found above
            for i in range(0, self.numPix):
                params = best_fit[int(i * plength) : int((i + 1) * plength)]
                age_params = [params[-1]]
                sfh_params = list(params[:-1])
                spec_name = "test_" + str(i)
                csp_name = ised_dir + "param_" + str(i) + ".txt"
                if dust == False:
                    if (sfh == 1 or sfh == -1) and recyc == True:
                        make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, recyc=True, delete_in=True)
                    elif sfh == 6:
                        make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, [file_names[i]], age_params, delete_in=True)
                    else:
                        make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, delete_in=True)
                else:
                    dust_params = list(sfh_params[-2:])
                    sfh_params = list(sfh_params[:-2])
                    if (sfh == 1 or sfh == -1) and recyc == True:
                        make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, dust_params=dust_params, recyc=True, delete_in=True)
                    elif sfh == 6:
                        make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, [file_names[i]], age_params, dust_params=dust_params, delete_in=True)
                    else:
                        make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, dust_params=dust_params, delete_in=True)
                test_sim.sourceColl[0].seds[i] = (test_sim.sourceColl[0].seds[i].from_file(working_dir + spec_name + "_norm.sed"))
            best_im = simulator.simulate(0, return_sim=True, save_fits=False)
            # Save fits and return relevant parameters
            if save_fits == True:
                hdu = fits.PrimaryHDU(best_im, seg_map_hdr)
                if fits_name == None:
                    hdu.writeto("best_fit.fits", overwrite=True)
                else:
                    hdu.writeto(fits_name + ".fits", overwrite=True)
            if return_image == True:
                return minimum, best_im, sampler, best_fit
            else:
                return minimum, sampler, best_fit
