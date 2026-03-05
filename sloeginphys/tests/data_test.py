#Tests that the data needed to run the tests and to run sloeginphys has been installed and placed correctly. If this fails, something is wrong with the installation. Note that this does not check the contents of files, just their existence. 

from astropy.io import fits
import asdf
import os
import numpy as np
from snappl.config import Config

def test_schema():
    cwd=os.getcwd()
    working_dir=cwd+"/sloeginphys/data/"
    assert os.path.isfile(working_dir+"roman_schema_direct.yaml")
    assert os.path.isfile(working_dir+"roman_schema_segmap.yaml")
    assert os.path.isfile(working_dir+"roman_schema_sn.yaml")
    Config.get(working_dir+"roman_schema_direct.yaml")
    Config.get(working_dir+"roman_schema_segmap.yaml")
    Config.get(working_dir+"roman_schema_sn.yaml")

def test_fits():
    cwd=os.getcwd()
    working_dir=cwd+"/sloeginphys/tests/"
    assert os.path.isfile(working_dir+"data.fits")
    assert os.path.isfile(working_dir+"seg.fits")
    assert os.path.isfile(working_dir+"sn.fits")
    assert os.path.isfile(working_dir+"sn_bc03.fits")
    fits.open(working_dir+"data.fits")
    fits.open(working_dir+"seg.fits")
    fits.open(working_dir+"sn.fits")
    fits.open(working_dir+"sn_bc03.fits")

#def test_asdf():
#    cwd=os.getcwd()
#    working_dir=cwd+"/sloeginphys/tests/"
#    assert os.path.isfile(working_dir+"data.asdf")
#    assert os.path.isfile(working_dir+"seg.asdf")
#    assert os.path.isfile(working_dir+"sn.asdf")
#    assert os.path.isfile(working_dir+"sn_bc03.asdf")
#    asdf.open(working_dir+"data.asdf")
#    asdf.open(working_dir+"seg.asdf")
#    asdf.open(working_dir+"sn.asdf")
#    asdf.open(working_dir+"sn_bc03.asdf")

def test_misc():
    cwd=os.getcwd()
    working_dir=cwd+"/sloeginphys/tests/"
    assert os.path.isfile(working_dir+"config_bc03.yaml")
    assert os.path.isfile(working_dir+"config_fsps.yaml")
    assert os.path.isfile(working_dir+"bc2003_lr_BaSeL_m82_kroup_ssp.ised")
    Config.get(working_dir+"config_bc03.yaml")
    Config.get(working_dir+"config_fsps.yaml")
    open(working_dir+"bc2003_lr_BaSeL_m82_kroup_ssp.ised", "rb")