#Tests specific simulation codes, BC03 and FSPS. One of these pairs of tests must pass for sloeginphys to function. 

import numpy as np
import sloeginphys
from sloeginphys.fit import fitter
from astropy.io import fits
from astropy.wcs import WCS
import os
from astropy.cosmology import FlatLambdaCDM
import pytest

def test_bc03():
    cwd=os.getcwd()
    working_dir=cwd+"/sloeginphys/tests/"
    one_sed=True
    theta=[10]
    plength=1
    pixPos=[2044, 2044]
    ised_dir=cwd+"/sloeginphys/tests/"
    csp_params=["BaSeL", "m82", "kroup", False, 0]
    dust=False
    recyc=None
    file_names=None
    sloeginphys.fit_utils._make_SED_bc03(working_dir, one_sed, theta, plength, pixPos, ised_dir, csp_params, recyc, file_names)
    f=np.loadtxt(working_dir+"test.txt")
    os.system("rm -f "+working_dir+"test.[123456789w]*")
    os.system("rm -f "+working_dir+"fort*")
    os.system("rm -f "+working_dir+"*.tmp")
    os.system("rm -f "+working_dir+"test.*ed")
    os.system("rm -f "+working_dir+"bc03.rm")
    assert round(f[0, 1], 14)==pytest.approx(6.4188e-10) 
    assert round(f[500, 1], 9)==pytest.approx(1.5141e-05) 
    assert round(f[1000, 1], 10)==pytest.approx(7.2015e-06) 
    assert round(f[1500, 1], 11)==pytest.approx(2.2513e-07) 
    assert round(f[2000, 1], 16)==pytest.approx(2.0256e-12)  

def test_fit_bc03():
    f=sloeginphys.fit.fitter("./sloeginphys/tests/data.fits", "./sloeginphys/tests/sn_bc03.fits", local=True, segmap="./sloeginphys/tests/seg.fits")
    f.make_map(ra=7.4643751, dec=-44.79948193)
    test_min=f.fit(np.array([10]), "./sloeginphys/tests/config_bc03.yaml", spec_data=["./sloeginphys/tests/sn_bc03.fits"])
    working_dir=cwd+"/sloeginphys/tests/"
    os.system("rm -f "+working_dir+"*_data.txt")
    os.system("rm -f "+working_dir+"*_temp.txt")
    assert test_min[1][0]==10
    assert test_min[2]<1e-10

def test_fsps():
    import fsps
    working_dir="./sloeginphys/tests/"
    one_sed=True
    theta=[10]
    param_dict={0:"tage"}
    plength=1
    sp=fsps.StellarPopulation(compute_vega_mags=False, vactoair_flag=False)
    pixPos=[2044, 2044]
    sloeginphys.fit_utils._make_SED_fsps(working_dir, one_sed, theta, param_dict, plength, sp, pixPos)
    f=np.loadtxt(working_dir+"test.txt")
    assert round(f[1000, 1], 9)==pytest.approx(4.9186e-05)
    assert round(f[2000, 1], 9)==pytest.approx(4.6588e-05)
    assert round(f[3000, 1], 9)==pytest.approx(4.0473e-05)
    assert round(f[4000, 1], 9)==pytest.approx(3.4133e-05)
    assert round(f[5000, 1], 11)==pytest.approx(5.2415e-07)

def test_fit_fsps():
    f=sloeginphys.fit.fitter("./sloeginphys/tests/data.fits", "./sloeginphys/tests/sn.fits", local=True, segmap="./sloeginphys/tests/seg.fits")
    f.make_map(ra=7.4643751, dec=-44.79948193)
    test_min=f.fit(np.array([10]), "./sloeginphys/tests/config_fsps.yaml", spec_data=["./sloeginphys/tests/sn.fits"])
    os.system("rm -f "+working_dir+"*_data.txt")
    os.system("rm -f "+working_dir+"*_temp.txt")
    assert test_min[1][0]==10
    assert test_min[2]<1e-10