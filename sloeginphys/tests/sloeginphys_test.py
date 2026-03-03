import numpy as np
import sloeginphys
from sloeginphys.fit import fitter
from astropy.io import fits
from astropy.wcs import WCS
import os
from astropy.cosmology import FlatLambdaCDM
import pytest

def test_overlap():
    ref_wcs=WCS(fits.open("./sloeginphys/tests/seg.fits")[0].header)
    data_wcs=WCS(fits.open("./sloeginphys/tests/sn.fits")[1].header)
    xmax=4088
    ymax=4088
    pixPos=np.array([[2044, 2044]])
    buffer=1
    spec=False
    data=np.zeros((ymax, xmax))
    band="F158"
    sca=1
    working_dir="./sloeginphys/tests/"
    NERSC=True
    p, m=sloeginphys.fit_utils._overlap(ref_wcs, data_wcs, xmax, ymax, pixPos, buffer, spec, data, band, sca, working_dir, NERSC)
    assert p==[[2044, 2044, [2044], [2044], [1.0]]]

def test_simcode():
    try:
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
    except:
        working_dir="./sloeginphys/tests/"
        one_sed=True
        theta=[10]
        plength=1
        pixPos=[2044, 2044]
        ised_dir="./sloeginphys/test"
        csp_params=["BaSeL", "m82", "kroup", False, 0]
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

def test_translate():
    test_pixPos=[[2044, 2045]]
    pix=[[2045, 2044, [2044], [2044], [1.0]]]
    one_sed=True
    working_dir="./sloeginphys/tests/"
    z=0.5
    cosmo=FlatLambdaCDM(H0=70, Om0=0.3)
    sloeginphys.fit_utils._translate_SED(test_pixPos, pix, one_sed, working_dir, z, cosmo)
    f=np.loadtxt("./sloeginphys/tests/2045_2044_data.txt")
    assert f[0, 1]==pytest.approx(3.9844e-34)
    assert f[1, 1]==pytest.approx(7.9687e-34)
    assert f[2, 1]==pytest.approx(1.1953e-33)
    assert f[3, 1]==pytest.approx(7.9687e-34)
    assert f[4, 1]==pytest.approx(3.9843e-34)

def test_init():
    f=sloeginphys.fit.fitter("./sloeginphys/tests/data.fits", "./sloeginphys/tests/sn.fits", local=True, segmap="./sloeginphys/tests/seg.fits")
    assert f.pixPos[0][0]==2040
    assert f.pixPos[0][1]==3570
    assert f.pixPos[1][0]==3570
    assert f.pixPos[1][1]==2040

def test_pickob():
    f=sloeginphys.fit.fitter("./sloeginphys/tests/data.fits", "./sloeginphys/tests/sn.fits", local=True, segmap="./sloeginphys/tests/seg.fits")
    f.pick_object(1)
    assert np.where(f.seg_map_data!=0)==(np.array([2040]), np.array([3570]))
    f.pick_object(2)
    assert np.where(f.seg_map_data!=0)==(np.array([3570]), np.array([2040]))

def test_xy():
    f=sloeginphys.fit.fitter("./sloeginphys/tests/data.fits", "./sloeginphys/tests/sn.fits", local=True, segmap="./sloeginphys/tests/seg.fits")
    num=f.get_ID_xy(3570, 2040)
    assert num==1
    num2=f.get_ID_xy(2040, 3570)
    assert num2==2

def test_ad():
    f=sloeginphys.fit.fitter("./sloeginphys/tests/data.fits", "./sloeginphys/tests/sn.fits", local=True, segmap="./sloeginphys/tests/seg.fits")
    num=f.get_ID_ad(7.4643751, -44.79948193)
    assert num==1
    num2=f.get_ID_ad(7.49341965, -44.8613257)
    assert num2==2

def test_makemap():
    f=sloeginphys.fit.fitter("./sloeginphys/tests/data.fits", "./sloeginphys/tests/sn.fits", local=True, segmap="./sloeginphys/tests/seg.fits")
    f.make_map(ra=7.4643751, dec=-44.79948193)
    assert np.where(f.seg_map_data!=0)==(np.array([2040]), np.array([3570]))
    f.make_map(ra=7.49341965, dec=-44.8613257)
    assert np.where(f.seg_map_data!=0)==(np.array([3570]), np.array([2040]))
    
def test_check():
    f=sloeginphys.fit.fitter("./sloeginphys/tests/data.fits", "./sloeginphys/tests/sn.fits", local=True, segmap="./sloeginphys/tests/seg.fits")
    b=f.check_config("./sloeginphys/tests/config_fsps.yaml")
    assert b==True

def test_fit():
    try: 
        import fsps
        f=sloeginphys.fit.fitter("./sloeginphys/tests/data.fits", "./sloeginphys/tests/sn.fits", local=True, segmap="./sloeginphys/tests/seg.fits")
        f.make_map(ra=7.4643751, dec=-44.79948193)
        test_min=f.fit(np.array([10]), "./sloeginphys/tests/config_fsps.yaml", spec_data=["./sloeginphys/tests/sn.fits"])
        os.system("rm -f "+working_dir+"*_data.txt")
        os.system("rm -f "+working_dir+"*_temp.txt")
        assert test_min[1][0]==10
        assert test_min[2]<1e-10
    except:
        f=sloeginphys.fit.fitter("./sloeginphys/tests/data.fits", "./sloeginphys/tests/sn_bc03.fits", local=True, segmap="./sloeginphys/tests/seg.fits")
        f.make_map(ra=7.4643751, dec=-44.79948193)
        test_min=f.fit(np.array([10]), "./sloeginphys/tests/config_bc03.yaml", spec_data=["./sloeginphys/tests/sn_bc03.fits"])
        os.system("rm -f "+working_dir+"*_data.txt")
        os.system("rm -f "+working_dir+"*_temp.txt")
        assert test_min[1][0]==10
        assert test_min[2]<1e-10
