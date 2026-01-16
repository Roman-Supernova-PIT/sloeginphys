import os
import numpy as np
import glob
import scipy.interpolate as spi
from snappl.logger import SNLogger

def make_csp_file(ised_dir, library, Z, imf, dust, sfh, ssp_name, file_name, sfh_params=None, dust_params=None, recyc=False):
    '''This function makes an input file used to run csp_galaxev.
    Parameters:
        ised_dir (string) - the directory where the ised files are stored
        library (string) - the library from BC03 to use (BaSeL, miles, or stelib)
        Z (string) - metallicity from BC03 (i.e. m82)
        imf (string) - initial mass function from BC03 to use (Kroupa, Salpeter, Chabrier, or TopHeavy)
        dust (bool) - whether or not to use dust parameters
        sfh (int) - star formation history to use based on BC03 conventions. See below
        ssp_name (string) - name to give the output files from BC03
        file_name (string) - name for the file this program outputs
        sfh_params (array) - array of the parameters defining the SFH. Different for each SFH
        dust_params (array, optional) - a two-element array of mu and tau_V. Only required if dust=True
        recyc (bool, optional) - whether or not to include dust recycling. Only required for sfh=1 or sfh=-1
    Returns:
        None; creates a text file in the working directory that can be used to run csp_galaxev
    SFH Options:
        0 - SSP/instantaneous burst. No SFH parameters.
        1 - Exponential with tau. Provide e-folding time in Gyr, fraction of gas to be recycled if gas recycling is turned on, and  cutoff time in Gyr after which SFR=0.
        -1 - Exponential with mu_SFR. Provide mu_SFR, fraction of gas to be recycled if gas recycling is turned on, and  cutoff time in Gyr after which SFR=0.
        2 - Single burst of finite length. Provide burst duration tau in Gyr.
        3 - Constant. Provide SFR in solar masses/year and cutoff time after which SFR=0.
        4 - Delayed. Provide maximum SFR time and cutoff time after which SFR=0.
        5 - Linearly decreasing. Provide time in Gyr at which SFR=0 and cutoff time after which SFR=0 in Gyr.
        6 - From file. Provide the file name.
        7 - Single or double exponential from Chen et al. NOT SUPPORTED.'''
    #Create log
    log=SNLogger()
    input_string=""
    #ISED name based on the bc03 conventions
    input_string+=(ised_dir+"bc2003_lr_"+library+"_"+Z+"_"+imf+"_ssp.ised\n")
    #Dust parameters
    if (dust==False):
        input_string+=("N\n")
    elif (dust==True):
        #Prevents program from proceeding if the format of the dust parameters is incorrect
        if ((dust_params==None) or (len(dust_params)!=2)):
            log.error("Please provide tau_V and mu")
            return
        else:
            input_string+=("Y\n"+str(dust_params[0])+"\n"+str(dust_params[1])+"\n")
    else:
        log.error("Invalid value for dust")
        return
    #Don't calculate the flux-weighted age or whatever it is
    input_string+=("0\n")
    #Put in the SFH parameters. Prevents program from proceeding if sfh_params is not the right format.
    if (sfh==0):
        input_string+=("0\n")
    elif (sfh==1):
        if (sfh_params==None):
            log.error("Please provide tau, Tcut, and epsilon if gas recpoycling is true")
            return
        if (recyc==True):
            if (len(sfh_params)!=3):
                log.error("Please provide tau, epsilon, and Tcut")
                return
            else:
                input_string+=("1\n"+str(sfh_params[0])+"\nY\n"+str(sfh_params[1])+"\n"+str(sfh_params[2])+"\n")
        elif (recyc==False):
            if (len(sfh_params)!=2):
                log.error("Please provide tau and Tcut")
                return
            else:
                input_string+=("1\n"+str(sfh_params[0])+"\nN\n"+str(sfh_params[1])+"\n")
    elif (sfh==-1):
        if (sfh_params==None):
            log.error("Please provide mu_SFR, Tcut, and epsilon if gas recycling is true")
            return
        if (recyc==True):
            if (len(sfh_params)!=3):
                log.error("Please provide mu_SFR, epsilon, and Tcut")
                return
            else:
                input_string+=("-1\n"+str(sfh_params[0])+"\nY\n"+str(sfh_params[1])+"\n"+str(sfh_params[2])+"\n")
        elif (recyc==False):
            if (len(sfh_params)!=2):
                log.error("Please provide mu_SFR and Tcut")
                return
            else:
                input_string+=("-1\n"+str(sfh_params[0])+"\nN\n"+str(sfh_params[1])+"\n")
    elif (sfh==2):
        if ((sfh_params==None) or (len(sfh_params)!=1)):
            log.error("Please provide burst duration tau")
            return
        else:
            input_string+=("2\n"+str(sfh_params[0])+"\n")
    elif (sfh==3):
        if ((sfh_params==None) or (len(sfh_params)!=2)):
            log.error("Please provide SFR in solar masses/yr and Tcut")
            return
        else:
            input_string+=("3\n"+str(sfh_params[0])+"\n"+str(sfh_params[1])+"\n")
    elif (sfh==4):
        if ((sfh_params==None) or (len(sfh_params)!=2)):
            log.error("Please provide max SFR time tau and Tcut")
            return
        else:
            input_string+=("4\n"+str(sfh_params[0])+"\n"+str(sfh_params[1])+"\n")
    elif (sfh==5):
        if ((sfh_params==None) or (len(sfh_params)!=2)):
            log.error("Please provide SFR=0 time tau and Tcut")
            return
        else:
            input_string+=("5\n"+str(sfh_params[0])+"\n"+str(sfh_params[1])+"\n")
    elif (sfh==6):
        if ((sfh_params==None) or (len(sfh_params)!=1)):
            log.error("Please provide file name")
            return
        else:
            input_string+=("6\n"+str(sfh_params[0])+"\n")
    elif (sfh==7):
        log.error("Not supported")
        return
    else:
        log.error("Invalid SFH type")
        return
    input_string+=str(ssp_name)
    #Opens the file and writes the big input string into it
    f=open(file_name, "w")
    f.write(input_string)
    f.close()
    return

def run_csp(work_dir, input_name, delete_in=False, verbose=False):
    '''The function runs csp_galaxev from python
    Parameters:
        work_dir (string) - the directory where the output files should end up
        input_name (string) - the name of the file with input parameters including extension. Can be made with make_csp_file
        delete_in (bool, optional) - whether or not to delete the input file. Useful if running a large number of these
    Returns:
        None; runs csp_galaxev
    Notes:
        Requires $bc03 to be in your .bashrc or elsewhere defined. Should be done as part of setting up BC03'''
    #Get the current directory
    tempdir=os.getcwd()
    #Move into the directory where the files should be made
    os.chdir(work_dir)
    #Run csp_galaxev
    if(verbose==False):
        os.system("$bc03/csp_galaxev < "+input_name+" &> bc03_logfile.txt")
    else:
        os.system("$bc03/csp_galaxev < "+input_name)
    #Delete the input files if desired
    if delete_in==True:
        os.system("rm -f "+input_name)
    #Go back to the original directory
    os.chdir(tempdir)
    return

def make_gpl_file(file_name, sed_name, age_range, out_name, wave_range=None):
    '''This function makes the input file for running galaxevp.
    Parameters:
        file_name (string) - the name of the file to make
        sed_name (string) - name the *.ised file you want to process. Created by make_csp
        age_range (array) - range of ages to calculate SEDs for. Must be an array, even if it's only one element long
        out_name (string) - the name of the output sed files
        wave_range (array, optional) - range of wavelengths to consider. Two element array of form [start, stop]
    Returns:
        None; creates text file that can be used to run galaxevpl'''
    input_string=""
    input_string+=sed_name+"\n"
    #Age MUST be an array, even if it's only one element long. Prevents program from proceeding otherwise.
    if hasattr(age_range, "__len__")==False:
        log.error("Age MUST be an array, even if it's only one element long")
        return
    else:
        input_string+=str(age_range[0])
        if len(age_range)>1:
            for i in range(0, len(age_range)):
                input_string+=", "+str(age_range[i])
    input_string+="\n"
    #Makes wavelength range if input. Prevents program from proceeding if wavelength range is invalid.
    if (wave_range!=None):
        if(len(wave_range)!=2):
            log.error("Invalid wavelength range")
            return
        else:
            input_string+=str(wave_range[0])+", "+str(wave_range[1])+"\n"
    else:
        input_string+="\n"
    input_string+=out_name
    #Opens the file and writes the parameters into it.
    f=open(file_name, "w")
    f.write(input_string)
    f.close()
    return

def run_gpl(work_dir, input_name, delete_in=False, verbose=False):
    '''This function runs galaxevpl from python
    Parameters:
        work_dir (string) - the directory where the final files should be created
        input_name (string) - the path to the input file including extension. Can be made with make_gpl_file
        delete_in (bool, optional) - whether or not to delete the input file. Useful if running a large number of these
    Returns:
        None; runs galaxevpl'''
    #Get the current directory
    tempdir=os.getcwd()
    #Move into the working directory
    os.chdir(work_dir)
    #Run galaxevpl
    if(verbose==False):
        os.system("$bc03/galaxevpl < "+input_name+" &> bc03_logfile.txt")
    else:
        os.system("$bc03/galaxevpl < "+input_name)
    #Delete input files if desired
    if delete_in==True:
        os.system("rm -f "+input_name)
    #Move back to the original directory
    os.chdir(tempdir)
    return

def csp_grid(ised_dir, work_dir, library, imf, sfh, dust=True, Z_range=True, Z=None, dust_params=None, sfh_params=None):
    '''The function generates a grid of SEDs using the csp function from BC03.
    Parameters:
        ised_dir (string) - path to the location of the reference .ised files
        work_dir (string) - path to the location the files should be created. Recommend choosing a location other than ised_dir
        library (string) - the library from BC03 to use (BaSeL, miles, or stelib)
        Z (string) - metallicity from BC03
        imf (string) - IMF from BC03 to use (Kroupa, Salpeter, Chabrier, or TopHeavy)
        sfh (int) - SFH to use based on BC03 conventions. See make_csp_file for more details
        dust (bool) - whether or not to use dust parameters
        Z_range (bool) - whether or not to grid over metallicities
        Z (string, optional) - metallicity to use if not gridding. Required only if Z_range is false
        dust_params (array, optional) - a two-element array of three-element arrays of mu and tau_V in the form [start, stop, step]. Only required if dust=True
        sfh_params (array, optional) - array of the parameters defining the SFH history. Different for each SFH. Each parameter used in running csp is represented by a three-element array of the form [start, stop, step]
    Returns:
        None; creates csp parameter files and runs csp_galaxev for all of them, then erases the input files.
    Notes:
        See make_csp_file for more details on the dust and SFH parameters.
        All parameters are entered in as a three-element array of the form [start, stop, step]'''
    #Create grid of metallicities if Z_range is true
    if Z_range==True:
        if imf=="TopHeavy":
            Z_grid=["z0001", "z02", "z004", "z0004", "z05", "z008"]
        else:
            Z_grid=["m22", "m32", "m42", "m52", "m62", "m72", "m82"]
    elif Z==None:
        log.error("Please provide a value for metallicity Z")
        return
    #Creates grid for dust parameters
    if (dust==True):
        if ((dust_params==None) or (len(dust_params)!=2) or (len(dust_params[0])!=3) or (len(dust_params[1])!=3)):
            log.error("Please provide ranges for mu and tau_V in the form [start, stop, step]")
            return
        else:
            mu_grid=np.arange(dust_params[0][0], dust_params[0][1], dust_params[0][2])
            tau_V_grid=np.arange(dust_params[1][0], dust_params[1][1], dust_params[1][2])
    #Iterates over the dust parameters, metallicity, and other SFH parameters. Different for each SFH. Creates csp file, then runs csp
    if (sfh==0):
        if (Z_range==True):
            for i in range(0, len(Z_grid)):
                if (dust==True):
                    for j in range(0, len(mu_grid)):
                        for k in range(0, len(tau_V_grid)):
                            outname=work_dir+"Z"+str(Z_grid[i])+"_mu"+str(mu_grid[j])+"_tauV"+str(tau_V_grid[k])
                            make_csp_file(ised_dir, library, Z_grid[i], imf, True, 0, outname, outname+".txt", dust_params=[mu_grid[j], tau_V_grid[k]])
                            run_csp(work_dir, outname+".txt", delete_in=True)
                else:
                    outname=work_dir+"Z"+str(Z_grid[i])
                    make_csp_file(ised_dir, library, Z_grid[i], imf, False, 0, outname, outname+".txt")
                    run_csp(work_dir, outname+".txt", delete_in=True)
        elif (dust==True):
            for j in range(0, len(mu_grid)):
                for k in range(0, len(tau_V_grid)):
                    outname=work_dir+"mu"+str(mu_grid[j])+"_tauV"+str(tau_V_grid[k])
                    make_csp_file(ised_dir, library, Z, imf, True, 0, outname, outname+".txt", dust_params=[mu_grid[j], tau_V_grid[k]])
                    run_csp(work_dir, outname+".txt", delete_in=True)
        else:
            log.error("You have entered no gridable parameters")
            return
    elif (sfh==1):
        if (sfh_params==None):
            log.error("Please provide tau, Tcut, and epsilon if gas recycling is being used")
            return
        elif ((len(sfh_params)==2) or (len(sfh_params)==3)):
            tau_grid=np.arange(sfh_params[0][0], sfh_params[0][1], sfh_params[0][2])
            Tcut_grid=np.arange(sfh_params[1][0], sfh_params[1][1], sfh_params[1][2])
            if (len(sfh_params)==3):
                epsilon_grid=np.arange(sfh_params[2][0], sfh_params[2][1], sfh_params[2][2])
                for i in range(0, len(tau_grid)):
                    for j in range(0, len(Tcut_grid)):
                        for k in range(0, len(epsilon_grid)):
                            if (Z_range==True):
                                for l in range(0, len(Z_grid)):
                                    if (dust==True):
                                        for m in range (0, len(mu_grid)):
                                            for n in range(0, len(tau_V_grid)):
                                                outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_epsilon"+str(epsilon_grid[k])+"_Z"+str(Z_grid[l])+"_mu"+str(mu_grid[m])+"_tauV"+str(tau_V_grid[n])
                                                make_csp_file(ised_dir, library, Z_grid[l], imf, True, 1, outname, outname+".txt", sfh_params=[tau_grid[i], epsilon_grid[k], Tcut_grid[j]], dust_params=[mu_grid[m], tau_V_grid[n]], recyc=True)
                                                run_csp(work_dir, outname+".txt", delete_in=True)
                                    else:
                                        outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_epsilon"+str(epsilon_grid[k])+"_Z"+str(Z_grid[l])
                                        make_csp_file(ised_dir, library, Z_grid[l], imf, False, 1, outname, outname+".txt", sfh_params=[tau_grid[i], epsilon_grid[k], Tcut_grid[j]], recyc=True)
                                        run_csp(work_dir, outname+".txt", delete_in=True)
                            elif(dust==True):
                                for l in range(0, len(mu_grid)):
                                    for m in range(0, len(tau_V_grid)):
                                        outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_epsilon"+"_mu"+str(mu_grid[l])+"_tauV"+str(tau_V_grid[m])
                                        make_csp_file(ised_dir, library, Z, imf, True, 1, outname, outname+".txt", sfh_params=[tau_grid[i], epsilon_grid[k], Tcut_grid[j]], dust_params=[mu_grid[l], tau_V_grid[m]], recyc=True)
                                        run_csp(work_dir, outname+".txt", delete_in=True)
                            else:
                                outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_epsilon"+str(epsilon_grid[k])
                                make_csp_file(ised_dir, library, Z, imf, False, 1, outname, outname+".txt", sfh_params=[tau_grid[i], epsilon_grid[k], Tcut_grid[j]], recyc=True)
                                run_csp(work_dir, outname+".txt", delete_in=True)
            elif(len(sfh_params)==2):
                for i in range(0, len(tau_grid)):
                    for j in range(0, len(Tcut_grid)):
                        if (Z_range==True):
                            for k in range(0, len(Z_grid)):
                                if (dust==True):
                                    for l in range(0, len(mu_grid)):
                                        for m in range(0, len(tau_V_grid)):
                                            outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_Z"+str(Z_grid[k])+"_mu"+str(mu_grid[l])+"_tauV"+str(tau_V_grid[m])
                                            make_csp_file(ised_dir, library, Z_grid[k], imf, True, 1, outname, outname+".txt", sfh_params=[tau_grid[i], Tcut_grid[j]], dust_params=[mu_grid[l], tau_V_grid[m]])
                                            run_csp(work_dir, outname+".txt", delete_in=True)
                                else:
                                    outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_Z"+str(Z_grid[k])
                                    make_csp_file(ised_dir, library, Z_grid[k], imf, False, 1, outname, outname+".txt", sfh_params=[tau_grid[i], Tcut_grid[j]])
                                    run_csp(work_dir, outname+".txt", delete_in=True)
                        elif (dust==True):
                            for k in range(0, len(mu_grid)):
                                for l in range(0, len(tau_V_grid)):
                                    outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_mu"+str(mu_grid[k])+"_tauV"+str(tau_V_grid[l])
                                    make_csp_file(ised_dir, library, Z, imf, True, 1, outname, outname+".txt", sfh_params=[tau_grid[i], Tcut_grid[j]], dust_params=[mu_grid[k], tau_V_grid[l]])
                                    run_csp(work_dir, outname+".txt", delete_in=True)
                        else:
                            outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])
                            make_csp_file(ised_dir, library, Z, imf, False, 1, outname, outname+".txt", sfh_params=[tau_grid[i], Tcut_grid[j]])
                            run_csp(work_dir, outname+".txt", delete_in=True)
        else:
            log.error("Invalid parameter space")
            return
    elif (sfh==-1):
        if (sfh_params==None):
            log.error("Please provide tau, Tcut, and epsilon if gas recycling is being used")
            return
        elif ((len(sfh_params)==2) or (len(sfh_params)==3)):
            muSFR_grid=np.arange(sfh_params[0][0], sfh_params[0][1], sfh_params[0][2])
            Tcut_grid=np.arange(sfh_params[1][0], sfh_params[1][1], sfh_params[1][2])
            if (len(sfh_params)==3):
                epsilon_grid=np.arange(sfh_params[2][0], sfh_params[2][1], sfh_params[2][2])
                for i in range(0, len(muSFR_grid)):
                    for j in range(0, len(Tcut_grid)):
                        for k in range(0, len(epsilon_grid)):
                            if (Z_range==True):
                                for l in range(0, len(Z_grid)):
                                    if (dust==True):
                                        for m in range (0, len(mu_grid)):
                                            for n in range(0, len(tau_V_grid)):
                                                outname=work_dir+"muSFR"+str(muSFR_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_epsilon"+str(epsilon_grid[k])+"_Z"+str(Z_grid[l])+"_mu"+str(mu_grid[m])+"_tauV"+str(tau_V_grid[n])
                                                make_csp_file(ised_dir, library, Z_grid[l], imf, True, -1, outname, outname+".txt", sfh_params=[muSFR_grid[i], epsilon_grid[k], Tcut_grid[j]], dust_params=[mu_grid[m], tau_V_grid[n]], recyc=True)
                                                run_csp(work_dir, outname+".txt", delete_in=True)
                                    else:
                                        outname=work_dir+"muSFR"+str(muSFR_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_epsilon"+str(epsilon_grid[k])+"_Z"+str(Z_grid[l])
                                        make_csp_file(ised_dir, library, Z_grid[l], imf, False, -1, outname, outname+".txt", sfh_params=[muSFR_grid[i], epsilon_grid[k], Tcut_grid[j]], recyc=True)
                                        run_csp(work_dir, outname+".txt", delete_in=True)
                            elif(dust==True):
                                for l in range(0, len(mu_grid)):
                                    for m in range(0, len(tau_V_grid)):
                                        outname=work_dir+"muSFR"+str(muSFR_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_epsilon"+"_mu"+str(mu_grid[l])+"_tauV"+str(tau_V_grid[m])
                                        make_csp_file(ised_dir, library, Z, imf, True, -1, outname, outname+".txt", sfh_params=[muSFR_grid[i], epsilon_grid[k], Tcut_grid[j]], dust_params=[mu_grid[l], tau_V_grid[m]], recyc=True)
                                        run_csp(work_dir, outname+".txt", delete_in=True)
                            else:
                                outname=work_dir+"muSFR"+str(muSFR_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_epsilon"+str(epsilon_grid[k])
                                make_csp_file(ised_dir, library, Z, imf, False, -1, outname, outname+".txt", sfh_params=[muSFR_grid[i], epsilon_grid[k], Tcut_grid[j]], recyc=True)
                                run_csp(work_dir, outname+".txt", delete_in=True)
            elif(len(sfh_params)==2):
                for i in range(0, len(muSFR_grid)):
                    for j in range(0, len(Tcut_grid)):
                        if (Z_range==True):
                            for k in range(0, len(Z_grid)):
                                if (dust==True):
                                    for l in range(0, len(mu_grid)):
                                        for m in range(0, len(tau_V_grid)):
                                            outname=work_dir+"muSFR"+str(muSFR_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_Z"+str(Z_grid[k])+"_mu"+str(mu_grid[l])+"_tauV"+str(tau_V_grid[m])
                                            make_csp_file(ised_dir, library, Z_grid[k], imf, True, -1, outname, outname+".txt", sfh_params=[muSFR_grid[i], Tcut_grid[j]], dust_params=[mu_grid[l], tau_V_grid[m]])
                                            run_csp(work_dir, outname+".txt", delete_in=True)
                                else:
                                    outname=work_dir+"muSFR"+str(muSFR_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_Z"+str(Z_grid[k])
                                    make_csp_file(ised_dir, library, Z_grid[k], imf, False, -1, outname, outname+".txt", sfh_params=[muSFR_grid[i], Tcut_grid[j]])
                                    run_csp(work_dir, outname+".txt", delete_in=True)
                        elif (dust==True):
                            for k in range(0, len(mu_grid)):
                                for l in range(0, len(tau_V_grid)):
                                    outname=work_dir+"muSFR"+str(muSFR_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_mu"+str(mu_grid[k])+"_tauV"+str(tau_V_grid[l])
                                    make_csp_file(ised_dir, library, Z, imf, True, -1, outname, outname+".txt", sfh_params=[muSFR_grid[i], Tcut_grid[j]], dust_params=[mu_grid[k], tau_V_grid[l]])
                                    run_csp(work_dir, outname+".txt", delete_in=True)
                        else:
                            outname=work_dir+"muSFR"+str(muSFR_grid[i])+"_Tcut"+str(Tcut_grid[j])
                            make_csp_file(ised_dir, library, Z, imf, False, -1, outname, outname+".txt", sfh_params=[muSFR_grid[i], Tcut_grid[j]])
                            run_csp(work_dir, outname+".txt", delete_in=True)
        else:
            log.error("Invalid parameter space")
            return
    elif (sfh==2):
        if((sfh_params==None) or (len(sfh_params)!=1) or len(sfh_params[0])!=3):
            log.error("Please provide start, stop, and step for burst duration tau")
            return
        else:
            tau_grid=np.arange(sfh_params[0][0], sfh_params[0][1], sfh_params[0][2])
            for i in range(0, len(tau_grid)):
                if (Z_range==True):
                    for j in range(0, len(Z_grid)):
                        if (dust==True):
                            for k in range(0, len(mu_grid)):
                                for l in range(0, len(tau_V_grid)):
                                    outname=work_dir+"tau"+str(tau_grid[i])+"_Z"+str(Z_grid[j])+"_mu"+str(mu_grid[k])+"_tauV"+str(tau_V_grid[l])
                                    make_csp_file(ised_dir, library, Z_grid[j], imf, True, 2, outname, outname+".txt", sfh_params=[tau_grid[i]], dust_params=[mu_grid[k], tau_V_grid[l]])
                                    run_csp(work_dir, outname+".txt", delete_in=True)
                        else:
                            outname=work_dir+"tau"+str(tau_grid[i])+"_Z"+str(Z_grid[j])
                            make_csp_file(ised_dir, library, Z_grid[j], imf, False, 2, outname, outname+".txt", sfh_params=[tau_grid[i]])
                            run_csp(work_dir, outname+".txt", delete_in=True)
                elif (dust==True):
                    for j in range(0, len(mu_grid)):
                        for k in range(0, len(tau_V_grid)):
                            outname=work_dir+"tau"+str(tau_grid[i])+"_mu"+str(mu_grid[j])+"_tauV"+str(tau_V_grid[k])
                            make_csp_file(ised_dir, library, Z, imf, True, 2, outname, outname+".txt", sfh_params=[tau_grid[i]], dust_params=[mu_grid[j], tau_V_grid[k]])
                            run_csp(work_dir, outname+".txt", delete_in=True)
                else:
                    outname=work_dir+"tau"+str(tau_grid[i])
                    make_csp_file(ised_dir, library, Z, imf, False, 2, outname, outname+".txt", sfh_params=[tau_grid[i]])
                    run_csp(work_dir, outname+".txt", delete_in=True)
    elif (sfh==3):
        if((sfh_params==None) or (len(sfh_params)!=2) or (len(sfh_params[0])!=3) or (len(sfh_params[1])!=3)):
            log.error("Please provide start, stop, and step for SFR and Tcut")
            return
        sfr_grid=np.arange(sfh_params[0][0], sfh_params[0][1], sfh_params[0][2])
        Tcut_grid=np.arange(sfh_params[1][0], sfh_params[1][1], sfh_params[1][2])
        for i in range(0, len(sfr_grid)):
            for j in range(0, len(Tcut_grid)):
                if (Z_range==True):
                    for k in range(0, len(Z_grid)):
                        if (dust==True):
                            for l in range(0, len(mu_grid)):
                                for m in range(0, len(tau_V_grid)):
                                    outname=work_dir+"SFR"+str(sfr_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_Z"+str(Z_grid[k])+"_mu"+str(mu_grid[l])+"_tauV"+str(tau_V_grid[m])
                                    make_csp_file(ised_dir, library, Z_grid[k], imf, True, 3, outname, outname+".txt", sfh_params=[sfr_grid[i], Tcut_grid[j]], dust_params=[mu_grid[l], tau_V_grid[m]])
                                    run_csp(work_dir, outname+".txt", delete_in=True)
                        else:
                            outname=work_dir+"SFR"+str(sfr_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_Z"+str(Z_grid[k])
                            make_csp_file(ised_dir, library, Z_grid[k], imf, False, 3, outname, outname+".txt", sfh_params=[sfr_grid[i], Tcut_grid[j]])
                            run_csp(work_dir, outname+".txt", delete_in=True)
                elif (dust==True):
                    for k in range(0, len(mu_grid)):
                        for l in range(0, len(tau_V_grid)):
                            outname=work_dir+"SFR"+str(sfr_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_mu"+str(mu_grid[k])+"_tauV"+str(tau_V_grid[l])
                            make_csp_file(ised_dir, library, Z, imf, True, 3, outname, outname+".txt", sfh_params=[sfr_grid[i], Tcut_grid[j]], dust_params=[mu_grid[k], tau_V_grid[l]])
                            run_csp(work_dir, outname+".txt", delete_in=True)
                else:
                    outname=work_dir+"SFR"+str(sfr_grid[i])+"_Tcut"+str(Tcut_grid[j])
                    make_csp_file(ised_dir, library, Z, imf, False, 3, outname, outname+".txt", sfh_params=[sfr_grid[i], Tcut_grid[j]])
                    run_csp(work_dir, outname+".txt", delete_in=True)
    elif (sfh==4):
        if((sfh_params==None) or (len(sfh_params)!=2) or (len(sfh_params[0])!=3) or (len(sfh_params[1])!=3)):
            log.error("Please provide start, stop, and step for tau and Tcut")
            return
        tau_grid=np.arange(sfh_params[0][0], sfh_params[0][1], sfh_params[0][2])
        Tcut_grid=np.arange(sfh_params[1][0], sfh_params[1][1], sfh_params[1][2])
        for i in range(0, len(tau_grid)):
            for j in range(0, len(Tcut_grid)):
                if (Z_range==True):
                    for k in range(0, len(Z_grid)):
                        if (dust==True):
                            for l in range(0, len(mu_grid)):
                                for m in range(0, len(tau_V_grid)):
                                    outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_Z"+str(Z_grid[k])+"_mu"+str(mu_grid[l])+"_tauV"+str(tau_V_grid[m])
                                    make_csp_file(ised_dir, library, Z_grid[k], imf, True, 4, outname, outname+".txt", sfh_params=[tau_grid[i], Tcut_grid[j]], dust_params=[mu_grid[l], tau_V_grid[m]])
                                    run_csp(work_dir, outname+".txt", delete_in=True)
                        else:
                            outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_Z"+str(Z_grid[k])
                            make_csp_file(ised_dir, library, Z_grid[k], imf, False, 4, outname, outname+".txt", sfh_params=[tau_grid[i], Tcut_grid[j]])
                            run_csp(work_dir, outname+".txt", delete_in=True)
                elif (dust==True):
                    for k in range(0, len(mu_grid)):
                        for l in range(0, len(tau_V_grid)):
                            outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_mu"+str(mu_grid[k])+"_tauV"+str(tau_V_grid[l])
                            make_csp_file(ised_dir, library, Z, imf, True, 4, outname, outname+".txt", sfh_params=[tau_grid[i], Tcut_grid[j]], dust_params=[mu_grid[k], tau_V_grid[l]])
                            run_csp(work_dir, outname+".txt", delete_in=True)
                else:
                    outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])
                    make_csp_file(ised_dir, library, Z, imf, False, 4, outname, outname+".txt", sfh_params=[tau_grid[i], Tcut_grid[j]])
                    run_csp(work_dir, outname+".txt", delete_in=True)
    elif (sfh==5):
        if((sfh_params==None) or (len(sfh_params)!=2) or (len(sfh_params[0])!=3) or (len(sfh_params[1])!=3)):
            log.error("Please provide start, stop, and step for tau and Tcut")
            return
        tau_grid=np.arange(sfh_params[0][0], sfh_params[0][1], sfh_params[0][2])
        Tcut_grid=np.arange(sfh_params[1][0], sfh_params[1][1], sfh_params[1][2])
        for i in range(0, len(tau_grid)):
            for j in range(0, len(Tcut_grid)):
                if (Z_range==True):
                    for k in range(0, len(Z_grid)):
                        if (dust==True):
                            for l in range(0, len(mu_grid)):
                                for m in range(0, len(tau_V_grid)):
                                    outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_Z"+str(Z_grid[k])+"_mu"+str(mu_grid[l])+"_tauV"+str(tau_V_grid[m])
                                    make_csp_file(ised_dir, library, Z_grid[k], imf, True, 5, outname, outname+".txt", sfh_params=[tau_grid[i], Tcut_grid[j]], dust_params=[mu_grid[l], tau_V_grid[m]])
                                    run_csp(work_dir, outname+".txt", delete_in=True)
                        else:
                            outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_Z"+str(Z_grid[k])
                            make_csp_file(ised_dir, library, Z_grid[k], imf, False, 5, outname, outname+".txt", sfh_params=[tau_grid[i], Tcut_grid[j]])
                            run_csp(work_dir, outname+".txt", delete_in=True)
                elif (dust==True):
                    for k in range(0, len(mu_grid)):
                        for l in range(0, len(tau_V_grid)):
                            outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])+"_mu"+str(mu_grid[k])+"_tauV"+str(tau_V_grid[l])
                            make_csp_file(ised_dir, library, Z, imf, True, 5, outname, outname+".txt", sfh_params=[tau_grid[i], Tcut_grid[j]], dust_params=[mu_grid[k], tau_V_grid[l]])
                            run_csp(work_dir, outname+".txt", delete_in=True)
                else:
                    outname=work_dir+"tau"+str(tau_grid[i])+"_Tcut"+str(Tcut_grid[j])
                    make_csp_file(ised_dir, library, Z, imf, False, 5, outname, outname+".txt", sfh_params=[tau_grid[i], Tcut_grid[j]])
                    run_csp(work_dir, outname+".txt", delete_in=True)
    elif (sfh==6):
        if((sfh_params==None) or (len(sfh_params)!=1) or (len(sfh_params[0])!=1)):
            log.error("Please provide file name. You cannot iterate over multiple files")
            return
        else:
            if (Z_range==True):
                for i in range(0, len(Z_grid)):
                    if (dust==True):
                        for j in range(0, len(mu_grid)):
                            for k in range(0, len(tau_V_grid)):
                                outname=work_dir+"Z"+str(Z_grid[i])+"_mu"+str(mu_grid[j])+"_tauV"+str(tau_V_grid[k])
                                make_csp_file(ised_dir, library, Z_grid[i], imf, True, 6, outname, outname+".txt", sfh_params=sfh_params, dust_params=[mu_grid[j], tau_V_grid[k]])
                                run_csp(work_dir, outname+".txt", delete_in=True)
                    else:
                        outname=work_dir+"Z"+str(Z_grid[i])
                        make_csp_file(ised_dir, library, Z_grid[i], imf, False, 6, outname, outname+".txt", sfh_params=sfh_params)
                        run_csp(work_dir, outname+".txt", delete_in=True)
            elif (dust==True):
                for j in range(0, len(mu_grid)):
                    for k in range(0, len(tau_V_grid)):
                        outname=work_dir+"mu"+str(mu_grid[j])+"_tauV"+str(tau_V_grid[k])
                        make_csp_file(ised_dir, library, Z, imf, True, 6, outname, outname+".txt", sfh_params=sfh_params, dust_params=[mu_grid[j], tau_V_grid[k]])
                        run_csp(work_dir, outname+".txt", delete_in=True)
            else:
                log.error("You have entered no gridable parameters")
                return
    elif (sfh==7):
        log.error("Not supported")
        return
    else:
        log.error("Invalid value for SFH")
        return
    return

def age_grid(age_range, work_dir, file_names=None, rename=True, full_name=None, verbose=False):
    '''This function takes a directory of .ised files or a list of files from running csp and converts them into useful SEDs over a range of ages.
    Parameters:
        age_range (array) - the ages to consider. Must be an array
        work_dir (string) - the directory where the output from csp_galaxev is stored
        file_name (string, optional) - the file to be passed into the function. Use if only specific files should be processed
        rename (bool, optional) - whether to add age to the name of the file. Defaults to true
        full_name (string, optional) - file name if the file should be completely renamed
    Returns:
        Creates galaxev_pl files that are machine-readable text files with SEDs. Makes the larger un-normalized ones as well as those normalized by mass'''
    #Gathers all the ised files in the working directory if needed
    if file_names!=None:
        files=file_names
    else:
        files=glob.glob(work_dir+"/*.ised")
    #Prevents program from proceeding if age isn't an array/list of some kind.
    if hasattr(age_range, "__len__")==False:
        log.error("Age MUST be an array, even if it's only one element long")
        return
    #Iterates over the files
    for i in range(0, len(files)):
        if file_names!=None:
            name=file_names[i].split(".")[0]
        else:
            name=files[i].split("/")[-1][:-5]
        make_gpl_file(work_dir+name+".txt", files[i], age_range, work_dir+name+".sed")
        run_gpl(work_dir, work_dir+name+".txt", delete_in=True, verbose=verbose)
        #Gets the mass file for mass normalization
        massfile=np.loadtxt(work_dir+name+".4color")
        #Eliminates the occasional duplicate rows BC03 produces. There is probably a better way to do this but I don't know it
        dup=np.inf
        flipflop=np.transpose(massfile)
        for i in range(0, len(flipflop[0])):
            test=flipflop[0][i]
            if((list(flipflop[0]).count(test))>1):
                dup=i
        if (np.isfinite(dup)):
            newmass=np.delete(massfile, dup, axis=0)
        else:
            newmass=massfile
        #Make spline to allow for arbitrary ages
        massfit=spi.make_splrep(newmass[:, 0], newmass[:, 10])
        #Loads in the gpl file we just made
        data=np.loadtxt(work_dir+name+".sed")
        lamb=data[:, 0]
        #Renormalizes the flux using the mass for each age and writes the output to an sed file
        for j in range(0, len(age_range)):
            log_age=np.log10(age_range[j])
            mass=spi.splev(log_age, massfit)
            flam=data[:, j+1]/mass
            if(full_name!=None):
                np.savetxt(full_name, np.transpose([lamb, flam]))
            elif (rename==True):
                np.savetxt(work_dir+name+"_age"+str(age_range[j])+"_norm.sed", np.transpose([lamb, flam]))
            else:
                np.savetxt(work_dir+name+"_norm.sed", np.transpose([lamb, flam]))
    return

def make_spec(working_dir, ised_dir, csp_params, spec_name, csp_name, sfh_params, age_params, dust_params=None, recyc=False, delete_in=False, full_name=None, verbose=False):
    '''This function makes a single spectrum buy combining functions make_csp_file, run_csp, and age_grid into one function
    Parameters:
        working_dir (string) - the working directory where the spectrum file should be made
        ised_dir (string) - the directory containing the unzipped ised files
        csp_params (array) -  the library name, metallicity, IMF choice, dust parameters, and SFH choice to be passed to make_csp_file. See make_csp_file documentation for more details
        spec_name (string) - the name of the final output spectrum
        csp_name (string) - the name of the parameter file that will be passed to run_csp
        sfh_params (array) - the parameters for the SFH selected above. See make_csp_file for more details.
        age_params (array) - the age or range of ages to use when making the final spectrum. Must be an array. See age_grid for more details
        dust_params (array, optional) - the parameters for dust. See make_csp_file for more details
        recyc (array, optional) - choice to use gas recycling. Only required if SFH is 1 or -1
        delete_in (bool, optional) - choice to delete the csp input file. Default choice is to keep the file
    Returns:
        Creates a normalized spectrum at a specific age for the given parameters.'''
    lib, metal, imf, dust, sfh=csp_params
    make_csp_file(ised_dir, lib, metal, imf, dust, sfh, spec_name, csp_name, sfh_params=sfh_params, dust_params=dust_params, recyc=recyc)
    run_csp(working_dir, csp_name, delete_in=delete_in, verbose=verbose)
    age_grid(age_params, working_dir, file_names=[spec_name], rename=False, full_name=full_name, verbose=verbose)
    return
