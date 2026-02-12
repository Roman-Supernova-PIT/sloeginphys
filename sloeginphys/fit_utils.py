import numpy as np
from astropy.wcs import WCS
from pypolyclip import clip_multi
from roman_wfss.modeling.linear.WFSSImageSimulator_NERSC import WFSSImageSimulator_NERSC

def overlap(ref_wcs, data_wcs, xmax, ymax, pixPos, buffer, spec, data, band, sca):
    """Overlaps the pixels between two different coordinate systems"""
    naxis=(xmax, ymax)
    #Find which pixels in the data correspond to pixels in the reference image
    xPos=np.transpose(pixPos)[1]
    yPos=np.transpose(pixPos)[0]
    temp_coord=ref_wcs.pixel_to_world(xPos, yPos)
    data_coord=data_wcs.world_to_pixel(temp_coord)
    #Round the coordinates and save them
    xs=[]
    ys=[]
    for j in range(0, len(data_coord[0])):
        xs.append(round(data_coord[0][j]))
        ys.append(round(data_coord[1][j]))
    #Make a square around the area of useful pixels
    left=np.min(xs)-buffer
    right=np.max(xs)+buffer
    bottom=np.min(ys)-buffer
    top=np.max(ys)+buffer
    pixel_list=[]
    new_seg_data=np.zeros((ymax, xmax))
    for x in range(left, right+1):
        for y in range(bottom, top+1):
            #Find the vertices at the edge of the pixel, keeping in mind that pypolyclip anchors at the bottom left corner of the pixel
            vert_x=[x+1, x+1, x, x]
            vert_y=[y+1, y, y, y+1]
            #Convert between the two frames
            temp_vert=data_wcs.pixel_to_world(vert_x, vert_y)
            vertices=ref_wcs.world_to_pixel(temp_vert)
            #Do the polyclipping
            px=[vertices[0]]
            py=[vertices[1]]
            xc, yc, area, slices=clip_multi(px, py, naxis)
            #Check if the pixel contains any of the non-zero pixels in the original segmentation map, and add to the pixel list and segmentation map if so
            for q in range(0, len(xc)):
                new_xc=[]
                new_yc=[]
                new_area=[]
                #Test the coordinate in the pixels against the segmentation map
                test_coord=[yc[q], xc[q]]
                for s in range(0, len(pixPos)):
                    pixPos_coord=pixPos[s]
                    if(list(test_coord)==list(pixPos_coord)):
                        #Add this pixel to the segmentation map
                        new_seg_data[y, x]=1
                        #Only add to the pixel list pixels that are included in the original segmentation map
                        new_xc.append(xc[q])
                        new_yc.append(yc[q])
                        new_area.append(area[q])
                        pixel_list.append([x, y, new_xc, new_yc, new_area])
    if(spec==True):
        #Make the simulator with the new segmentation map. Prevents us from simulating an entire empty image.
        test_sim=WFSSImageSimulator_NERSC(data, data_wcs, new_seg_data, ref_wcs, "PRISM", sca, xmax, ymax)
        return(pixel_list, test_sim, new_seg_data)
    else:
        return(pixel_list, new_seg_data)

def make_SED_bc03(working_dir, one_sed, theta, plength, pixPos, ised_dir, csp_params, recyc, file_names):
    """Make an SED using BC03"""
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
            
    else:
        for i in range(0, len(pixPos)):
            params = theta[int(i * plength) : int((i + 1) * plength)]
            #Simulate the spectra using the pixels from the segmentation map
            age_params = [params[-1]]
            sfh_params = list(params[:-1])
            spec_name = "test_" + str(i)
            csp_name = working_dir + "param_" + str(i) + ".txt"
            full_name=working_dir+str(pixPos[i][1])+"_"+str(pixPos[i][0])+".txt"
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
                
    return

def make_SED_fsps(working_dir, one_sed, theta, param_dict, plength, sp, pixPos):
    """Make an SED using FSPS"""
    if(one_sed==True):
        params = theta[int(0 * plength) : int((0 + 1) * plength)]
        for j in range(0, len(param_dict)):
            sp.params[param_dict[j]] = params[j]
            #This is kind of a cheat. tage is a parameter in sp but must also be input in making the spectrum, so here we set it with all the others, then pull it out for actually making the spectrum
            tage=sp.params["tage"]
            #Make the spectrum
            spec = sp.get_spectrum(tage=tage, peraa=True)
            spec = np.transpose(spec)
            np.savetxt(working_dir+"test.txt", spec)
    else:
        for i in range(0, len(pixPos)):
            params = theta[int(i * plength) : int((i + 1) * plength)]
            for j in range(0, len(param_dict)):
                sp.params[param_dict[j]] = params[j]
                #This is kind of a cheat. tage is a parameter in sp but must also be input in making the spectrum, so here we set it with all the others, then pull it out for actually making the spectrum
                tage=sp.params["tage"]
                # Make the spectrum
                spec = sp.get_spectrum(tage=tage, peraa=True)
                spec = np.transpose(spec)
                np.savetxt(working_dir+str(pixPos[i][0])+"_"+str(pixPos[i][1])+".txt", spec)

def translate_SED(test_pixPos, pix, one_sed, working_dir, z, cosmo, verbose=False, name=None):
    """Translate the SEDs made with BC03 or FSPS from the original coordinate system to another one"""
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
        if name==None:
            np.savetxt(working_dir+str(x)+"_"+str(y)+"_data.txt", out_data)
        else:
            np.savetxt(working_dir+str(x)+"_"+str(y)+"_"+name+".txt", out_data)