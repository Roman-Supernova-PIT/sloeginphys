def overlap(ref_wcs, data_wcs, xmax, ymax, pixPos, buffer, spec, data):
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
        test_sim=WFSSImageSimulator_nohdr(data, new_seg_data, data_wcs, "SNPrism")
        return(pixel_list, test_sim)
    else:
        return(pixel_list, new_seg_data)

def sim_sed():
    