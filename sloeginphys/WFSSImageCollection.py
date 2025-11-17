import sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import concurrent.futures
import multiprocessing as mp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr, norm
import pandas as pd
from .WFSSImage import WFSSImage
from .Source import Source
from .myUtils import *
from .indices import decimate
from .band import Band

class WFSSBase_nohdr:

    def __init__(self, segmap, segmap_wcs, subsample=1):
        
        self.imgs = []

        self.segMapData = np.array(segmap, dtype=int)

        self.segMapWCS = segmap_wcs
        
        self.subsample = subsample

        self.nObjs = int(np.max(self.segMapData))
        
        pixPos = getObjsPos(self.segMapData, self.nObjs)

        subPos = subsamplePos(pixPos, self.nObjs, subsample)

        subPosCen = subsamplePosCen(pixPos, self.nObjs, subsample)

        pixPos = [np.vstack([p[1], p[0]]).T for p in pixPos]

        self.sourceColl = [Source(sID+1, pix, subpix) for sID, pix, subpix in zip(range(len(pixPos)), pixPos, subPos)]

        for i in range(self.nObjs):
            pos = pixPos[i]
            wcsPos = self.segMapWCS.pixel_to_world_values(pos[:,0]-1, pos[:,1]-1)
            self.sourceColl[i].addWCSPos(np.vstack( (wcsPos[0], wcsPos[1]) ).T)

            pos = subPos[i]
            wcsPos = self.segMapWCS.pixel_to_world_values(pos[:,:,:,0].flatten()-1, pos[:,:,:,1].flatten()-1)

            self.sourceColl[i].addWCSSubPos(np.vstack( (wcsPos[0], wcsPos[1]) ).T.reshape(pixPos[i].shape[0], subsample**2, 4, 2))

            pos = subPosCen[i]
            wcsPosCen = self.segMapWCS.pixel_to_world_values(pos[:,:,0].flatten()-1, pos[:,:,1].flatten()-1)
            self.sourceColl[i].addWCSSubPosCen(np.vstack( (wcsPosCen[0], wcsPosCen[1]) ).T.reshape(pixPos[i].shape[0], subsample**2, 2))


        return

    def pixAreaOverlap_polyclip(self):

        posList = np.array([p for source in self.sourceColl for p in source.WCSSubPos.reshape(source.WCSSubPos.shape[0]*self.subsample**2*4,2)], dtype=np.float64)
        objPosN = np.array([len(source.WCSSubPos)*self.subsample**2*4 for source in self.sourceColl], dtype=int)

        self.objPixA = []

        for img in self.imgs:
            self.objPixA.append(img.getPixArea_polyclip(posList, objPosN))


        return
        

class WFSSImageSimulator_nohdr(WFSSBase_nohdr):
    
    def __init__(self, directImage, segMap, segMapWCS, filt, subsample=1, disptype='prism'):
        
        WFSSBase_nohdr.__init__(self, segMap, segMapWCS, subsample)

        self.directImageData = directImage
        self.method = "sar_bypixel"
        self.filt = filt

        for source in self.sourceColl:

            if(self.method == 'sar'):
                source.addSED(filt, "/Users/Ann/Desktop/NGRoman/testsed.txt")
                source.compute_weights(self.directImageData)
            elif(self.method == 'sar_bypixel'):
                source.addSED(filt, "/global/u1/a/aisaacs/BC03/BaSeL3.1_Atlas/Kroupa_IMF/truth/tau2_age1_norm.sed")
                source.compute_weights(self.directImageData)

            else:
                source.addPixSED(filt, len(self.imgs[0].spectralOrder.wavavg))

            self.numpix=source.segMapSubPos.shape[0]

        return


    def simulate(self, imgN, return_sim=False, save_fits=True, redo_objpix=False):

        if(not hasattr(self, 'objPixA')):
            if((self.method == 'sar') or (self.method == 'sar_bypixel')):
                self.pixAreaOverlap_polyclip()
            else:
                self.pixAreaOverlap()
        elif(redo_objpix==True):
            if (self.method == "sar") or (self.method == "sar_bypixel"):
                self.pixAreaOverlap_polyclip()
            else:
                self.pixAreaOverlap()

        sOrd = self.imgs[imgN].spectralOrder

        pix = np.zeros((4088, 4088), dtype=np.float64)

        # pix += np.random.normal(loc=0, scale=2, size=(4088,4088))

        prevposl = [0]
        for i in range(1, len(self.sourceColl)):
            prevposl.append(prevposl[i-1] + self.sourceColl[i-1].segMapPixPos.shape[0] * self.subsample**2)


        sens = sOrd.sensitivity

        wav = sOrd.wavelengths
        nwav = len(wav)
        dwav = sOrd.wavstep * 1e4


        for objID in range(len(self.sourceColl)):

            #print("Obj "+str(objID+1)+" / "+str(len(self.sourceColl))+"...", end=" ")

            npos = self.sourceColl[objID].segMapPixPos.shape[0] * self.subsample**2
            prevpos = prevposl[objID]

            xg, yg, val, indices, wavl, nn = get_val(self.objPixA[imgN], prevpos, npos, nwav)

            # need to apply a few things:
            # 1) sensitivity curve    (sens)
            # 2) flat field           (flat)
            # 3) relative pixel areas (area)
            # 4) source spectra       (flam)

            flam = np.array([sed(wav*10**4, fnu=False) for sed in self.sourceColl[objID].seds])

            dldr = sOrd.dispersion(np.array([xg, yg]).T, np.array([wav[i] for i in wavl], dtype=np.float64))

            val = multiply_val(val, flam, sens, wavl, wav, dwav, dldr, nn)

            # add flat field. still don't have it
            #val *= flat

            # sum over wavelengths
            vv, yy, xx = decimate(val, yg, xg, dims=(4088, 4088))


            # apply pixel areas -- we still don't have this information I think?
            # vv *= detdata.relative_pixelarea(xx, yy)

            # now sum into the image
            pix[yy-1, xx-1] += vv #+ np.random.normal(loc=0, scale=2*np.sqrt(abs(vv)))


            #print("done")

        if (return_sim==True):
            return pix
        return

        


class WFSSBase:

    def __init__(self, segMapFile, subsample=1):

        self.segMapFile = segMapFile
        self.imgs = []

        try:

            with fits.open(self.segMapFile) as hdul:
                self.segMapHeader = hdul[0].header
                self.segMapData = np.array(hdul[0].data, dtype=int)
                # testing
                # self.segMapData = makemap(4088, 4088)
                ###

        except Exception as e:

            print("Error: ", e)
            sys.exit(-1)

        self.segMapWCS = WCS(self.segMapHeader)

        self.subsample = subsample

        self.nObjs = int(np.max(self.segMapData))

        pixPos = getObjsPos(self.segMapData, self.nObjs)

        subPos = subsamplePos(pixPos, self.nObjs, subsample)

        subPosCen = subsamplePosCen(pixPos, self.nObjs, subsample)

        pixPos = [np.vstack([p[1], p[0]]).T for p in pixPos]

        self.sourceColl = [Source(sID+1, pix, subpix) for sID, pix, subpix in zip(range(len(pixPos)), pixPos, subPos)]

        for i in range(self.nObjs):
            pos = pixPos[i]
            wcsPos = self.segMapWCS.pixel_to_world_values(pos[:,0]-1, pos[:,1]-1)
            self.sourceColl[i].addWCSPos(np.vstack( (wcsPos[0], wcsPos[1]) ).T)

            pos = subPos[i]
            wcsPos = self.segMapWCS.pixel_to_world_values(pos[:,:,:,0].flatten()-1, pos[:,:,:,1].flatten()-1)

            self.sourceColl[i].addWCSSubPos(np.vstack( (wcsPos[0], wcsPos[1]) ).T.reshape(pixPos[i].shape[0], subsample**2, 4, 2))

            pos = subPosCen[i]
            wcsPosCen = self.segMapWCS.pixel_to_world_values(pos[:,:,0].flatten()-1, pos[:,:,1].flatten()-1)
            self.sourceColl[i].addWCSSubPosCen(np.vstack( (wcsPosCen[0], wcsPosCen[1]) ).T.reshape(pixPos[i].shape[0], subsample**2, 2))

            # # for testing
            # wcsPosCen = self.segMapWCS.pixel_to_world_values([2022.50766-1,], [2056.92632-1,])
            # # wcsPosCen = self.segMapWCS.pixel_to_world_values([2023-1,], [2057-1,])

            # self.sourceColl[i].addWCSSubPosCen(np.vstack( (wcsPosCen[0], wcsPosCen[1]) ).T.reshape(1, subsample**2, 2))
            # self.sourceColl[i].WCSPos = self.sourceColl[i].WCSPos[:1]
            # self.sourceColl[i].WCSSubPos = self.sourceColl[i].WCSSubPos[:1]
            # self.sourceColl[i].WCSSubPos = self.sourceColl[i].WCSSubPos[:1]
            # self.sourceColl[i].segMapPixPos = self.sourceColl[i].segMapPixPos[:1]
            # self.sourceColl[i].segMapSubPos = self.sourceColl[i].segMapSubPos[:1]
            ####


        return


# the old function with no psf

    def pixAreaOverlap_polyclip(self):

        posList = np.array([p for source in self.sourceColl for p in source.WCSSubPos.reshape(source.WCSSubPos.shape[0]*self.subsample**2*4,2)], dtype=np.float64)
        objPosN = np.array([len(source.WCSSubPos)*self.subsample**2*4 for source in self.sourceColl], dtype=int)

        self.objPixA = []

        for img in self.imgs:
            self.objPixA.append(img.getPixArea_polyclip(posList, objPosN))


        return


    def pixAreaOverlap(self):

        posList = np.array([p for source in self.sourceColl for p in source.WCSSubPosCen.reshape(source.WCSSubPos.shape[0]*self.subsample**2,2)], dtype=np.float64)
        objPosN = np.array([len(source.WCSSubPos)*self.subsample**2 for source in self.sourceColl], dtype=int)

        self.objPixA = []

        for img in self.imgs:
            self.objPixA.append(img.getPixArea(posList, objPosN))


        return



class WFSSImageSimulator(WFSSBase):

    def __init__(self, directImage, segMapFile, rollang=0, subsample=1, disptype='prism', method=''):

        WFSSBase.__init__(self, segMapFile, subsample)

        self.directImage = directImage
        self.method = method

        try:
            with fits.open(directImage) as hdul:
                self.directImageHeader = hdul[1].header
                self.directImageData = np.array(hdul[1].data)

        except Exception as e:
            print("Error: ", e)
            sys.exit(-1)


        self.rollang = rollang if(isinstance(rollang, list)) else [rollang]


        for r in self.rollang:

            newheader = fits.Header(self.directImageHeader, copy=True)
            newwcs = WCS(newheader)

            # rotate counterclockwise by rollang (in deg)
            Rm =  np.array([[np.cos(r*np.pi/180),np.sin(r*np.pi/180)],[-np.sin(r*np.pi/180),np.cos(r*np.pi/180)]])
            newwcs.wcs.cd = Rm @ newwcs.wcs.cd
            newwcsheader = newwcs.to_header(relax=True)
            for h in ["1_1","1_2","2_1","2_2"]:
                newheader["CD"+h] = newwcsheader["PC"+h]

            newheader['FILTER'] = "SNPrism" if(disptype == 'prism') else "SNGrism"

            self.imgs.append(WFSSImage(header=newheader))

        for source in self.sourceColl:

            if(method == 'sar'):
                source.addSED(self.directImageHeader["FILTER"], "/Users/Ann/Desktop/NGRoman/testsed.txt")
                source.compute_weights(self.directImageData)
            elif(method == 'sar_bypixel'):
                source.addSED(self.directImageHeader["FILTER"], "/global/u1/a/aisaacs/BC03/BaSeL3.1_Atlas/Kroupa_IMF/truth/tau2_age1_norm.sed")
                source.compute_weights(self.directImageData)

            else:
                source.addPixSED(self.directImageHeader["FILTER"], len(self.imgs[0].spectralOrder.wavavg))

            self.numpix=source.segMapSubPos.shape[0]

        return


    def simulate(self, imgN, return_sim=False, save_fits=True, redo_objpix=False):

        if(not hasattr(self, 'objPixA')):
            if((self.method == 'sar') or (self.method == 'sar_bypixel')):
                self.pixAreaOverlap_polyclip()
            else:
                self.pixAreaOverlap()
        elif(redo_objpix==True):
            if (self.method == "sar") or (self.method == "sar_bypixel"):
                self.pixAreaOverlap_polyclip()
            else:
                self.pixAreaOverlap()

        sOrd = self.imgs[imgN].spectralOrder

        pix = np.zeros((4088, 4088), dtype=np.float64)

        # pix += np.random.normal(loc=0, scale=2, size=(4088,4088))

        prevposl = [0]
        for i in range(1, len(self.sourceColl)):
            prevposl.append(prevposl[i-1] + self.sourceColl[i-1].segMapPixPos.shape[0] * self.subsample**2)


        sens = sOrd.sensitivity

        wav = sOrd.wavelengths
        nwav = len(wav)
        dwav = sOrd.wavstep * 1e4


        for objID in range(len(self.sourceColl)):

            #print("Obj "+str(objID+1)+" / "+str(len(self.sourceColl))+"...", end=" ")

            npos = self.sourceColl[objID].segMapPixPos.shape[0] * self.subsample**2
            prevpos = prevposl[objID]

            xg, yg, val, indices, wavl, nn = get_val(self.objPixA[imgN], prevpos, npos, nwav)

            # need to apply a few things:
            # 1) sensitivity curve    (sens)
            # 2) flat field           (flat)
            # 3) relative pixel areas (area)
            # 4) source spectra       (flam)

            flam = np.array([sed(wav*10**4, fnu=False) for sed in self.sourceColl[objID].seds])

            dldr = sOrd.dispersion(np.array([xg, yg]).T, np.array([wav[i] for i in wavl], dtype=np.float64))

            val = multiply_val(val, flam, sens, wavl, wav, dwav, dldr, nn)

            # add flat field. still don't have it
            #val *= flat

            # sum over wavelengths
            vv, yy, xx = decimate(val, yg, xg, dims=(4088, 4088))


            # apply pixel areas -- we still don't have this information I think?
            # vv *= detdata.relative_pixelarea(xx, yy)

            # now sum into the image
            pix[yy-1, xx-1] += vv #+ np.random.normal(loc=0, scale=2*np.sqrt(abs(vv)))


            #print("done")

        if (save_fits==True):
            hdu = fits.PrimaryHDU(pix, self.imgs[imgN].header)
            hdu.writeto("sim_test"+str(imgN)+".fits", overwrite=True)

        if (return_sim==True):
            return pix
        return




class WFSSImageCollection(WFSSBase):


    def __init__(self, imgsList, segMapFile, targetIDs=1, subsample=1):

        WFSSBase.__init__(self, segMapFile, subsample)

        for img in imgsList:
            self.imgs.append(WFSSImage(img))

        self.nImgs = len(self.imgs)

        self.targetIDs = targetIDs if(isinstance(targetIDs, list)) else [targetIDs]

        return

    def setTargetIDs(self, targetIDs):

        self.targetIDs = targetIDs

        return

    def addTarget(self, tID):

        self.targetIDs.append(tID)

        return

    def getWFSSRegs(self):

        # Build a (N, 2) array with ALL the positions to consider, and keep track of which set of positions correspond to which source
        # so we can put everything back together after the call to getObjBBoxes. This way we do not copy all the source collection
        # but only the coordinates

        # posList = np.array([p for objPos in self.objPos for p in objPos], dtype=np.float64)
        posList = np.array([p for source in self.sourceColl for p in source.WCSPos], dtype=np.float64)
        # objPosN = np.array([p.shape[0] for p in self.objPos], dtype=int)
        objPosN = np.array([len(source.WCSPos) for source in self.sourceColl], dtype=int)


        # Call the function getObjBBoxes in parallel for every image
        with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context('fork')) as executor:
            futures = [executor.submit(img.getObjBBoxes, posList, objPosN) for img in self.imgs]

        self.objShapes = []
        self.objRegs = []
        self.objShapesArr = np.empty( (self.nImgs, self.nObjs, 4), dtype=np.float32 )

        i = 0
        for f in futures:
            bbox, regs, arr = f.result()
            self.objShapes.append(bbox)
            self.objRegs.append(regs)
            self.objShapesArr[i] = arr
            i = i+1

        return


    def buildContaminationMatrix(self):

        if(not hasattr(self, 'objShapes')):
            self.getWFSSRegs()

        m_or = getIntersectionMatrix(self.objShapesArr, self.nObjs, self.nImgs)

        return m_or


    def findContamination(self):
        """
        #### **Purpose:**
        The given code analyzes the interactions between a set of objects, represented by a symmetric matrix `m_or`.
        The goal of the code is to compute the connected components of target objects, where each target object is identified by an ID from the list `self.targetIDs`.
        The matrix `m_or` encodes whether two objects intersect (True if they intersect, False if they do not).
        #### **Inputs:**
        - `m_or`: A symmetric `n x n` matrix (where `n` is the number of objects), where:
          - `m_or[i][j] = True` if object `i` intersects object `j`.
        - `self.targetIDs`: A list of target object IDs to be analyzed.
        #### **Outputs:**
        - `self.targetsCont`: A list where each element is a numpy array representing a connected component of the target objects.
        A connected component is a set of objects that are either directly or indirectly connected through intersections.
        ---
        ### **Explanation of the Code:**
        1. **Initialization:**
           ```python
           self.targetsCont = []
           ```
           An empty list `self.targetsCont` is initialized. This will eventually hold the connected components for each target object.
        2. **Iterating over target object IDs:**
           ```python
           for targetID in self.targetIDs:
           ```
           The code loops over each `targetID` in `self.targetIDs` to process each target object individually.
        3. **Extracting the intersection information:**
           ```python
           v0 = m_or[targetID-1,:]
           vi = np.vstack([v0*False, v0])
           ```
           For each `targetID`:
           - `v0 = m_or[targetID-1,:]`: The row corresponding to the `targetID` (adjusted by `-1` for zero-based indexing) is extracted from the matrix `m_or`. This row represents the intersection status between the `targetID` and all other objects.
           - `vi = np.vstack([v0*False, v0])`: A new array `vi` is created by stacking two rows:
             - The first row is a zeroed-out version of `v0` (indicating no initial intersections).
             - The second row is just `v0`, representing the initial intersection states.
        4. **Expanding the connected component:**
           ```python
           while(np.sum(vi[-1,:]) != np.sum(vi[-2,:])):
               vi = np.vstack([vi, vi[-1,:] + np.any(m_or[vi[-1,:]*(~vi[-2,:])], axis=0)])
           ```
           The while-loop iterates to expand the connected component:
           - It checks if the current row (`vi[-1,:]`, the last row) has the same number of `True` values (intersections) as the previous row (`vi[-2,:]`). If not, the loop continues to expand the connected component.
           - `vi[-1,:] + np.any(m_or[vi[-1,:]*(~vi[-2,:])], axis=0)`: This expression looks for new intersections to add to the connected component:
             - `vi[-1,:]` represents the current state of intersections.
             - `(~vi[-2,:])` negates the previous row (to only consider new potential intersections).
             - `np.any(m_or[...], axis=0)` computes whether any new intersection exists between objects that were in the last expanded component and objects that were not yet included.
             - The result is added to the current row, which is then appended to the `vi` array.
        5. **Storing the connected component:**
           ```python
           self.targetsCont.append(vi[1:,:])
           ```
           After the while-loop concludes, the connected component corresponding to `targetID` is stored in `self.targetsCont`. The first row (`vi[0,:]`) is ignored because it was initialized as all `False` values and does not represent an actual intersection.
        ---
        ### **Key Observations:**
        - The matrix `m_or` is used to capture intersections between objects, and the code is determining how each target object (from `self.targetIDs`) is connected to others.
        - The while-loop performs an iterative expansion of the connected component for each target object, adding new objects that are indirectly connected through intersections.
        - The final result, `self.targetsCont`, contains a list of connected components, with each component being a numpy array of objects (rows of `m_or`) that are connected through direct or indirect intersections.
        ---
        """


        if(not hasattr(self, 'objShapes')):
            self.getWFSSRegs()


        if(not hasattr(self, 'm_or')):
            m_or = self.buildContaminationMatrix()
            self.m_or = m_or

        self.targetsCont = []

        for targetID in self.targetIDs:

            v0 = self.m_or[targetID-1,:]
            vi = [v0*False,v0]

            while(np.sum(vi[-1]) != np.sum(vi[-2])):
                vi.append(vi[-1]+np.any(self.m_or[vi[-1]*(~vi[-2])],axis=0))

            self.targetsCont.append(np.vstack([v for v in vi[1:]]))

        return

    def printRegs(self, deg=0):

        if(not hasattr(self, 'targetCont')):
            self.findContamination()


        for i in range(self.nImgs):
            self.imgs[i].printRegs(self.objRegs[i], self.targetIDs, self.nObjs, self.targetsCont, deg-1)

        return


    def buildMatrix(self, deg=0):

        if(not hasattr(self, 'targetCont')):
            self.findContamination()

        # if(not hasattr(self, 'objPixA')):
        #     self.pixAreaOverlap()

        cont_all = np.zeros((self.nObjs), dtype=bool)

        for tID in self.targetIDs:
            if(deg > self.targetsCont[tID-1].shape[0]-1):
                deg = -1
            cont_all += self.targetsCont[tID-1][deg]

        totpos = self.sourceColl[0].segMapPixPos.shape[0] * self.subsample**2
        prevposl = [0]
        for i in range(1, len(self.sourceColl)):
            prevposl.append(prevposl[i-1] + self.sourceColl[i-1].segMapPixPos.shape[0] * self.subsample**2)
            totpos += self.sourceColl[i].segMapPixPos.shape[0] * self.subsample**2


        sOrd = self.imgs[0].spectralOrder

        sens = sOrd.sensitivity

        wav = sOrd.wavelengths
        nwav = len(wav)
        dwav = sOrd.wavstep * 1e4
        flat = 1

        ff = [np.ones(nwav)]

        # [1,x,x,x,m] |1| = |1|
        # [x,x,x,x,m] |x|   |x|
        # [x,x,x,x,m] |x|   |x|
        # [x,x,x,x,m] |x|   |x|
        # [x,x,x,x,m] |m|   |x|
        # [n,x,x,x,m]       |n|

        # m = number of lamba x number of objects
        # n = number of pixels of the prism images

        #print(self.nObjs, self.nImgs)

        dim_x = 4088
        dim_y = 4088

        nRows = dim_x * dim_y * self.nImgs # n
        # nCols = nwav * totpos # m
        nCols = sOrd.bin_n * totpos # m

        self.ncols = nCols

        pixToIndex = np.arange(dim_x * dim_y, dtype=int).reshape(dim_x, dim_y)

        row_i = []
        col_i = []
        aij = []

        for imgN in range(self.nImgs):

            #print("Image #"+str(imgN+1))

            # objPA = self.objPixA[imgN] # I can call the img->getpixarea function here


            #########################
            posList = np.array([p for source in self.sourceColl for p in source.WCSSubPosCen.reshape(source.WCSSubPos.shape[0]*self.subsample**2,2)], dtype=np.float64)
            objPosN = np.array([len(source.WCSSubPos)*self.subsample**2 for source in self.sourceColl], dtype=int)
            objPA = self.imgs[imgN].getPixArea(posList, objPosN)
            #########################

            for objID in range(self.nObjs):

                #print("Obj #"+str(objID+1))


                npos = self.sourceColl[objID].segMapPixPos.shape[0] * self.subsample**2
                prevpos = prevposl[objID]

                xg, yg, val, indices, wavl, nn = get_val(objPA, prevpos, npos, nwav)

                rowIndex = imgN * dim_x * dim_y + pixToIndex[yg - 1, xg - 1]
                colIndex = nn * sOrd.bin_n + sOrd.bin_index[wavl]

                # dldr = sOrd.dispersion(np.array([xg, yg]).T, np.array([wav[i] for i in wavl], dtype=np.float64))

                dldr = np.empty(len(sens))

                val = multiply_val(val, np.array(ff*npos), sens, wavl, wav, dwav, dldr, nn)
                val *= flat

                val, rowIndex, colIndex = decimate(val, rowIndex, colIndex, dims=(nRows, nCols)) # this is taking a while...

                #print(len(rowIndex), len(colIndex), min(colIndex), max(colIndex), max(wavl))

                row_i.append(rowIndex)
                col_i.append(colIndex)
                aij.append(val)


        #print("building matrix")
        rowi, coli, aval = make_array(row_i, col_i, aij)

        A = coo_matrix((aval, (rowi, coli)), shape=(nRows, nCols))

        #print("done", A.get_shape(), len(aval), len(aval)/(nRows*nCols), A.size*np.float64().nbytes/1e9)

        self.Amat = A
        # self.Anorm = norm(A)
        self.Anorm = np.sqrt(np.sum(np.abs(aval)**2)) # it seems much faster

        return


    def buildDataVector(self):

        self.readImgsData()

        b = np.empty(4088*4088*self.nImgs, dtype=np.float64)

        i = 0
        for img in self.imgs:
            b[i*4088*4088:(i+1)*4088*4088] = img.imgData.ravel() #+ np.random.normal(loc=0, scale=3*np.sqrt(abs(img.imgData.ravel())))
            i += 1

        self.bvec = b

        return

    def readImgsData(self):

        for img in self.imgs:
            #print("Reading img "+img.imgFile)
            img.getImage()

        return


    def solve(self, lamb):

        if(not hasattr(self, 'Amat')):
            self.buildMatrix()

        if(not hasattr(self, 'bvec')):
            self.buildDataVector()

        damp = 10**lamb * self.Anorm
        x0 = np.ones(self.ncols) * 1e-20

        r = lsqr(self.Amat, self.bvec, atol=1e-6, btol=1e-6, damp=damp, calc_var=True, show=True, x0=x0)
        # (x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var) = r

        wavelengths = self.imgs[0].spectralOrder.wavavg

        ww = np.tile(wavelengths, int(self.Amat.shape[1]/len(wavelengths)))

        df = pd.DataFrame({"w":ww,"f":r[0]})#/(dldrs*10**4)*(0.001*10**4)})
        df.to_csv("result.txt", index=False, sep=" ", header=False)

        rr = r[0].reshape(-1,len(wavelengths)).sum(axis=0)
        df = pd.DataFrame({"w":wavelengths,"f":rr})
        df.to_csv("result_sum.txt", index=False, sep=" ", header=False)

        # df = pd.DataFrame({"w":wavelengths,"f":dldr})
        # df.to_csv("dldr.txt", index=False, sep=" ", header=False)

        return r

    def xy(self, r):
        return np.log10(r[3]), np.log10(r[8])


    def gridoptimize(self):
        """
        Method to call the optimizer

        Parameters
        ----------
        matrix : `su.core.modules.extract.multi.Matrix`
           The matrix operator to optimize

        Returns
        -------
        state : `su.core.modules.extract.multi.Result`
           The optimized state
        """

        # compute a grid of log damping
        logdamp = np.arange(-5, -1, 0.05, dtype=float)

        maxcurv = -np.inf
        state0 = self.solve(logdamp[0])
        state = state1 = self.solve(logdamp[1])

        for i in range(1, len(logdamp) - 1):
            state2 = self.solve(logdamp[i + 1])

            curv = self.menger(self.xy(state0), self.xy(state1), self.xy(state2))
            if curv > maxcurv:
                maxcurv = curv
                state = state1
                dampu = logdamp[i]

            state0 = state1
            state1 = state2

            #print("#########", i, logdamp[i], curv, maxcurv)


        self.result = state

        #print(dampu)

        wavelengths = self.imgs[0].spectralOrder.wavelengths
        dldr = self.imgs[0].spectralOrder.dispersion(np.array([2023, 2057]).T, wavelengths)

        ww = np.tile(wavelengths, int(self.Amat.shape[1]/len(wavelengths)))
        dldrs = np.tile(dldr, int(self.Amat.shape[1]/len(wavelengths)))


        df = pd.DataFrame({"w":ww,"f":state[0]})
        df.to_csv("result.txt", index=False, sep=" ", header=False)



        # df = pd.DataFrame({"w":ww,"f":state[0]/(dldrs*10**4)*(0.001*10**4)})
        # df.to_csv("result.txt", index=False, sep=" ", header=False)


        return state





    def menger(self, xyj, xyk, xyl):
        """
        Function to compute the Menger curvature from three (x,y) pairs

        Parameters
        ----------
        xyj : 2-tuple
           The lower pair of points (x,y)

        xyk : 2-tuple
           The central pair of points (x,y)

        xyl : 2-tuple
           The upper pair of points  (x,y)

        Returns
        -------
        curv : float
           The curvature at the central pair of points

        Notes
        -----
        1) The Menger curvature is a measure of the local curvature that is equal
           to the reciprocal of the radius of a circle that contains the
           current point and adjacent points.

        2) see https://en.wikipedia.org/wiki/Menger_curvature


        """
        # xyj=np.array(xyj)
        # xyk=np.array(xyk)
        # xyl=np.array(xyl)

        if np.allclose(xyj, xyk) or np.allclose(xyk, xyl) or np.allclose(xyj, xyl):
            # if any of the points are the same, then the curvature should be zero
            curv = 0.
        else:

            # could maybe consider subtracting off xyk from xyj and xyl to
            # improve the precision?

            num = 2. * np.abs(xyj[0] * (xyk[1] - xyl[1])
                              + xyk[0] * (xyl[1] - xyj[1])
                              + xyl[0] * (xyj[1] - xyk[1]))

            djk = np.hypot(xyk[0] - xyj[0], xyk[1] - xyj[1])
            dkl = np.hypot(xyl[0] - xyk[0], xyl[1] - xyk[1])
            dlj = np.hypot(xyj[0] - xyl[0], xyj[1] - xyl[1])
            den = djk * dkl * dlj

            curv = num / den

        return curv
