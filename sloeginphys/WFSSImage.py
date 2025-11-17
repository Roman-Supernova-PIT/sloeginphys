import sys
import os
import concurrent.futures
import multiprocessing as mp
from astropy.io import fits
from astropy.wcs import WCS
from shapely import box as SBox
from pypolyclip import clip_multi
from pypolyclip import polyclip
from .SpectralOrder import SpectralOrder
from .myUtils import *

class WFSSImage:

    def __init__(self, imgFile=0, header=0):

        '''
        read image header
        create SpectralOrder object with info from header
        create a WCS object from header

        after init:
        self.spectralOrder = new SpectralOrder
        self.WCS = new WCS (astropy?)
        self.imgFile = imgFile (string)
        '''

        if(header == 0):
            try:

                header = fits.getheader(imgFile)

            except Exception as e:

                print("Error: ", e)
                sys.exit(-1)

            self.imgFile = imgFile

        self.header = header

        disptype = "prism" if header["FILTER"] == "SNPrism" else "grism"
        self.spectralOrder = SpectralOrder(header["SCA_NUM"], disptype)
        self.WCS = WCS(header)

        return


    def getImage(self):

        '''
        read image file and return it
        astropy or simple numpy array?
        asdf?
        '''

        if(not hasattr(self, "imgData")):
            try:

                with fits.open(self.imgFile) as hdul:
                    self.imgData = hdul[0].data

            except Exception as e:
                print("Error: ", e)
                sys.exit(-1)


        return self.imgData


    def dispersion(self, positions):

        return self.spectralOrder.dispersion(positions)


    def deltas(self, positions):

        return self.spectralOrder.deltas(positions)



    def getObjBBoxes(self, objPos, objPosN):
        """
        Calculates bounding boxes of the objects' traces.

        Parameters:
        objPos (numpy.ndarray): A 2D array of shape (N, 2) where N is the number of objects. Each row represents
                                 the world coordinates (x, y) of an object.
        objPosN (numpy.ndarray): A 2D array of shape (N,) where N is the number of objects. This array keeps track
                                 of which posiions belong to which object.

        Returns:
        tuple:
            - boxes (SBox): A set of bounding boxes created from the calculated min and max pixel coordinates for each object.
            - bbox (numpy.ndarray): A 3D array of shape (4, N, 2) representing the coordinates of each bounding box
                                     (top-left and bottom-right corners).
            - pmin (numpy.ndarray): A 2D array of shape (N, 2) representing the minimum coordinates (x, y) for each bounding box.
            - pmax (numpy.ndarray): A 2D array of shape (N, 2) representing the maximum coordinates (x, y) for each bounding box.
        """

        pixCoo = self.WCS.world_to_pixel_values(objPos[:,0], objPos[:,1])
        pixCoo = np.vstack( (pixCoo[0]+1, pixCoo[1]+1) ).T

        bbox = getBBoxAll(pixCoo, objPosN)

        dx, dy = self.deltas(bbox)

        nObj = objPosN.shape[0]

        bbox = bbox.reshape(4, nObj, 2)

        bbox[:2,:,0] = bbox[:2,:,0] + (dx.min(axis=1).reshape(4,nObj)[:2])
        bbox[2:,:,0] = bbox[2:,:,0] + (dx.max(axis=1).reshape(4,nObj)[2:])

        bbox[:2,:,1] = bbox[:2,:,1] + (dy.min(axis=1).reshape(4,nObj)[:2])
        bbox[2:,:,1] = bbox[2:,:,1] + (dy.max(axis=1).reshape(4,nObj)[2:])

        pmin = np.array([[bbox[:,obj,0].min(),bbox[:,obj,1].min()] for obj in range(nObj)])
        pmax = np.array([[bbox[:,obj,0].max(),bbox[:,obj,1].max()] for obj in range(nObj)])

        boxes = SBox(pmin[:,0], pmin[:,1], pmax[:,0], pmax[:,1])

        # self.printRegions(bbox, nObj, tID)

        return boxes, bbox, np.array([pmin[:,0], pmin[:,1], pmax[:,0], pmax[:,1]], dtype=np.float64).T


    def printRegs(self, bbox, tIDs, nObj, cont, deg):

        for i in range(len(tIDs)):

            tID = tIDs[i]
            if(deg > cont[i].shape[0]-1):
                deg = -1
            cont_i = cont[i][deg]

            cen = [(bbox[:,obj,0].mean(),bbox[:,obj,1].mean()) for obj in range(nObj)]
            wid = [(bbox[:,obj,0].max()-bbox[:,obj,0].min(),bbox[:,obj,1].max()-bbox[:,obj,1].min(),0) for obj in range(nObj)]

            reg = np.append(cen, wid, axis=1)

            fname = self.imgFile.split("/")[-1].split(".")[0]
            header = "# Region file format: DS9 version 4.1\n"
            header += "global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n"
            header += "image\n"

            with open(fname+"_t"+str(tID).zfill(3)+".reg","w") as ff:
                ff.write(header)

                for objN in range(reg.shape[0]):
                    # print(self.imgs[objImg].imgFile)
                    # print(self.objShapes[objImg][objN])
                    s = reg[objN]
                    color = "white"
                    width = "1"
                    dash = "1"
                    text = "{%i}" % (objN+1)
                    if(objN+1 == tID):
                        color = "blue"
                        width = "3"
                        dash = "0"
                    elif(cont_i[objN]):
                        color = "cyan"
                        width = "3"
                        dash = "0"
                    ff.write("box(%8.3f, %8.3f, %8.3f, %8.3f, %8.3f) # color=%s width=%s move=0 dash=%s\n" % (s[0], s[1], s[2], s[3], s[4], color, width, dash))
                    ff.write("#text(%8.3f, %8.3f) color=%s width=3 font=\"helvetica 20 normal roman\" move=0 text=%s\n" % (s[0], s[1], color, text))

        return


    def getPixArea_polyclip(self, objPos, objPosN):
        """
        Converts world coordinates to pixel coordinates, calculates deltas, and computes the area of objects
        in pixel space using a polygon clipping algorithm. The function utilizes compiled C-code for efficient
        polygon area calculations, returning the pixel coordinates and areas of objects along with their indices.

        Parameters:
        objPos (numpy.ndarray): A 2D array of shape (N, 2) where N is the number of objects. Each row contains
                                 the world coordinates (x, y) of an object.
        objPosN (numpy.ndarray): A 2D array of shape (N,) where N is the number of objects. This array keeps track
                                 of which posiions belong to which object.

        Returns:
        tuple:
            - xc (numpy.ndarray): A 1D array of integer x-coordinates of the clipped polygon vertices in pixel space.
            - yc (numpy.ndarray): A 1D array of integer y-coordinates of the clipped polygon vertices in pixel space.
            - areas (numpy.ndarray): A 1D array of the area (in pixels) for each clipped polygon.
            - indices (numpy.ndarray): A 1D array of indices that divide the polygon points into separate polygons.

        Notes:
        - The function uses the `polyclip.multi` function, which is implemented in compiled C code, to calculate
          polygon areas and manage pixel clipping.
        """

        pixCoo = self.WCS.world_to_pixel_values(objPos[:,0], objPos[:,1])
        pixCoo = np.vstack( (pixCoo[0]+1, pixCoo[1]+1) ).T


        dx, dy = self.deltas(pixCoo)

        px, py, l, r, b, t, npoly, npix = makePypolyclipInput(pixCoo, dx, dy, self.header["NAXIS1"], self.header["NAXIS2"])

        indices = np.linspace(0, px.size, npoly + 1, dtype=np.int32)

        nclip = np.zeros(1, dtype=np.int32)

        areas = np.empty(npix, dtype=np.float32)
        xc = np.empty(npix, dtype=np.int32)
        yc = np.empty(npix, dtype=np.int32)


        polyclip.multi(l, r, b, t,
                       px,
                       py,
                       npoly, indices, xc, yc, nclip, areas)

        nclip = nclip[0]
        areas = areas[:nclip]
        xc = xc[:nclip]
        yc = yc[:nclip]

        #print(pixCoo.shape, dy.shape, px.shape, xc.shape, npix, nclip, len(indices))


        # len(indices) should be equal to nPolygons + 1

        return xc, yc, areas, indices


    def getPixArea(self, objPos, objPosN):
        """
        Converts world coordinates to pixel coordinates, calculates deltas, and computes the area of objects
        in pixel space using a polygon clipping algorithm. The function utilizes compiled C-code for efficient
        polygon area calculations, returning the pixel coordinates and areas of objects along with their indices.

        Parameters:
        objPos (numpy.ndarray): A 2D array of shape (N, 2) where N is the number of objects. Each row contains
                                 the world coordinates (x, y) of an object.
        objPosN (numpy.ndarray): A 2D array of shape (N,) where N is the number of objects. This array keeps track
                                 of which posiions belong to which object.

        Returns:
        tuple:
            - xc (numpy.ndarray): A 1D array of integer x-coordinates of the clipped polygon vertices in pixel space.
            - yc (numpy.ndarray): A 1D array of integer y-coordinates of the clipped polygon vertices in pixel space.
            - areas (numpy.ndarray): A 1D array of the area (in pixels) for each clipped polygon.
            - indices (numpy.ndarray): A 1D array of indices that divide the polygon points into separate polygons.

        Notes:
        - The function uses the `polyclip.multi` function, which is implemented in compiled C code, to calculate
          polygon areas and manage pixel clipping.
        """

        pixCoo = self.WCS.world_to_pixel_values(objPos[:,0], objPos[:,1])
        pixCoo = np.vstack( (pixCoo[0] + 1, pixCoo[1] + 1) ).T

        dx, dy = self.deltas(pixCoo)

        #### MUST BE ODD ####
        ngrid = 15
        #####################

        psf_l = np.empty((self.spectralOrder.wavelengths.shape[0], 251, 251), dtype=np.float64)
        for i in range(psf_l.shape[0]):
            psf_l[i] = self.spectralOrder.psfFunc(self.spectralOrder.wavelengths[i])

        xc, yc, val = getVals(pixCoo, dx, dy, self.header["NAXIS1"], self.header["NAXIS2"], self.spectralOrder.wavelengths, ngrid, psf_l)

        indices = np.arange(0, pixCoo.shape[0]*dx.shape[1]*(ngrid**2) + ngrid**2, ngrid**2)

        #print(pixCoo.shape, dx.shape, dy.shape, xc.shape, yc.shape, val.shape, indices.shape, (xc.nbytes+yc.nbytes+val.nbytes+indices.nbytes)/1e9)

        return xc, yc, val, indices
