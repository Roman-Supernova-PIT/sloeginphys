import numpy as np
from numpy.polynomial.polynomial import polyval
import numba as nb
# from shapely import box as SBox
from scipy.special import erf

@nb.jit(nopython=True)
def eval2D(x0, y0, coeffs, Nc, No, xe, ye):
    l = np.empty(No, dtype=np.float64)

    for i in range(No):
        l[i] = x0**xe[i] * y0**ye[i]

    c = np.empty(Nc, dtype=np.float64)
    for i in range(Nc):
        c[i] = np.dot(coeffs[i], l)

    return c

@nb.jit(parallel = True, nopython=True) # it probably doesn't lead to much improvement to run this in parallel if N is not big "enough"
def evalAll2D(positions, coeffs, xe, ye):

    N = len(positions)
    Nc = coeffs.shape[0]
    No = coeffs.shape[1]


    c = np.empty((N,Nc), dtype=np.float64)

    for i in nb.prange(N):
        c[i] = eval2D(positions[i][0], positions[i][1], coeffs, Nc, No, xe, ye)

    return c

@nb.jit(parallel=True, nopython=True)
def evalN(p0, coeffs):
    Np = p0.shape[0]
    Nw = p0.shape[1]

    r = np.empty((Np, Nw), dtype=np.float64)

    for i in nb.prange(Np):
        for j in nb.prange(Nw):
            r[i][j] = eval1D(p0[i][j], coeffs[i])

    return r

@nb.jit(nopython=True)
def eval1D(p0, coeffs):
    if(len(coeffs) == 0):   # is this really necessary in our case?
        return 0
    return polyval(p0, coeffs)


@nb.jit(parallel=True, nopython=True)
def invert(wavelengths, coeffs, dpdt, d2pdt2):
    Nw = len(wavelengths)
    Np = coeffs.shape[0]
    t = np.empty((Np, Nw), dtype=np.float64)

    for i in nb.prange(Np):
        for j in nb.prange(Nw):
            t[i][j] = halley_solver(wavelengths[j], coeffs[i], dpdt[i], d2pdt2[i])

    return t


@nb.jit(parallel=True, nopython=True)
def invert_single(wavelengths, coeffs, dpdt, d2pdt2):
    Nw = len(wavelengths)
    Np = coeffs.shape[0]
    t = np.empty((Np,1), dtype=np.float64)

    for i in nb.prange(Np):
            t[i][0] = halley_solver(wavelengths[i], coeffs[i], dpdt[i], d2pdt2[i])

    return t


@nb.jit(nopython=True)
def within_tol(x, y, atol, rtol):
    return bool(np.abs(x - y) <= atol + rtol * np.abs(y))

@nb.jit(nopython=True)
def halley_solver(w0, coeffs, dpdt, d2pdt2):

    p0 = np.float64(0.5)

    tol = 1.e-8

    for _ in range(99):
        fval = eval1D(p0, coeffs)
        if within_tol(fval, w0, tol, 0.0):
            return p0

        fder = eval1D(p0, dpdt)
        if within_tol(fder, 0.0, tol, 0.0):
            return np.float64(np.inf)
        newton_step = (fval - w0) / fder

        fder2 = eval1D(p0, d2pdt2)
        adj = 0.5 * newton_step * fder2 / fder

        if np.abs(adj) < 1:
            newton_step /= 1.0 - adj

        p0 -= newton_step

    return np.float64(np.inf)




# there is probably no need for numba here, just an example
@nb.jit(nopython=True)
def getBBox(pixCoo):

    xmin = np.min(pixCoo[:,0]) - 0.5
    xmax = np.max(pixCoo[:,0]) + 0.5
    ymin = np.min(pixCoo[:,1]) - 0.5
    ymax = np.max(pixCoo[:,1]) + 0.5

    return np.array([xmin, xmax, ymin, ymax], dtype=np.float64)

@nb.jit(parallel = True, nopython=True)
def getBBoxAll(pixCoo, objPosN):

    objNBB = np.empty((objPosN.shape[0], 4), dtype=np.float64)

    for i in nb.prange(objPosN.shape[0]):
        i_prev = np.sum(objPosN[:i])
        objNBB[i] = getBBox(pixCoo[i_prev:i_prev+objPosN[i]])

    p1 = np.vstack((objNBB[:,0],objNBB[:,2])).T
    p2 = np.vstack((objNBB[:,1],objNBB[:,2])).T
    p3 = np.vstack((objNBB[:,1],objNBB[:,3])).T
    p4 = np.vstack((objNBB[:,0],objNBB[:,3])).T

    return np.vstack((p1, p2, p3, p4))


@nb.jit(parallel = True, nopython=True)
def makemap(ii, jj):

    gal = np.array([2036, 2044, 8], dtype=np.float64)
    sn = np.array([2022.5, 2057, 1], dtype=np.float64)
    # gal = np.array([2036.4357,2044.1993,8.2371763,4.1353833,237.11908], dtype=no.float64)

    objs = np.empty((1,3),dtype=np.float64)
    objs[0] = gal #sn
    # objs[1] = gal

    s = np.zeros((ii, jj), dtype=np.int32)
    # 2022.50766e+03   2056.92632e+03
    for i in nb.prange(ii):
        for j in nb.prange(jj):
            for n in np.arange(objs.shape[0]):
                obj = objs[n]
                if(np.sqrt( (j-(obj[0]-1))**2 + (i-(obj[1]-1))**2 ) <= obj[2] ):
                    s[i][j] = n+1
                    break

    return s

# @nb.jit(parallel = True)
# def getpos(mapdata, n):

#     pos = [np.where(mapdata == 0)] * n

#     for i in nb.prange(n):
#         pos[i] = np.where(mapdata == 1)

#     return pos

@nb.jit(parallel = True, nopython=True)
def getObjsPos(segMapData, nObjs):

    pos = [np.where(segMapData == 0)] * nObjs

    for i in nb.prange(nObjs):
        y, x = np.where(segMapData == i+1)
        pos[i] = (y+1, x+1)

    return pos

@nb.jit(parallel = True, nopython=True)
def subsamplePos(pixPos, nObjs, subsample):

    posMax = np.max(np.array([p.shape[0] for p,_ in pixPos]))

    subPos = [np.empty( (posMax, subsample**2, 4, 2), dtype=np.float64)] * nObjs


    for i in nb.prange(nObjs):
        posy, posx = pixPos[i]

        subPos_i = np.empty((posx.shape[0], subsample**2, 4, 2), dtype=np.float64)
        # subPos_i = np.empty((posx.shape[0], subsample**2, 2), dtype=np.float64)

        for j in range(posx.shape[0]):
            # Generate the subsampled grid for xx and yy
            xx = np.arange(posx[j] - 0.5, posx[j] + 0.5 + 1 / (subsample), 1 / (subsample))[:subsample + 1]
            yy = np.arange(posy[j] - 0.5, posy[j] + 0.5 + 1 / (subsample), 1 / (subsample))[:subsample + 1]

            # # Loop over each subregion and compute the four corners
            for sx in range(subsample):
                for sy in range(subsample):
                    # Define the corners of each subregion
                    bottom_left = [xx[sx], yy[sy]]  # min(xx), min(yy)
                    bottom_right = [xx[sx+1], yy[sy]]  # max(xx), min(yy)
                    top_left = [xx[sx], yy[sy+1]]  # min(xx), max(yy)
                    top_right = [xx[sx+1], yy[sy+1]]  # max(xx), max(yy)

                    # Store the corners in subPos_i
                    subPos_i[j, sx * subsample + sy] = np.array([bottom_left, bottom_right, top_right, top_left], dtype=np.float64)

            # # Loop over each subregion and compute the center of the subregion
            # for sx in range(subsample):
            #     for sy in range(subsample):
            #         # Define the corners of each subregion
            #         x_center = (xx[sx] + xx[sx + 1]) / 2  # Center of x-axis
            #         y_center = (yy[sy] + yy[sy + 1]) / 2  # Center of y-axis

            #         # Store the center of the subregion in subPos_i
            #         subPos_i[j, sx * subsample + sy] = np.array([x_center, y_center], dtype=np.float64)


        # Store the subregions' corners for all pixels in the main array subPos
        subPos[i] = subPos_i


    return subPos


@nb.jit(parallel = True, nopython=True)
def subsamplePosCen(pixPos, nObjs, subsample):

    posMax = np.max(np.array([p.shape[0] for p,_ in pixPos]))

    subPos = [np.empty( (posMax, subsample**2, 2), dtype=np.float64)] * nObjs


    for i in nb.prange(nObjs):
        posy, posx = pixPos[i]

        subPos_i = np.empty((posx.shape[0], subsample**2, 2), dtype=np.float64)
        # subPos_i = np.empty((posx.shape[0], subsample**2, 2), dtype=np.float64)

        for j in range(posx.shape[0]):
            # Generate the subsampled grid for xx and yy
            xx = np.arange(posx[j] - 0.5, posx[j] + 0.5 + 1 / (subsample), 1 / (subsample))[:subsample + 1]
            yy = np.arange(posy[j] - 0.5, posy[j] + 0.5 + 1 / (subsample), 1 / (subsample))[:subsample + 1]

            # # Loop over each subregion and compute the four corners
            for sx in range(subsample):
                for sy in range(subsample):
                    # Define the corners of each subregion
                    bottom_left = [xx[sx], yy[sy]]  # min(xx), min(yy)
                    bottom_right = [xx[sx+1], yy[sy]]  # max(xx), min(yy)
                    top_left = [xx[sx], yy[sy+1]]  # min(xx), max(yy)
                    top_right = [xx[sx+1], yy[sy+1]]  # max(xx), max(yy)
                    x_center = (xx[sx] + xx[sx + 1]) / 2  # Center of x-axis
                    y_center = (yy[sy] + yy[sy + 1]) / 2  # Center of y-axis

                    # Store the corners in subPos_i
                    subPos_i[j, sx * subsample + sy] = np.array([x_center, y_center], dtype=np.float64)

            # # Loop over each subregion and compute the center of the subregion
            # for sx in range(subsample):
            #     for sy in range(subsample):
            #         # Define the corners of each subregion
            #         x_center = (xx[sx] + xx[sx + 1]) / 2  # Center of x-axis
            #         y_center = (yy[sy] + yy[sy + 1]) / 2  # Center of y-axis

            #         # Store the center of the subregion in subPos_i
            #         subPos_i[j, sx * subsample + sy] = np.array([x_center, y_center], dtype=np.float64)


        # Store the subregions' corners for all pixels in the main array subPos
        subPos[i] = subPos_i


    return subPos

# @nb.jit(parallel = True)
# def sumMatrices(int_mat):

#     m_or = int_mat[0]

#     for i in nb.prange(int_mat.shape[0]):
#         m_or += int_mat[i]


#     return m_or


# @nb.jit(parallel = True)
# def getContForTarg(targetIDs, m_or):

#     nobjs = m_or.shape[0]
#     ntargs = targetIDs.shape[0]

#     v0 = np.zeros( (ntargs, 15, nobjs), dtype=np.bool_)

#     for t in nb.prange(len(targetIDs)):

#         tID = targetIDs[t]
#         vi = v0[tID]
#         vi[1] = m_or[tID,:]
#         i = 2

#         while(np.sum(vi[i-1]) != np.sum(vi[i-2]) and i<15):

#             for j in nb.prange(nobjs):
#                 for k in range(j):
#                     if(vi[i-1][k] == True and vi[i-1][j]+m_or[k][j] == True):
#                         vi[i][j] = True
#                         break

#             i += 1
#         print(i)

#     return v0


# @nb.jit
# def findCont(int_mat, tID):

#     m_or = int_mat[0]

#     for i in nb.prange(int_mat.shape[0]):
#         m_or += int_mat[i]

#     v0 = np.delete(m_or, tID-1)

#     m_red = np.delete(np.delete(m_or, tID-1, axis=0), tID-1, axis=1)

#     vprev = v0*False
#     vi = v0

#     while(not np.all(np.equal(vi,vprev))):
#         vprev = vi
#         vi = vprev+np.any(m_red[vprev],axis=0)

#     return vi


# @nb.jit(parallel = True)
# def makePypolyclipInput(pixCoo, dx, dy):

#     nPos = dx.shape[0]
#     nLambda = dx.shape[1]

#     px = np.empty( (nLambda, int(nPos/4), 4), dtype=np.float64)
#     py = np.empty( (nLambda, int(nPos/4), 4), dtype=np.float64)

#     for l in nb.prange(nLambda):

#         pxl = np.empty( (int(nPos/4), 4), dtype=np.float64)
#         pyl = np.empty( (int(nPos/4), 4), dtype=np.float64)

#         for i in nb.prange(int(nPos/4)):

#             pxl[i,:] = pixCoo[4*i:4*i+4,0] + dx[4*i:4*i+4,l]
#             pyl[i,:] = pixCoo[4*i:4*i+4,1] + dy[4*i:4*i+4,l]

#         px[l] = pxl
#         py[l] = pyl

#     return px.reshape(-1,4), py.reshape(-1,4)




@nb.jit(nopython=True)
def myclip(value, low, high):
    return max(min(value, high), low)


@nb.jit(parallel = True, nopython=True)
def makePypolyclipInput(pixCoo, dx, dy, nx, ny):

    nPos = dx.shape[0]
    nLambda = dx.shape[1]

    nPos4 = nPos // 4

    px = np.empty( (nLambda*nPos), dtype=np.float32)
    py = np.empty( (nLambda*nPos), dtype=np.float32)

    l = np.empty( (nLambda*nPos4), dtype=np.int32)
    r = np.empty( (nLambda*nPos4), dtype=np.int32)
    b = np.empty( (nLambda*nPos4), dtype=np.int32)
    t = np.empty( (nLambda*nPos4), dtype=np.int32)
    npix = 0

    for lb in nb.prange(nLambda):

        for i in nb.prange(nPos4):

            idx = lb*nPos + 4*i

            px[idx:idx+4] = pixCoo[4*i:4*i+4,0] + dx[4*i:4*i+4,lb] - 0.5 # because pypolyclip sets the integer positions on the pixel border
            py[idx:idx+4] = pixCoo[4*i:4*i+4,1] + dy[4*i:4*i+4,lb] - 0.5 # and not on the center

            idx4 = lb*nPos4 + i

            l[idx4] = myclip(int(np.min(px[idx:idx+4])), 0, nx)
            r[idx4] = myclip(int(np.max(px[idx:idx+4])), 0, nx)
            b[idx4] = myclip(int(np.min(py[idx:idx+4])), 0, ny)
            t[idx4] = myclip(int(np.max(py[idx:idx+4])), 0, ny)

            npix += (r[idx4] - l[idx4] + 1) * (t[idx4] - b[idx4] + 1)

    return px, py, l, r, b, t, nLambda*nPos4, npix

@nb.jit(parallel = True, nopython=True)
def getVals(pixCoo, dx, dy, nx, ny, wav, ngrid, psf_l):

    nPos = dx.shape[0]
    nLambda = dx.shape[1]

    npix = ngrid ** 2 # how many pixels for the PSF

    xc = np.empty( (npix * nLambda * nPos), dtype=np.int32)
    yc = np.empty( (npix * nLambda * nPos), dtype=np.int32)
    val = np.empty( (npix * nLambda * nPos), dtype=np.float64)

    # pixOff = np.array([-2,-1,0,1,2], dtype=np.float64)
    rr = (ngrid - 1) // 2
    pixOff = np.arange(-rr, rr + 1, 1)

    for lb in nb.prange(nLambda):

        psf = psf_l[lb]

        for p in nb.prange(nPos):

            idx = lb*nPos*npix + npix*p

            for j in nb.prange(ngrid):
                for i in nb.prange(ngrid):

                    xbar = pixCoo[p,0] + dx[p,lb]
                    ybar = pixCoo[p,1] + dy[p,lb]
                    xc[idx+i+ngrid*j] = pixOff[i] + xbar + 0.5
                    yc[idx+i+ngrid*j] = pixOff[j] + ybar + 0.5
                    # val[idx+i+ngrid*j] = fake_ePSF(xbar, ybar, pixOff[i], pixOff[j], lb, wav)
                    val[idx+i+ngrid*j] = rpsf_phot(xc[idx+i+ngrid*j] - xbar, yc[idx+i+ngrid*j] - ybar, psf)

    return xc, yc, val


@nb.jit(nopython=True)
def fake_ePSF(xbar, ybar, xpix, ypix, lb, wav):

    # sigma = np.linspace(0.5, 1.2, wav.shape[0]) # should depend on wavelength

    # sigu = sigma[lb]

    sigu = 2.2/2.35 * 1/1.58 * wav[lb]

    x1 = erf( ( xbar - np.floor(xbar + xpix + 0.5) - 0.5) / ( np.sqrt(2) * sigu ) )
    x2 = erf( ( xbar - np.floor(xbar + xpix + 0.5 ) + 0.5 ) / ( np.sqrt(2) * sigu ) )

    y1 = erf( ( ybar - np.floor(ybar + ypix + 0.5) - 0.5) / ( np.sqrt(2) * sigu ) )
    y2 = erf( ( ybar - np.floor(ybar + ypix + 0.5 ) + 0.5 ) / ( np.sqrt(2) * sigu ) )


    return 1/4 * (x1 - x2) * (y1 - y2)





# c--------------------------------------
# c
# c this will evaluate a PSF at a given (dx,dy)
# c offset;
# c
@nb.jit(nopython=True)
def rpsf_phot(dx, dy, psf):


    sub = 4

    rx = (251/2 - 0.5) + dx * sub
    ry = (251/2 - 0.5) + dy * sub
    ix = int(rx)
    iy = int(ry)
    fx = rx-ix
    fy = ry-iy

    A1 =  psf[iy  ,ix  ]
    B1 = (psf[iy  ,ix+1]-psf[iy  ,ix-1])/2
    C1 = (psf[iy+1,ix  ]-psf[iy-1,ix  ])/2
    D1 = (psf[iy  ,ix+1]+psf[iy  ,ix-1]-2*A1)/2
    F1 = (psf[iy+1,ix  ]+psf[iy-1,ix  ]-2*A1)/2
    E1 = (psf[iy+1,ix+1]-A1)

    A2 =  psf[iy  ,ix+1]
    B2 = (psf[iy  ,ix+2]-psf[iy  ,ix  ])/2
    C2 = (psf[iy+1,ix+1]-psf[iy-1,ix+1])/2
    D2 = (psf[iy  ,ix+2]+psf[iy  ,ix  ]-2*A2)/2
    F2 = (psf[iy+1,ix+1]+psf[iy-1,ix+1]-2*A2)/2
    E2 =-(psf[iy+1,ix  ]-A2)

    A3 =  psf[iy+1,ix  ]
    B3 = (psf[iy+1,ix+1]-psf[iy+1,ix-1])/2
    C3 = (psf[iy+2,ix  ]-psf[iy  ,ix  ])/2
    D3 = (psf[iy+1,ix+1]+psf[iy+1,ix-1]-2*A3)/2
    F3 = (psf[iy+2,ix  ]+psf[iy  ,ix  ]-2*A3)/2
    E3 =-(psf[iy  ,ix+1]-A3)

    A4 =  psf[iy+1,ix+1]
    B4 = (psf[iy+1,ix+2]-psf[iy+1,ix  ])/2
    C4 = (psf[iy+2,ix+1]-psf[iy  ,ix+1])/2
    D4 = (psf[iy+1,ix+2]+psf[iy+1,ix  ]-2*A4)/2
    F4 = (psf[iy+2,ix+1]+psf[iy  ,ix+1]-2*A4)/2
    E4 = (psf[iy  ,ix  ]-A4)


    V1 = A1 + B1*( fx ) + C1*( fy ) + D1*( fx )**2 + E1*( fx )*( fy ) + F1*( fy )**2

    V2 = A2 + B2*(fx-1) + C2*( fy ) + D2*(fx-1)**2 + E2*(fx-1)*( fy ) + F2*( fy )**2

    V3 = A3 + B3*( fx ) + C3*(fy-1) + D3*( fx )**2 + E3*( fx )*(fy-1) + F3*(fy-1)**2

    V4 = A4 + B4*(fx-1) + C4*(fy-1) + D4*(fx-1)**2 + E4*(fx-1)*(fy-1) + F4*(fy-1)**2

    rpsf_phot = (1-fx)*(1-fy)*V1 + ( fx )*(1-fy)*V2 + (1-fx)*( fy )*V3 + ( fx )*( fy )*V4


    return rpsf_phot



@nb.jit(parallel = True, nopython=True)
def get_val(pixAO, nprev, npos, nwav):

    xg, yg, val, indices = pixAO

    wavind0 = indices[1:].reshape(nwav, int(indices[1:].shape[0]/nwav))
    wavind = np.empty((wavind0.shape[0], wavind0.shape[1]+1), dtype=np.int32)
    wavind[0,0] = 0
    wavind[:,1:] = wavind0

    for i in nb.prange(1, wavind0.shape[0]):
        wavind[i,0] = wavind0[i-1, -1]

    nusewav = (wavind[:, nprev + npos] - wavind[:, nprev]).T

    nuseadd = np.empty(nusewav.shape[0], dtype=np.int32)

    for i in range(nuseadd.shape[0]):
        nuseadd[i] = np.sum(nusewav[:i])

    ntot = np.sum(nusewav)

    xn = np.empty(ntot, dtype=np.int32)
    yn = np.empty(ntot, dtype=np.int32)
    vn = np.empty(ntot, dtype=np.float64)
    wn = np.empty(ntot, dtype=np.int32)
    nn = np.empty(ntot, dtype=np.int32)

    for w in nb.prange(nwav):
        nuseprev = nuseadd[w]
        nobjprev = 0
        for n in range(npos):
            i1 = wavind[w][nprev + n]
            i2 = wavind[w][nprev + n+1]
            xn[nuseprev + nobjprev : nuseprev + nobjprev + (i2-i1)] = xg[i1:i2]
            yn[nuseprev + nobjprev : nuseprev + nobjprev + (i2-i1)] = yg[i1:i2]
            vn[nuseprev + nobjprev : nuseprev + nobjprev + (i2-i1)] = val[i1:i2]
            wn[nuseprev + nobjprev : nuseprev + nobjprev + (i2-i1)] = np.ones((i2-i1), dtype=np.int32)*w
            nn[nuseprev + nobjprev : nuseprev + nobjprev + (i2-i1)] = n

            nobjprev += i2-i1

    return xn, yn, vn, wavind, wn, nn





@nb.jit(parallel = True, nopython=True)
def multiply_val(val, flam, sens, wavl, wave, dwav, dldr, nn):

    hc = 6.626196e-27 * 2.99792458e10 # (erg * cm)

    # flam -> erg/(cm²·s·A)
    # fphot -> phot/(cm2*nm*s)
    # dldr -> um / pix ?
    # flam * lambda/hc -> photons/(cm^2*s*A)
    # dldr * 1e4 -> A/pix

    expt = 302.275
    # expt = 1000

    collecting_area = 3.757e4

    for i in nb.prange(len(wavl)):
        wuse = wavl[i]
        val[i] *= flam[nn[i]][wuse] * (wave[wuse]*1e-4/hc) * sens[wuse] * expt * dwav * collecting_area #* dldr[i]

    return val



@nb.jit(parallel = False, nopython=True)
def make_array(rowi, coli, aval):

    n = np.zeros( len(rowi), dtype=np.int32)
    for i in nb.prange(len(rowi)):
        n[i] += len(rowi[i])

    ntot = n.sum()

    r = np.empty( ntot, dtype=np.int32)
    c = np.empty( ntot, dtype=np.int32)
    v = np.empty( ntot, dtype=np.float64)

    for i in nb.prange(len(rowi)):
        nu = n[i]
        npr = n[:i].sum()
        r[npr:npr+nu] = rowi[i]
        c[npr:npr+nu] = coli[i]
        v[npr:npr+nu] = aval[i]

    return r, c, v


@nb.jit(parallel = True, nopython=True)
def myfunc(nImgs, nObjs, nwav, objPixA, objPosN, prevposl):

    nRows = 4088 * 4088 * nImgs # n
    nCols = nwav * nObjs # m

    pixToIndex = np.arange(4088 * 4088, dtype=int).reshape(4088, 4088)

    row_i = []
    col_i = []
    aij = []

    for imgN in range(self.nImgs):

        print("Image #"+str(imgN+1))

        objPA = self.objPixA[imgN]

        for objID in range(self.nObjs):

            print("Obj #"+str(objID+1))


            npos = self.sourceColl[objID].segMapPixPos.shape[0] * self.subsample**2
            prevpos = prevposl[objID]
            xg, yg, val, indices, wavl = get_val(self.objPixA[imgN], prevpos, npos, np.ones(npos), nwav)

            rowIndex = imgN * 4088 * 4088 + pixToIndex[xg, yg]
            colIndex = objID * nwav + wavl

            # dldr = sOrd.dispersion(np.array([xg, yg]).T, np.array([wav[i] for i in wavl], dtype=np.float64))

            val = multiply_val(val, ff, sens, wavl, wav, dwav)
            val *= flat

            # print(len(rowIndex), len(colIndex), max(rowIndex), max(wavl))

            row_i.append(rowIndex)
            col_i.append(colIndex)
            aij.append(val)

    return



@nb.jit(parallel = True, nopython=True)
def getIntersectionMatrix(shapes, nobj, nimg):

    m_or = np.empty( (nobj, nobj), dtype=np.bool_)

    for i in nb.prange(nobj):
        m_or[i, i] = True
        for j in range(i+1, nobj):  # Only compute for j > i (upper triangle)
            intersects_all = False
            k = 0
            intersects_i = False
            while( k<nimg and intersects_i == False ):
                intersects_i = getIntersection(shapes[k][i], shapes[k][j])
                k = k+1

            intersects_all = intersects_i

            m_or[i, j] = intersects_all
            m_or[j, i] = intersects_all

    return m_or



@nb.jit(nopython=True)
def getIntersection(obj1, obj2):

    # https://github.com/SFML/SFML/blob/12d81304e63e333174d943ba3ff572e38abd56e0/include/SFML/Graphics/Rect.inl#L109
    # objN = xmin ymin xmax ymax

    interLeft = max(obj1[0], obj2[0])
    interTop = max(obj1[1], obj2[1])
    interRight = min(obj1[2], obj2[2])
    interBottom = min(obj1[3], obj2[3])

    return ((interLeft <= interRight) and (interTop <= interBottom))
