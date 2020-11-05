from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
from numpy.fft import fft, fft2, fftshift
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling import models, fitting
from mpl_toolkits.axes_grid1 import make_axes_locatable
import galario.double as glr
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.convolution.kernels import CustomKernel

"""
Using lmfit
"""
def ring_gaussian(x, y, pix_size, amp, r0, x0, y0, fwhm, theta, i):
    r0 *= glr.arcsec
    x0 *= glr.arcsec
    y0 *= glr.arcsec

    amp *= (pix_size ** 2)

    sig2fwhm = 2.35482
    sigma = (fwhm / sig2fwhm) * glr.arcsec
    theta =(180-theta) * np.pi / 180.
    i *= np.pi/180
    x_new = (-(x-x0)*np.cos(theta) + (y-y0)*np.sin(theta))/np.cos(i)
    y_new = (x-x0)*np.sin(theta) + (y-y0)*np.cos(theta)


    r = np.sqrt((x_new) ** 2 + (y_new) ** 2)

    return amp*np.exp(-1/2*((r - r0)/sigma)**2)

def convolved_ring_gaussian(x, y, amp, r0, x0, y0, fwhm, theta, i, bmaj, bmin, bpa, pix_size):
    im = ring_gaussian(x, y, pix_size, amp, r0, x0, y0, fwhm, theta, i)
    convolved = convolve_image(im, bmaj,bmin,bpa,pix_size*glr.arcsec)

    return convolved

def convolve_image(im, bmaj, bmin, bpa, pix_size, xflip=True):
    """Convolve an input image in 2d array im with coords x and y and pixel
    size pix_size with an elliptical Gaussian beam. Beam axes are FWHM and
    assumed to be in the same units as pix_size, position angle is in
    degrees. Calls jypix_to_jybeam()."""
    npxbm = int((bmaj + bmin) / pix_size)
    size = (npxbm + 1) * pix_size

    # more elegant: creates a true symmetric array around 0,0 for the beam
    # xm = np.arange(-1 * size, size + pix_size, pix_size)
    # Logan: np.arange is unpredictable, use linspace and round to
    # nearest odd instead!
    npix_xm = int(2.0*size/pix_size // 2 * 2 + 1)
    xm = np.linspace(-size, size, npix_xm)
    ym = xm
    xxm, yym = np.meshgrid(xm, ym)
    sig2fwhm = 2*np.sqrt(2*np.log(2))  # sigma * this number = fwhm
    beam = elliptical_gaussian(xxm, yym, 1, 0, 0, bmaj / sig2fwhm,bmin / sig2fwhm, bpa, 0.0, xflip)
    import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1,1)
    # plt.imshow(beam, origin='lower')
    # plt.show()
    imconv = convolve(im, beam, boundary='extend', normalize_kernel=True)
    # convert jy/pix to jy/beam by multiplying by pix/beam. Still need to
    # update your FITS header!
    imconv = jypix_to_jybeam(imconv, bmaj, bmin, pix_size)
    return imconv

def elliptical_gaussian(x,y, amplitude, xo, yo, sigma_x, sigma_y, theta,
                        offset, xflip=True):
    """Returns a 2D elliptical Gaussian for beam convolution. Gaussian size
    is in sigma, angle is in radians. RA decreasing with increasing x pixel
    coord is assumed. Called by convolve_image()"""
    # it may be necessary to flip x due to coordinate system of observations.
    # This is the case when this function is used by convolve_fits_2d()
    if xflip:
        x *= -1
    theta = (90. - theta)*(np.pi/180.)
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) \
        + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) \
        + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) \
        + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    gaussian = offset + amplitude * np.exp(- (a * ((x - xo) ** 2)
                                              + 2 * b * (x - xo) * (y - yo)
                                              + c * ((y - yo) ** 2)))
    return gaussian

def jypix_to_jybeam(intensity, bmaj, bmin, pix_size):
    """Converts intensity in Jy/pixel to intensity in Jy/beam, given the beam
    ellipse in FWHM and the pixel size. These are assumed to be in the same
    units (e.g. arcsec) so check this."""
    sig2fwhm = 2*np.sqrt(2*np.log(2))  # sigma * this number = fwhm
    # Volume of 2D Gaussian = Amp*area, area=2*pi*sig_x*sig_y, convert sig to
    # FWHM to arrive at below formula.
    beam_area = 2.0*np.pi*bmaj*bmin/(sig2fwhm**2)
    pix_area = pix_size**2
    pix_per_beam = beam_area / pix_area
    # Apply conversion by multiplying Jy/pix by pix/beam
    return intensity*pix_per_beam

# plot colorbars properly
def colorbar(mappable):
    """Given a mappable object returned by imshow, plot a colorbar on the
    subplot correctly. Code based on
    https://joseph-long.com/writing/colorbars/."""
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

"""

Change dataname and only dataname!!!!

"""
#dataname = 'J1604_uniform-1.fits'
dataname = 'J1608_B6_robust.fits'
#dataname = 'J1608_B7_robust.fits'
#dataname = 'LkCa15_uniform.fits'
#dataname = 'PDS70_robust-1.fits'


fit = fits.open(dataname)
data = fit[0].data[0, 0]
x_shape, y_shape = np.shape(data)

bmaj = fit[0].header['bmaj']* glr.deg #rad
bmin = fit[0].header['bmin']* glr.deg #rad
bpa = fit[0].header['bpa']  #degrees
pix_size=np.abs(fit[0].header['cdelt1'])*3600#arcsec


print(bmaj,bmin,bpa,pix_size)
print(glr.arcsec)
_, _, x_mesh, y_mesh, _ = glr.get_coords_meshgrid(x_shape, x_shape, pix_size*glr.arcsec, origin='upper') #rad



gmodel = Model(convolved_ring_gaussian, independent_vars=['x', 'y'], prefix='gauss_')

if (dataname ==  'PDS70_robust-1.fits'):
    param_list = [
        dict(name='gauss_bmaj', value=bmaj, vary=False),
        dict(name='gauss_bmin', value=bmin, vary=False),
        dict(name='gauss_bpa', value=bpa, vary=False),
        dict(name='gauss_pix_size', value=pix_size, vary=False),
        dict(name='gauss_amp', value=0.165, min=1e-4, vary=True),
        dict(name='gauss_r0', value=0.7, min=1e-3, max=2., vary=True),
        dict(name='gauss_x0', value=0.5, min=-2, max=2, vary=True),
        dict(name='gauss_y0', value=-0.6, min=-2, max=2, vary=True),
        dict(name='gauss_fwhm', value=0.3, min=0.01, max=2, vary=True),
        dict(name='gauss_theta', value=160., min=0.1, max=180., vary=True),
        dict(name='gauss_i', value=10, min=0., max=90., vary=True)
        ]
if (dataname == 'J1604_uniform-1.fits'):
    param_list = [
        dict(name='gauss_bmaj', value=bmaj, vary=False),
        dict(name='gauss_bmin', value=bmin, vary=False),
        dict(name='gauss_bpa', value=bpa, vary=False),
        dict(name='gauss_pix_size', value=pix_size, vary=False),
        dict(name='gauss_amp', value=0.32, min=1e-4, vary=True),
        dict(name='gauss_r0', value=0.6, min=1e-3, max=1., vary=True),
        dict(name='gauss_x0', value=-0.2, min=-1, max=1, vary=True),
        dict(name='gauss_y0', value=0.5, min=-1, max=1, vary=True),
        dict(name='gauss_fwhm', value=0.2, min=0.01, max=2, vary=True),
        dict(name='gauss_theta', value=90., min=0.1, max=180., vary=True),
        dict(name='gauss_i', value=45, min=0., max=90., vary=True)
    ]

if (dataname == 'LkCa15_uniform.fits'):
    param_list = [
        dict(name='gauss_bmaj', value=bmaj, vary=False),
        dict(name='gauss_bmin', value=bmin, vary=False),
        dict(name='gauss_bpa', value=bpa, vary=False),
        dict(name='gauss_pix_size', value=pix_size, vary=False),
        dict(name='gauss_amp', value=0.165, min=1e-4, vary=True),
        dict(name='gauss_r0', value=0.48, min=1e-3, max=1., vary=True),
        dict(name='gauss_x0', value=-0.05, min=-1, max=1, vary=True),
        dict(name='gauss_y0', value=0.01, min=-1, max=1, vary=True),
        dict(name='gauss_fwhm', value=0.5, min=0.01, max=2, vary=True),
        dict(name='gauss_theta', value=61., min=0.1, max=180., vary=True),
        dict(name='gauss_i', value=49, min=0., max=90., vary=True)
    ]

if (dataname == 'J1608_B6_robust.fits'):


    param_list = [
        dict(name='gauss_bmaj', value=bmaj,  vary=False),
        dict(name='gauss_bmin', value=bmin, vary=False),
        dict(name='gauss_bpa', value=bpa, vary=False),
        dict(name='gauss_pix_size', value=pix_size, vary=False),
        dict(name='gauss_amp', value=0.165, min=1e-4, vary=True),
        dict(name='gauss_r0', value=0.48, min=1e-3, max=1., vary=True),
        dict(name='gauss_x0', value=-0.05, min=-1, max=1, vary=True),
        dict(name='gauss_y0', value=0.07, min=-1, max=1, vary=True),
        dict(name='gauss_fwhm', value=0.2, min=0.01, max=2, vary=True),
        dict(name='gauss_theta', value=107., min=0.1, max=180., vary=True),
        dict(name='gauss_i', value=72, min=0., max=90., vary=True)


    ]

if (dataname == 'J1608_B7_robust.fits'):

    param_list = [
        dict(name='gauss_bmaj', value=bmaj, vary=False),
        dict(name='gauss_bmin', value=bmin, vary=False),
        dict(name='gauss_bpa', value=bpa, vary=False),
        dict(name='gauss_pix_size', value=pix_size, vary=False),
        dict(name='gauss_amp', value=0.1, min=1e-3, vary=True),
        dict(name='gauss_r0', value=0.5, min=-1., max=1., vary=True),
        dict(name='gauss_x0', value=0.07, min=-1.2, max=1.2, vary=True),
        dict(name='gauss_y0', value=0.02, min=-1.2, max=1.2, vary=True),
        dict(name='gauss_fwhm', value=0.2, min=0.05, max=1.0, vary=True),
        dict(name='gauss_theta', value=108., min=0.1, max=180., vary=True),
        dict(name='gauss_i', value=72., min=0., max=90., vary=True)

    ]



params = Parameters()
for p in param_list:
    params.add(**p)


print('parameter names: {}'.format(gmodel.param_names))
print('independent variables: {}'.format(gmodel.independent_vars))



result = gmodel.fit(data, params, x = x_mesh, y = y_mesh,method='least_squares')









# H,W = np.shape(result.best_fit)
# F = fft2(result.best_fit)/(W*H)
# F = fftshift(F)
# P=np.abs(F)


print(result.fit_report())




def ploting(dataname, data, result):
    plt.figure(figsize=(9,3))
    plt.suptitle(dataname)
    plt.subplot(1,3,1)
    m =plt.imshow(data, origin="lower", cmap='viridis')
    plt.title("Data")
    colorbar(m)
    plt.subplot(1,3,2)
    m = plt.imshow(result.best_fit, origin= "lower", cmap='viridis')
    plt.title("Model")
    colorbar(m)
    plt.subplot(1,3,3)
    m =plt.imshow(data - result.best_fit, origin="lower", cmap='viridis')
    plt.title("Residual")
    colorbar(m)
    plt.tight_layout()
    plt.savefig(dataname+".jpg")
    plt.show()
    print(dataname)

ploting(dataname = dataname, data = data ,result = result)

# plt.figure(1)
# plt.title(dataname+" Fourier Domain")
# plt.imshow(P, origin="lower")
plt.show()
