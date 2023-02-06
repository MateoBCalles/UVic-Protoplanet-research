from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from numpy.fft import fft, fft2, fftshift
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling import models, fitting
from mpl_toolkits.axes_grid1 import make_axes_locatable



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





fit = fits.open('J1604_uniform-1.fits')
#fit = fits.open('LkCa15_uniform.fits')
#fit = fits.open('PDS70_robust-1.fits')

"""
to ensure the right initial parameters are being used for lmfit:
Scroll down to the variable: results
and comment out/in the appropiate fit
"""


data = fit[0].data[0, 0]



x_shape, y_shape = np.shape(data)

x_mesh , y_mesh = np.meshgrid(range(x_shape),range(y_shape))



"""
Using lmfit
"""
def gaussian_ring_1(x,y,amp=0.05, r0 = 30 , x0= 160 , y0 = 120 , fwhm = 10, theta = 0, i= 45):
    sig2fwhm = 2.35482
    sigma = (fwhm / sig2fwhm)
    theta *= np.pi / 180.
    i *= np.pi/180
    x_new = (-(x-x0)*np.cos(theta) + (y-y0)*np.sin(theta))/np.cos(i)
    y_new = (x-x0)*np.sin(theta) + (y-y0)*np.cos(theta)


    r = np.sqrt((x_new) ** 2 + (y_new) ** 2)
    return amp*np.exp(-1/2*((r - r0)/sigma)**2)


gmodel = Model(gaussian_ring_1,independent_vars=['x', 'y'])



print('parameter names: {}'.format(gmodel.param_names))
print('independent variables: {}'.format(gmodel.independent_vars))



"""
        PDS70
"""
#result = gmodel.fit(data, x = x_mesh, y = y_mesh, amp=data.max(), r0 = 100, x0= 700, y0 = 500, fwhm = 50, i = 20)

"""
        J1604 & LkCa15
"""

result = gmodel.fit(data, x = x_mesh, y = y_mesh, amp=data.max(), r0 = 30, x0= 160, y0 = 120, fwhm = 10, theta= 80, i = 1)



H,W = np.shape(result.best_fit)
F = fft2(result.best_fit)/(W*H)
F = fftshift(F)
P=np.abs(F)


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
    plt.show()
    plt.savefig(dataname+".jpg")
    print(dataname)

ploting(dataname = "LkCal15",data = data ,result = result)

plt.figure(1)
plt.imshow(P)
plt.show()
