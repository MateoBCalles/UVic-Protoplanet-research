import galario.double as glr
import numpy as np
import matplotlib.pyplot as plt
from uvplot import UVTable, COLUMNS_V0
from emcee import EnsembleSampler
import time
import corner
from galario import deg, arcsec

plt.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt

import pickle


def ring_gaussian(x, y, pix_size, amp, r0, x0, y0, fwhm, theta, i):
    # intensity must be in Jy/pix for galario
    amp *= (pix_size ** 2)

    # all dimensions in radians for galario.

    sig2fwhm = 2.35482
    sigma = (fwhm / sig2fwhm) * glr.arcsec

    theta *= np.pi / 180.
    i *= np.pi / 180

    x_new = (-(x - x0) * np.cos(theta) + (y - y0) * np.sin(theta)) / np.cos(i)
    y_new = (x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)

    r = np.sqrt((x_new) ** 2 + (y_new) ** 2)
    gaussian_im = amp * np.exp(-1 / 2 * ((r - r0) / sigma) ** 2)

    return gaussian_im


def ring_gaussian_uv(u, v, nxy, pix_size, amp, x0, y0, r0, fwhm,
                     theta, i):
    _, _, x_m, y_m, _ = glr.get_coords_meshgrid(nxy, nxy, pix_size * glr.arcsec
                                                , origin='upper')
    r0 *= glr.arcsec
    x0 *= glr.arcsec
    y0 *= glr.arcsec
    # call image plane fn
    im = ring_gaussian(x_m, y_m, pix_size, amp, r0, x0, y0, fwhm, theta, i)

    # compute visibilities and return them
    gaussian_vis = glr.sampleImage(im, pix_size * glr.arcsec, u, v, origin='upper')
    return gaussian_vis


def get_data_uv(vis_file):
    """returns visibility data in vis_file created by UVTABLE in the order
    u(m), v(m), real (Jy), imag (Jy), weight."""
    return np.require(np.loadtxt(vis_file, unpack=True), requirements='C')


def lnpriorfn(p, par_ranges):
    """ Uniform prior probability function """

    ln_prior = 0.0
    for p, p_bounds in zip(p, par_ranges):
        if p_bounds[0] < p < p_bounds[1]:
            ln_prior += np.log(1.0 / (p_bounds[1] - p_bounds[0]))
        else:
            ln_prior += -np.inf
    return ln_prior


def lnpostfn(p, p_ranges, x_m, y_m, dxy, u, v, Re, Im, w):
    """ Log of posterior probability function """

    lnprior = lnpriorfn(p, p_ranges)

    chi2 = 0.0
    # Speedup: no need to compute chi2 if the prior probability is zero!
    if lnprior != -np.inf:
        # unpack the parameters
        amp, r0, dRA, dDec, fwhm, PA, INC = p
        amp = 10. ** amp
        dRA *= arcsec
        dDec *= arcsec
        r0 *= arcsec

        im = ring_gaussian(x_m, y_m, dxy, amp, r0, dRA, dDec, fwhm, PA, INC)

        # convert to radian

        chi2 = glr.chi2Image(im, dxy * glr.arcsec, u, v, Re, Im, w, check=True, origin='upper')

    return -0.5 * chi2 + lnprior


filenames = 'J1604_B6_robust_-1.0'
dataname = filenames + '.txt'
f = open(dataname)
obs_wl = float(f.read().split(('\n'))[1].split(' ')[
                   -1])  # reads the second line of the uvplot.txt file which has the wavelenght in it

f.close()
obs_u, obs_v, obs_vis_re, obs_vis_im, obs_vis_w = get_data_uv(dataname)

obs_u /= obs_wl
obs_v /= obs_wl
obs_vis = obs_vis_re + obs_vis_im * 1.0j

nxy, dxy = glr.get_image_size(obs_u, obs_v)
_, _, x_m, y_m, _ = glr.get_coords_meshgrid(nxy, nxy, dxy, origin='upper')

# parameter space domain
# amp, r0, x0, y0, fwhm, PA, inc
# 0.001 to 0.3
p_ranges = [[-3, -0.1],
            [0.01, 1.],
            [-1.0, 1.0],
            [-1.0, 1.0],
            [0.01, 1],
            [0., 180.],
            [0., 90.]]
ndim = len(p_ranges)  # number of dimensions
nwalkers = 24  # number of walkers

nthreads = 4  # CPU threads that emcee should use

sampler = EnsembleSampler(nwalkers, ndim, lnpostfn,
                          args=[p_ranges, x_m, y_m, dxy / glr.arcsec, obs_u, obs_v, obs_vis_re, obs_vis_im,
                                obs_vis_w],
                          threads=nthreads)

nsteps = 10000  # total number of MCMC steps

# initial guess for the parameters
#     amp,      r0,    x0,    y0,     fwhm,  PA,    inc
p0 = [-1.175478207566941, 0.60537062, -0.01825170, 0.02556048, 0.25523626, 97.1469632, 14.4801880]
bounds_range = (np.array(p_ranges)[:, 1] - np.array(p_ranges)[:, 0])

pos = [p0 + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

state, prob = (None, None)

pos, prob, state = sampler.run_mcmc(pos, nsteps, rstate0=state, log_prob0=prob)
samples = sampler.chain[:, -1000:, :].reshape((-1, ndim))

# save the chain to disk
pickle.dump(sampler.chain, open(filenames + 'chain.p', 'wb'))
# save the walker positions, posterior value, and sampler state so you can restart the sampler later where you left off:
mcmc_state_stuff = (pos, prob, state)
pickle.dump(mcmc_state_stuff, open(filenames + 'mcmc_state_stuff.p', 'wb'))

fig = corner.corner(samples, labels=["amp", "$r_0$", "$x_0$", "$y_0$", "fwhm", "PA", "inc"],
                    show_titles=True, quantiles=[0.16, 0.50, 0.84],
                    label_kwargs={'labelpad': 20, 'fontsize': 0}, fontsize=3)


fig.set_size_inches(24, 24)
fig.suptitle(dataname)
fig.savefig(filenames + "cornerplot.png")

bestfit = [np.percentile(samples[:, i], 50) for i in range(ndim)]

# reads the second line of the uvplot.txt file which has the wavelenght in it


amp, r0, x0, y0, fwhm, PA, inc = bestfit
print(10. ** amp, r0, x0, y0, fwhm, PA, inc)
uvbin_size = 3e4  # uv-distance bin, units: wle

amp = 10. ** amp
# convert to radians
dRA = x0 * glr.arcsec
dDec = y0 * glr.arcsec
PA *= glr.deg
inc *= glr.deg

uv_obs = UVTable(uvtable=[obs_u * obs_wl, obs_v * obs_wl, obs_vis_re, obs_vis_im, obs_vis_w], wle=obs_wl,
                 columns=COLUMNS_V0)
uv_obs.apply_phase(-dRA, -dDec)  # center the source on the phase center
uv_obs.deproject(inc, PA)
axes = uv_obs.plot(linestyle='.', color='k', label='Data', uvbin_size=uvbin_size)
#
# # model uv-plot

vis_mod = ring_gaussian_uv(obs_u, obs_v, nxy, dxy / glr.arcsec, amp, x0, y0, r0, fwhm, PA / glr.deg, inc / glr.deg)
uv_mod = UVTable(uvtable=[obs_u * obs_wl, obs_v * obs_wl, vis_mod.real, vis_mod.imag, obs_vis_w], wle=obs_wl,
                 columns=COLUMNS_V0)
uv_mod.apply_phase(-dRA, -dDec)  # center the source on the phase center
uv_mod.deproject(inc, PA)
uv_mod.plot(axes=axes, linestyle='-', color='r', label='Model', yerr=False, uvbin_size=uvbin_size)
axes[0].figure.suptitle(dataname)
axes[0].figure.set_size_inches(12, 12)
axes[0].figure.savefig(filenames + "Deprojection.png")
