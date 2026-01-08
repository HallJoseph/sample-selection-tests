# Code for trying to produce an appropriate model for the eRASS selection function
# Created: 2026-01-06, by: Joseph Hall

import matplotlib.pyplot as plt
import numpy as np

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.integrate import dblquad
from load_catalogue import load_catalogue


H0 = 70
COSMO = FlatLambdaCDM(H0, 0.3)
LITLLE_H = H0/100


def xlf_schechter(lx, phi_star, lx_star, alpha):
    return (phi_star * ((lx/lx_star)**-alpha) * np.exp(-lx/lx_star) * 1/lx_star)


def integrand(lx, z, phi_star, lx_star, alpha):
    # Get comoving volume
    comovol = COSMO.differential_comoving_volume(z).value

    # Get Schechter func value
    phi_l = xlf_schechter(lx, phi_star, lx_star, alpha)

    # Return integrand
    return phi_l * comovol


def sigmoid(x, y, A, B, C, D):
    x_part = 1 / (1 + np.exp(A*x + B))
    y_part = 1 / (1 + np.exp(C*y + D))
    return x_part * y_part


def model_counts(axes, theta: tuple, schechter_pred):
    sigmoid_vals = np.zeros_like(schechter_pred)
    for zind, z in axes[0]:
        for lind, l in enumerate(axes[1]):
            sigmoid_vals[zind][lind] = sigmoid(z, l, *theta)

    return schechter_pred * sigmoid_vals


def lnprob(theta, axes, data, model_func, **kwargs):
    model = model_func(axes, theta, **kwargs)
    kwargs_factorials = kwargs.get('data_log_factorials')
    data_log_factorials = kwargs_factorials if kwargs_factorials is not None else calculate_log_factorials(data)

    return np.sum(-model + data * np.log(model) - data_log_factorials)


def uniform_nonzero_prior(theta):
    if any(np.array(theta) < 0):
        return -np.inf
    return 0

def log_probability(theta, x, data, model_func, prior_func=uniform_nonzero_prior, **kwargs):
    """
    Log probability function for the MCMC
    :param theta: Model parameters
    :param x: The x-axis of the data to evaluate the model on
    :param data: Observed data
    :param model_func: The model function to be used
    :param prior_func: Function for the priors of the model
    :return:
    """
    lp = prior_func(theta)
    # print("prior", lp)
    if not np.isfinite(lp):
        return -np.inf
    prob = lp + lnprob(theta, x, data, model_func, **kwargs)
    if np.isnan(prob):
        return -np.inf
    return prob


def main(sample_path="data/emain_wen-han_final_20250328_1052", sample_area=1.1085567827):
    # Using WARPS/REFLEX XLF 
    phi_star = 2.94e-7 # * (u.Mpc ** -3)
    l_star = 2.64e44 # * u.erg / u.second
    alpha = 1.69
    lumins = np.logspace(42, 45.3)# *u.erg/u.second

    # Load in the sample catalogue and set up histogram grid
    emain, wh = load_catalogue(sample_path)

    # Constrain z range of emain to remove "fuzz"
    emain = emain[(emain["BEST_Z_1"] <= 0.2) & (emain["BEST_Z_1"] >= 0.1)]
    histo2d = np.histogram2d(emain["BEST_Z_1"], np.log10(emain["L500_1"])+42)
    z_bins = histo2d[1]
    lumin_bins = 10 ** histo2d[2] # * u.erg/u.second

    # Set up grid of expected values from Schechter function for these bins
    schechter_grid = np.zeros_like(histo2d[0])
    for zind, z_lo in enumerate(z_bins[:-1]):
        z_hi = z_bins[zind+1]
        for lind, l_lo in enumerate(lumin_bins[:-1]):
            l_hi = lumin_bins[lind+1]

            # Evaluate schechter function for this bin
            schechter_pred = dblquad(integrand, z_lo, z_hi, l_lo, l_hi, args=(phi_star, l_star, alpha))# * 1/u.sr
            # print(z_lo, l_lo, schechter_pred)
            schechter_grid[zind][lind] = schechter_pred[0]
    
    # Convert grid from clusters per sr to just clusters:
    schechter_grid *= sample_area

    # Do some emcee to constrain sigmoid

    plt.imshow(schechter_grid.T - histo2d[0].T, extent=[z_bins[0], z_bins[-1], np.log10(lumin_bins[0]), np.log10(lumin_bins[-1])], 
               aspect="auto", origin="lower")
    plt.ylabel("log(L_500)")
    plt.xlabel("Redshift")
    plt.colorbar(label="N clust exp - N clust obs")
    plt.show()
    return

    plt.ylabel("log(L_500) (From eRASS catalogue)")
    plt.xlabel("Redshift")
    plt.colorbar()
    plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(lumins, phi)
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_xlabel("Luminosity [erg s^-1]")
    # ax.set_ylabel("$\phi$ [Mpc^-3 (erg s^-1)^-1]")
    # plt.show()


if __name__ == "__main__":
    main()
    pass
