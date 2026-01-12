# Code for trying to produce an appropriate model for the eRASS selection function
# Created: 2026-01-06, by: Joseph Hall

import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
import math

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.integrate import dblquad
from scipy.special import factorial
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
    x_part = 1 / (1 + np.exp(A*(B-x)))
    y_part = 1 / (1 + np.exp(C*(D-y)))
    return x_part * y_part


def model_counts(axes, theta: tuple, **kwargs):
    schechter_pred = kwargs['schechter_pred']
    sigmoid_vals = np.zeros_like(schechter_pred)
    for zind, z in enumerate(axes[0]):
        for lind, l in enumerate(axes[1]):
            sigmoid_vals[zind][lind] = sigmoid(z, np.log10(l), *theta)
    
    # plt.imshow(sigmoid_vals.T, extent=(min(axes[0]), max(axes[0]), np.log10(min(axes[1])), np.log10(max(axes[1]))), aspect="auto", origin="lower")
    # plt.colorbar()
    # plt.show()
    # 
    # plt.imshow(sigmoid_vals.T*schechter_pred.T, extent=(min(axes[0]), max(axes[0]), np.log10(min(axes[1])), np.log10(max(axes[1]))), aspect="auto", origin="lower")
    # plt.colorbar()
    # plt.show()
    return schechter_pred * sigmoid_vals


def calculate_log_factorials(data):
    """
    Calculate the factorials of the data and return the log of the factorials
    :param data: The data to be fit
    :return: The factorials of the data
    """
    log_factorials = np.log(factorial(data))  # Calculate the factorials of the data using scipy
    # Check if any are infinite
    if np.any(np.isinf(log_factorials)):
        try:
            # Try to calculate the factorials using a for loop and math.factorial
            log_factorials = np.log(np.array([float(math.factorial(int(d))) for d in data]))
        except:
            # If that fails, use Stirling's approximation to calculate the factorials
            log_factorials = data * np.log(data) - data + 0.5 * np.log(2 * np.pi * data)

    return log_factorials


def lnprob(theta, axes, data, model_func, **kwargs):
    kwargs_factorials = kwargs.get('data_log_factorials')
    data_log_factorials = kwargs_factorials if kwargs_factorials is not None else calculate_log_factorials(data)
    model = model_func(axes, theta, **kwargs)

    # plt.imshow((data-model).T, origin="lower", extent=(min(axes[0]), max(axes[0]), np.log10(min(axes[1])), np.log10(max(axes[1]))), aspect="auto")
    # plt.colorbar()
    # plt.show()

    return np.sum(-model + data * np.log(model) - data_log_factorials)


def uniform_nonzero_prior(theta):
    #if any(np.array(theta) < 0):
    #    return -np.inf
    A_prior = (-150 <= theta[0]) & (theta[0] <= 0)
    B_prior = (0 < theta[1]) & (theta[1] < 1)
    C_prior = (0 < theta[2]) & (theta[2] <= 20)
    D_prior = (42 < theta[3]) & (theta[3] < 45)
    if all((A_prior, B_prior, C_prior, D_prior)):
        return 0
    else:
        return -np.inf


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


def fit2d(mid_points, histo2d, schechter_grid, z_bins, lumin_bins):
    # Do some emcee to constrain sigmoid
    initial_theta = (-100, 0.2, 5, 43)
    ndim, nwalkers = 4, 100
    pos = np.array(initial_theta) + 1e-4 * np.random.randn(nwalkers, ndim)

    print(log_probability(initial_theta, mid_points, histo2d[0], model_counts, schechter_pred=schechter_grid, 
                                            data_log_factorials=calculate_log_factorials(histo2d[0])))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(mid_points, histo2d[0], model_counts),
                                    kwargs={"schechter_pred": schechter_grid, 
                                            "data_log_factorials": calculate_log_factorials(histo2d[0])})
    sampler.run_mcmc(pos, 5000, progress=True)

    # Get the samples and do a corner plot
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)[:] # Using 100 steps as burn in and thinning by 15
    fig = corner.corner(flat_samples, labels=["A", "B", "C", "D"], show_titles=True)
    plt.show()

    # Plot the chains
    fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["A", "B", "C", "D"]
    for aid, ax in enumerate(axes):
        ax.plot(samples[:, :, aid], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[aid])
    axes[-1].set_xlabel("step number")
    plt.show()

    # Plot median chain values
    #samples_T = samples.T
    pred_from_emcee = model_counts(mid_points, np.median(flat_samples.T, axis=1), schechter_pred=schechter_grid)

    plt.imshow((pred_from_emcee/schechter_grid).T, extent=[z_bins[0], z_bins[-1], np.log10(lumin_bins[0]), np.log10(lumin_bins[-1])], 
               aspect="auto", origin="lower")
    plt.ylabel("log(L_500)")
    plt.xlabel("Redshift")
    plt.colorbar(label="$\sigma(L, z)$ fraction from sigmoid")
    plt.show()

    plt.imshow(pred_from_emcee.T, extent=[z_bins[0], z_bins[-1], np.log10(lumin_bins[0]), np.log10(lumin_bins[-1])], 
               aspect="auto", origin="lower")
    plt.ylabel("log(L_500)")
    plt.xlabel("Redshift")
    plt.colorbar(label="N(L, z) from Schechter and Sigmoid")
    plt.show()

    plt.imshow(pred_from_emcee.T - histo2d[0].T, extent=[z_bins[0], z_bins[-1], np.log10(lumin_bins[0]), np.log10(lumin_bins[-1])], 
               aspect="auto", origin="lower")
    plt.ylabel("log(L_500)")
    plt.xlabel("Redshift")
    plt.colorbar(label="N clust exp - N clust obs")
    plt.show()
    return

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

    # Get midpoints
    mid_points = [(z_bins[1:] + z_bins[:-1])/2, (lumin_bins[1:] + lumin_bins[:-1])/2]

    # 2D fit
    fit2d(mid_points, histo2d, schechter_grid, z_bins, lumin_bins)

    # 1D fit
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
