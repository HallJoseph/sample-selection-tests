# Code for trying to produce an appropriate model for the eRASS selection function
# Created: 2026-01-06, by: Joseph Hall

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import emcee
import corner
import math
import tqdm
import pandas as pd

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from scipy.integrate import dblquad
from scipy.special import factorial
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.stats import chi

from load_catalogue import load_catalogue

new_rc_params = {
    'text.usetex': False,
    "svg.fonttype": 'none',
    "font.family": 'helvetica',
    "font.size": 16,
    "mathtext.fontset": 'custom',
    "mathtext.rm": "helvetica",
    "mathtext.it": "helvetica"
}
mpl.rcParams.update(new_rc_params)

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


def sigmoid_1D(x, A, B):
    return 1 / (1 + np.exp(A*(B-x)))


def sigmoid_2d(x, y, A, B, C, D):
    x_part = sigmoid_1D(x, A, B)
    y_part = sigmoid_1D(y, C, D)

    #print(x_part.shape)
    x_arr = np.array([[x] for x in x_part])
    if len(x_arr.shape) == 3:
        x_arr = x_arr.transpose(0, 2, 1)
        #print(x_arr.shape)

        y_arr = np.array([[y] for y in y_part])
    else:
        y_arr = np.array([y_part])
    #print(y_arr.shape)
    #print((x_arr @ y_arr).shape)
    return x_arr @ y_arr


def sigmoid(x, y, A, B, C, D):
    x_part = sigmoid_1D(x, A, B)
    y_part = sigmoid_1D(y, C, D)
    return x_part * y_part


def model_counts(axes, theta: tuple, do_plot=False, **kwargs):
    schechter_pred = kwargs['schechter_pred']
    sigmoid_vals = np.zeros_like(schechter_pred)
    # for zind, z in enumerate(axes[0]):
    #     for lind, l in enumerate(axes[1]):
    #         sigmoid_vals[zind][lind] = sigmoid(z, np.log10(l), *theta)

    sigmoid_vals = sigmoid_2d(axes[0], axes[1], *theta)
    
    # plt.imshow(sigmoid_vals.T, extent=(min(axes[0]), max(axes[0]), np.log10(min(axes[1])), np.log10(max(axes[1]))), aspect="auto", origin="lower")
    # plt.colorbar()
    # plt.show()
    # 
    # plt.imshow(sigmoid_vals.T*schechter_pred.T, extent=(min(axes[0]), max(axes[0]), np.log10(min(axes[1])), np.log10(max(axes[1]))), aspect="auto", origin="lower")
    # plt.colorbar()
    # plt.show()
    if do_plot:
        plt.imshow(sigmoid_vals)
        plt.show()
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
    

def uniform_1d_prior(theta):
    #if any(np.array(theta) < 0):
    #    return -np.inf
    A_prior = (15 >= theta[0]) & (theta[0] >= 1)  # -150 <= theta[0]) & 
    B_prior = (35 < theta[1]) & (theta[1] < 39)
    if all((A_prior, B_prior)):
        return 0
    else:
        return -np.inf


def log_probability(theta, x, data, model_func, prior_func=uniform_1d_prior, **kwargs):
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


def fit2d(mid_points, histo2d, schechter_grid, z_bins, lumin_bins, sample_df=None):
    print(histo2d[1])
    # Do some emcee to constrain sigmoid
    initial_theta = (-100, 0.2, 5, 43)
    ndim, nwalkers = 4, 100
    pos = np.array(initial_theta) + 1e-4 * np.random.randn(nwalkers, ndim)

    mid_points[1] = np.log10(mid_points[1])

    print(log_probability(initial_theta, mid_points, histo2d[0], model_counts, schechter_pred=schechter_grid, 
                                            data_log_factorials=calculate_log_factorials(histo2d[0])))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(mid_points, histo2d[0], model_counts, uniform_nonzero_prior),
                                    kwargs={"schechter_pred": schechter_grid, 
                                            "data_log_factorials": calculate_log_factorials(histo2d[0])})
    sampler.run_mcmc(pos, 1000, progress=True)

    ## Get the samples and do a corner plot
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)[:] # Using 100 steps as burn in and thinning by 15
    # fig = corner.corner(flat_samples, labels=["A", "B", "C", "D"], show_titles=False)
    # plt.show()
#
    ## Plot the chains
    #fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
    #samples = sampler.get_chain()
    #labels = ["A", "B", "C", "D"]
    #for aid, ax in enumerate(axes):
    #    ax.plot(samples[:, :, aid], "k", alpha=0.3)
    #    ax.set_xlim(0, len(samples))
    #    ax.set_ylabel(labels[aid])
    #axes[-1].set_xlabel("step number")
    #plt.show()

    # Plot median chain values
    #samples_T = samples.T
    pred_from_emcee = model_counts(mid_points, np.median(flat_samples.T, axis=1), schechter_pred=schechter_grid)

    plt.imshow((pred_from_emcee/schechter_grid).T, extent=[z_bins[0], z_bins[-1], np.log10(lumin_bins[0]), np.log10(lumin_bins[-1])], 
               aspect="auto", origin="lower")
    plt.ylabel("$\log(L_{500})$")
    plt.xlabel("Redshift")
    plt.colorbar(label="$\sigma(L, z)$")
    plt.tight_layout()
    plt.show()
# 
    # plt.imshow(pred_from_emcee.T, extent=[z_bins[0], z_bins[-1], np.log10(lumin_bins[0]), np.log10(lumin_bins[-1])], 
    #            aspect="auto", origin="lower")
    # plt.ylabel("log(L_500)")
    # plt.xlabel("Redshift")
    # plt.colorbar(label="N(L, z) from Schechter and Sigmoid")
    # plt.show()
# 
    # plt.imshow(pred_from_emcee.T - histo2d[0].T, extent=[z_bins[0], z_bins[-1], np.log10(lumin_bins[0]), np.log10(lumin_bins[-1])], 
    #            aspect="auto", origin="lower")
    # plt.ylabel("log(L_500)")
    # plt.xlabel("Redshift")
    # plt.colorbar(label="N clust exp - N clust obs")
    # plt.show()

    # Optionally perform goodness of fit check
    #plt.imshow(histo2d[0])
    #plt.show()
    #plt.imshow(pred_from_emcee)
    #plt.show()
    if sample_df is not None:
        # Test goodness of fit (posterior predictive p approach)
        median_res = np.median(flat_samples.T, axis=1)
        sel_func = sigmoid_2d(mid_points[0], mid_points[1], median_res[0], median_res[1], median_res[2], median_res[3])
        data_chi_sq = np.sum(((histo2d[0]-pred_from_emcee)**2) / pred_from_emcee)
        
        chi_list = []
        theta_tests = flat_samples[np.random.choice(range(len(flat_samples)), 5000, False)]
        samp_ids = sample_df["sample"].drop_duplicates().values
        samp_tests = flat_samples[np.random.choice(len(samp_ids), 5000, False)]
        print(mid_points[1][0])
        print()
        print("evaluating tests")
        sel_func_tests = sigmoid_2d(
            np.array([mid_points[0]]*len(theta_tests)),
            np.array([mid_points[1]]*len(theta_tests)),
            theta_tests[:, 0][:, None], 
            theta_tests[:, 1][:, None],
            theta_tests[:, 2][:, None],
            theta_tests[:, 3][:, None]
        )
        print(sel_func_tests[0])
        print(schechter_grid.shape)
        pred_counts_test = schechter_grid * sel_func_tests

        print(pred_counts_test.shape)
        for samp_id in tqdm.tqdm(samp_ids):
            samp_clusts = sample_df[sample_df["sample"]==samp_id].copy()
            samp_hist = np.histogram2d(samp_clusts['z'], samp_clusts["log_l"], bins=[histo2d[1], histo2d[2]])

            samp_hist_sel = samp_hist[0] * sel_func_tests

            samp_chis = np.sum(((samp_hist_sel-pred_counts_test)**2) / pred_counts_test, axis=(1,2))

            #print((((samp_hist_sel-pred_counts_test)**2) / pred_counts_test).shape)
            if any(samp_chis > 1e4):
                samp_clusts.to_csv("chi_too_big.csv")
                continue
            
            chi_list += list(samp_chis)

        pct = (np.sum(np.array(chi_list) > data_chi_sq) / len(chi_list)) * 100
        print(pct)
        plt.hist(chi_list, bins=25, density=True)
        plt.vlines(data_chi_sq, 0, 0.05, color='red', label=f"data chi < {pct:.2f}%")
        plt.xlabel("$\chi ^2$")
        plt.legend()
        plt.show()

    return


def fit2d_alt_ppc(mid_points, histo2d, schechter_grid, z_bins, lumin_bins, sample_df=None):
    print(histo2d[1], "OI")
    # Do some emcee to constrain sigmoid
    initial_theta = (-100, 0.2, 5, 43)
    ndim, nwalkers = 4, 100
    pos = np.array(initial_theta) + 1e-4 * np.random.randn(nwalkers, ndim)

    mid_points[1] = np.log10(mid_points[1])

    print(log_probability(initial_theta, mid_points, histo2d[0], model_counts, schechter_pred=schechter_grid, 
                                            data_log_factorials=calculate_log_factorials(histo2d[0])))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(mid_points, histo2d[0], model_counts, uniform_nonzero_prior),
                                    kwargs={"schechter_pred": schechter_grid, 
                                            "data_log_factorials": calculate_log_factorials(histo2d[0])})
    sampler.run_mcmc(pos, 1000, progress=True)

    ## Get the samples and do a corner plot
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)[:] # Using 100 steps as burn in and thinning by 15
    #fig = corner.corner(flat_samples, labels=["A", "B", "C", "D"], show_titles=True)
    #plt.show()
#
    ## Plot the chains
    #fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
    #samples = sampler.get_chain()
    #labels = ["A", "B", "C", "D"]
    #for aid, ax in enumerate(axes):
    #    ax.plot(samples[:, :, aid], "k", alpha=0.3)
    #    ax.set_xlim(0, len(samples))
    #    ax.set_ylabel(labels[aid])
    #axes[-1].set_xlabel("step number")
    #plt.show()

    # Plot median chain values
    #samples_T = samples.T
    pred_from_emcee = model_counts(mid_points, np.median(flat_samples.T, axis=1), schechter_pred=schechter_grid)

    # plt.imshow((pred_from_emcee/schechter_grid).T, extent=[z_bins[0], z_bins[-1], np.log10(lumin_bins[0]), np.log10(lumin_bins[-1])], 
    #            aspect="auto", origin="lower")
    # plt.ylabel("log(L_500)")
    # plt.xlabel("Redshift")
    # plt.colorbar(label="$\sigma(L, z)$ fraction from sigmoid")
    # plt.show()
# 
    # plt.imshow(pred_from_emcee.T, extent=[z_bins[0], z_bins[-1], np.log10(lumin_bins[0]), np.log10(lumin_bins[-1])], 
    #            aspect="auto", origin="lower")
    # plt.ylabel("log(L_500)")
    # plt.xlabel("Redshift")
    # plt.colorbar(label="N(L, z) from Schechter and Sigmoid")
    # plt.show()
# 
    # plt.imshow(pred_from_emcee.T - histo2d[0].T, extent=[z_bins[0], z_bins[-1], np.log10(lumin_bins[0]), np.log10(lumin_bins[-1])], 
    #            aspect="auto", origin="lower")
    # plt.ylabel("log(L_500)")
    # plt.xlabel("Redshift")
    # plt.colorbar(label="N clust exp - N clust obs")
    # plt.show()

    # Optionally perform goodness of fit check
    #plt.imshow(histo2d[0])
    #plt.show()
    #plt.imshow(pred_from_emcee)
    #plt.show()
    if sample_df is not None:
        # Test goodness of fit (posterior predictive p approach)
        median_res = np.median(flat_samples.T, axis=1)
        sel_func = sigmoid_2d(mid_points[0], mid_points[1], median_res[0], median_res[1], median_res[2], median_res[3])
        data_chi_sq = np.sum(((histo2d[0]-pred_from_emcee)**2) / pred_from_emcee)
        
        chi_list = []
        theta_tests = flat_samples[np.random.choice(range(len(flat_samples)), 5000, False)]
        samp_ids = sample_df["sample"].drop_duplicates().values
        samp_tests = flat_samples[np.random.choice(len(samp_ids), 5000, False)]
        print(mid_points[1][0])
        print()
        print("evaluating tests")
        sel_func_tests = sigmoid_2d(
            np.array([mid_points[0]]*len(theta_tests)),
            np.array([mid_points[1]]*len(theta_tests)),
            theta_tests[:, 0][:, None], 
            theta_tests[:, 1][:, None],
            theta_tests[:, 2][:, None],
            theta_tests[:, 3][:, None]
        )
        print(sel_func_tests[0])
        print(schechter_grid.shape)
        pred_counts_test = schechter_grid * sel_func_tests

        print(pred_counts_test.shape)

        # Generate y_reps based on a poisson distribution with means from pred_counts_test
        y_reps = np.random.poisson(pred_counts_test)
        
        # test stat for y rep (sum of clusters found)
        test_y_rep = np.sum(y_reps, axis=(1,2))
        # test stat for y
        test_y = np.sum(histo2d[0])
        p_val = 100 * sum(test_y_rep > test_y) / len(test_y_rep)
        #plt.hist(test_y_rep, 25, label=r"$y^\text{rep}$")
        #plt.vlines(test_y, 0, 600, color="red", label=f"Observed\np = {p_val:.2f}")
        #plt.legend()
        #plt.xlabel("Number of Clusters")
        #plt.show()

        # Test stat with chi square instead
        chi_y_rep = np.sum(((y_reps-pred_counts_test)**2) / pred_counts_test, axis=(1,2))
        chi_y_obs_theta = np.sum(((histo2d[0]-pred_counts_test)**2) / pred_counts_test, axis=(1,2))
        chi_p_val = 100 * sum(chi_y_rep > chi_y_obs_theta) / len(chi_y_rep)

        plt.scatter(chi_y_obs_theta, chi_y_rep, s=1)
        plt.plot(np.linspace(min(chi_y_rep), max(chi_y_rep), 2), np.linspace(min(chi_y_rep), max(chi_y_rep), 2), color="red")
        plt.gca().set_aspect(1)
        plt.title(f"p={chi_p_val}")
        plt.xlabel(r"$\chi^2(y, \theta)")
        plt.ylabel(r"$\chi^2(y^\text{rep}, \theta)")
        plt.show()

        return
        for samp_id in tqdm.tqdm(samp_ids):
            samp_clusts = sample_df[sample_df["sample"]==samp_id].copy()
            samp_hist = np.histogram2d(samp_clusts['z'], samp_clusts["log_l"], bins=[histo2d[1], histo2d[2]])

            samp_hist_sel = samp_hist[0] * sel_func_tests

            samp_chis = np.sum(((samp_hist_sel-pred_counts_test)**2) / pred_counts_test, axis=(1,2))

            #print((((samp_hist_sel-pred_counts_test)**2) / pred_counts_test).shape)
            if any(samp_chis > 1e4):
                samp_clusts.to_csv("chi_too_big.csv")
                continue
            
            chi_list += list(samp_chis)

        pct = (np.sum(np.array(chi_list) > data_chi_sq) / len(chi_list)) * 100
        print(pct)
        plt.hist(chi_list, bins=25, density=True)
        plt.vlines(data_chi_sq, 0, 0.05, color='red', label=f"data chi < {pct:.2f}%")
        plt.xlabel("$\chi ^2$")
        plt.legend()
        plt.show()

    return


def calc_fluxes(lz_grid, axes):
    # Returns fluxes of an lz grid in units of erg / s / Mpc^2
    flux_ax, counts = [], []
    for zind, z in enumerate(axes[0]):
        for lind, l in enumerate(axes[1]):
            dist = COSMO.luminosity_distance(z) # .to(u.)
            flux = (l / (4*np.pi*dist**2)).value
            flux_ax.append(flux)
            counts.append(lz_grid[zind][lind])
    
    return (flux_ax, counts)


def model_counts_1d(flux_ax, theta, **kwargs):
    schechter_pred = kwargs['schechter_pred']
    sigmoid_vals = sigmoid_1D(flux_ax, *theta)
    #plt.scatter(np.log10(flux_ax), sigmoid_vals)
    #plt.show()
    return schechter_pred * sigmoid_vals


def fit_1d_grid(schechter_grid, histo2d, mid_points):
    schechter_fluxes = calc_fluxes(schechter_grid, mid_points)
    obs_fluxes = calc_fluxes(histo2d[0], mid_points)

    flux_ax = obs_fluxes[0]

    initial_theta = (10, 37)
    ndim, nwalkers = 2, 100
    pos = np.array(initial_theta) + 1e-4 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(flux_ax, obs_fluxes[1], model_counts_1d),
                                    kwargs={"schechter_pred": schechter_fluxes[1], 
                                            "data_log_factorials": calculate_log_factorials(obs_fluxes[1])})
    sampler.run_mcmc(pos, 5000, progress=True)

    # Get the samples and do a corner plot
    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)[:] # Using 100 steps as burn in and thinning by 15
    fig = corner.corner(flat_samples, labels=["A", "B"], show_titles=False)
    plt.show()

    # Plot the chains
    fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["A", "B"]
    for aid, ax in enumerate(axes):
        ax.plot(samples[:, :, aid], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[aid])
    axes[-1].set_xlabel("step number")
    plt.show()
    
    pred_from_emcee = model_counts_1d(flux_ax, 
                                      (5.9, 35.75), 
                                      schechter_pred=schechter_fluxes[1])
    plt.scatter(*schechter_fluxes, label="Schechter Function")
    plt.scatter(*obs_fluxes, label="Observed")
    plt.scatter(flux_ax, pred_from_emcee)
    plt.xscale("log")
    # plt.yscale("log")
    plt.xlabel("log(flux (erg/s/Mpc^2))")
    plt.ylabel("count")
    plt.legend()
    plt.show()


def evalaute_schechter_lz(base_grid, z_bins, lumin_bins, sample_area=1.1085567827):
    # Using WARPS/REFLEX XLF 
    phi_star = 2.94e-7 # * (u.Mpc ** -3)
    l_star = 2.64e44 # * u.erg / u.second
    alpha = 1.69

    # Set up grid of expected values from Schechter function for these bins
    schechter_grid = np.zeros_like(base_grid)
    for zind, z_lo in enumerate(tqdm.tqdm(z_bins[:-1])):
        z_hi = z_bins[zind+1]
        for lind, l_lo in enumerate(lumin_bins[:-1]):
            l_hi = lumin_bins[lind+1]

            # Evaluate schechter function for this bin
            schechter_pred = dblquad(integrand, z_lo, z_hi, l_lo, l_hi, args=(phi_star, l_star, alpha))# * 1/u.sr
            # print(z_lo, l_lo, schechter_pred)
            schechter_grid[zind][lind] = schechter_pred[0]
    
    # Convert grid from clusters per sr to just clusters:
    schechter_grid *= sample_area
    return schechter_grid


def sample_schechter(schechter_prob_interp, z_min=0.1, z_max=0.2, log_l_min=42, log_l_max=48, schechter_sum=548, n_samp=1000, do_plot=False):
    # Set up for loops and sample dictionaries for later use
    samples_list = []

    # Define high resolution probability grid
    z_arr = np.linspace(z_min, z_max, 2000)
    l_arr = np.linspace(log_l_min, log_l_max, 2000)
    z_grid, l_grid = np.meshgrid(z_arr, l_arr, indexing="ij")
    prob_dist = schechter_prob_interp((z_grid, l_grid))
    prob_dist /= np.sum(prob_dist)
    z_probs = np.sum(prob_dist, axis=1)
    
    # Sample schechter
    for x in tqdm.tqdm(range(n_samp), desc="Sampling Schechter"):
        # redshift samples
        z_choices = np.random.choice(len(z_arr), 548, p=z_probs)
        z_selects = z_arr[z_choices]

        # Sample luminosities
        l_probs = prob_dist[z_choices]
        for z_select, l_prob in zip(z_selects, l_probs):
            l_select = np.random.choice(l_arr, 1, p=l_prob/np.sum(l_prob))[0]
            samples_list.append({
                "sample": x,
                "z": z_select,
                "log_l": l_select
            })
    
    samples_df = pd.DataFrame.from_records(samples_list)
    samples_df["log_flux"] = samples_df["log_l"] - np.log10((4*np.pi*(COSMO.luminosity_distance(samples_df["z"])**2).value))

    if do_plot:
        plt.hist2d(samples_df["z"], samples_df["log_l"])
        plt.colorbar()
        plt.xlabel("redshift")
        plt.ylabel("log(L)")
        plt.show()

        plt.hist(samples_df["log_flux"], bins=25)
        plt.xlabel("log(flux)")
        plt.show()
    return samples_df


def flux_curve_sampled(sample_df, erosita_flux_hist, schechter_sum, flux_midpoints):
    # Bin sampled fluxes to same histogram (roughly) as erosita, working with density for now
    sample_flux_hist = np.histogram(sample_df["log_flux"], erosita_flux_hist[1])  # , density=True)
    sample_flux_hist_scale = sample_flux_hist[0] * schechter_sum / np.sum(sample_flux_hist[0])
    
    plt.step(flux_midpoints, erosita_flux_hist[0], where='mid', label="Observed")
    plt.step(flux_midpoints, sample_flux_hist_scale, where='mid', label="Schechter")
    plt.legend()
    plt.xlabel("$\log(f_{500})")
    plt.ylabel("Frequency")
    plt.clf()

    # Now fit a sigmoid to this
    initial_theta = (10, 37)
    ndim, nwalkers = 2, 100
    pos = np.array(initial_theta) + 1e-4 * np.random.randn(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(flux_midpoints, erosita_flux_hist[0], model_counts_1d),
                                    kwargs={"schechter_pred": sample_flux_hist_scale, 
                                            "data_log_factorials": calculate_log_factorials(erosita_flux_hist[0])})
    sampler.run_mcmc(pos, 5000, progress=True)

    # Get the samples and do a corner plot
    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)[:] # Using 100 steps as burn in and thinning by 15
    fig = corner.corner(flat_samples, labels=["A", "B"], show_titles=True)
    plt.show()

    # Plot the chains
    # fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
    # samples = sampler.get_chain()
    # labels = ["A", "B"]
    # for aid, ax in enumerate(axes):
    #     ax.plot(samples[:, :, aid], "k", alpha=0.3)
    #     ax.set_xlim(0, len(samples))
    #     ax.set_ylabel(labels[aid])
    # axes[-1].set_xlabel("step number")
    # plt.show()
    
    pred_from_emcee = model_counts_1d(flux_midpoints, 
                                      (np.median(flat_samples.T, axis=1)), 
                                      schechter_pred=sample_flux_hist_scale)
    # schecter_interp = Inter
    plt.figure(figsize=(7,6))
    plt.scatter(flux_midpoints, erosita_flux_hist[0], label="Observed", zorder=1)
    plt.step(flux_midpoints, sample_flux_hist_scale, where='mid', zorder=1)
    plt.step(flux_midpoints, sample_flux_hist_scale, label="Schechter", where='mid', zorder=1)
    #plt.step(flux_midpoints, pred_from_emcee, label=r"Schecter $ \times \text{ }\sigma(f)$", where='mid', zorder=1)#, c='lightgreen')
    ## plt.yscale("log")
    plt.xlabel(r"$\log(f_{500} \text{ (erg/s/Mpc}^2))$")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Test goodness of fit (posterior predictive p approach)
    median_res = np.median(flat_samples.T, axis=1)
    sel_func = sigmoid_1D(flux_midpoints, median_res[0], median_res[1])
    data_chi_sq = np.sum(((erosita_flux_hist[0]-pred_from_emcee)**2) / pred_from_emcee)
    chi_list = []
    theta_tests = flat_samples[np.random.choice(range(len(flat_samples)), 5000, False)]
    samp_ids = sample_df["sample"].drop_duplicates().values
    samp_tests = flat_samples[np.random.choice(len(samp_ids), 5000, False)]

    sel_func_tests = sigmoid_1D(np.array([flux_midpoints]*len(theta_tests)), theta_tests[:, 0][:, None], theta_tests[:, 1][:, None])
    
    # What is this line: this is getting the "model" for each of the 500 draws of theta
    pred_counts_test = sample_flux_hist_scale * sel_func_tests
    print(pred_counts_test.shape)

    # Generate y_reps based on a poisson distribution with means from pred_counts_test
    y_reps = np.random.poisson(pred_counts_test)
    
    # test stat for y rep (sum of clusters found)
    test_y_rep = np.sum(y_reps, axis=(1))
    # test stat for y
    test_y = np.sum(erosita_flux_hist[0])
    p_val = 100 * sum(test_y_rep > test_y) / len(test_y_rep)
    plt.hist(test_y_rep, 25, label=r"$y^\text{rep}$")
    plt.vlines(test_y, 0, 600, color="red", label=f"Observed\np = {p_val:.2f}")
    plt.legend()
    plt.xlabel("Number of Clusters")
    plt.show()

    # Test stat with chi square instead
    chi_y_rep = np.sum(((y_reps-pred_counts_test)**2) / pred_counts_test, axis=(1))
    chi_y_obs_theta = np.sum(((erosita_flux_hist[0]-pred_counts_test)**2) / pred_counts_test, axis=(1))
    chi_p_val = 100 * sum(chi_y_rep > chi_y_obs_theta) / len(chi_y_rep)

    plt.scatter(chi_y_obs_theta, chi_y_rep, s=1)
    plt.plot(np.linspace(min(chi_y_rep), max(chi_y_rep), 2), np.linspace(min(chi_y_rep), max(chi_y_rep), 2), color="red")
    plt.gca().set_aspect(1)
    plt.title(f"p={chi_p_val}")
    plt.xlabel(r"$\chi^2(y, \theta)")
    plt.ylabel(r"$\chi^2(y^\text{rep}, \theta)")
    plt.show()
    
    return

    for samp_id in tqdm.tqdm(samp_ids):
        samp_clusts = sample_df[sample_df["sample"]==samp_id].copy()
        samp_hist = np.histogram(samp_clusts["log_flux"], bins=erosita_flux_hist[1])

        samp_hist_sel = samp_hist[0] * sel_func_tests

        samp_chis = np.sum(((samp_hist_sel-pred_counts_test)**2) / pred_counts_test, axis=1)
        #print((((samp_hist_sel-pred_counts_test)**2) / pred_counts_test).shape)
        if any(samp_chis > 1e4):
            samp_clusts.to_csv("chi_too_big.csv")
            continue
        
        chi_list += list(samp_chis)

    pct = (np.sum(np.array(chi_list) > data_chi_sq) / len(chi_list)) * 100
    plt.hist(chi_list, bins=25, density=True)
    plt.vlines(data_chi_sq, 0, 0.05, color='red', label=f"data chi < {pct:.2f}%")
    plt.xlabel("$\chi ^2$")
    plt.legend()
    plt.show()


def main(sample_path="data/emain_wen-han_final_20250328_1052", schechter_clust_path="schechter_clusts.csv"):
    np.random.seed(42)
    # Load in the sample catalogue and set up histogram grid
    emain, wh = load_catalogue(sample_path)

    # Constrain z range of emain to remove "fuzz"
    emain = emain[(emain["BEST_Z_1"] <= 0.2) & (emain["BEST_Z_1"] >= 0.1)]
    histo2d = np.histogram2d(emain["BEST_Z_1"], np.log10(emain["L500_1"])+42)

    z_bins = histo2d[1]
    lumin_bins = 10 ** histo2d[2] # * u.erg/u.second
    plt.imshow(histo2d[0].T, extent=[z_bins[0], z_bins[-1], np.log10(lumin_bins[0]), np.log10(lumin_bins[-1])], 
               aspect="auto", origin="lower")
    plt.ylabel("log(L_500)")
    plt.xlabel("Redshift")
    plt.ylabel("$\log(L_{500})$")
    plt.xlabel("Redshift")
    plt.colorbar(label="N(L, z)")
    plt.clf()
    #return

    # Calculate flux of emain clusters
    emain["log_flux"] = np.log10(emain["L500_1"])+42 - np.log10((4*np.pi*(COSMO.luminosity_distance(emain["BEST_Z_1"])**2).value))
    erosita_flux_hist = np.histogram(emain["log_flux"], bins=75) #)len(emain["log_flux"]))  #, density=True)
    flux_midpoints = (erosita_flux_hist[1][1:]+erosita_flux_hist[1][:-1])/2

    schechter_grid = evalaute_schechter_lz(histo2d[0], z_bins, lumin_bins)
    schechter_sum = np.sum(schechter_grid)

    # Get midpoints
    mid_points = [(z_bins[1:] + z_bins[:-1])/2, (lumin_bins[1:] + lumin_bins[:-1])/2]

    # 2D fit
    #fit2d(mid_points, histo2d, schechter_grid, z_bins, lumin_bins)

    # 1D fit
    #fit_1d_grid(schechter_grid, histo2d, mid_points)

    # Try to load a pregen schechter path
    try:
        sample_df = pd.read_csv(schechter_clust_path)

    except Exception as exc:
        # Fit by sampling the schechter function
        schechter_grid_fine = np.zeros((50, 50))
        fine_z_bins = np.linspace(min(z_bins), max(z_bins), 51)
        fine_l_bins = 10 ** np.linspace(min(np.log10(lumin_bins)), max(np.log10(lumin_bins)), 51)
        schechter_grid_fine = evalaute_schechter_lz(schechter_grid_fine, fine_z_bins, fine_l_bins)
        fine_mid_points = [(fine_z_bins[1:] + fine_z_bins[:-1])/2, np.log10((fine_l_bins[1:] + fine_l_bins[:-1])/2)]

        # Normalise schechter function so it sums to 1 (for making a pdf)
        schechter_sum = np.sum(schechter_grid_fine)
        print(schechter_grid_fine.shape)
        schechter_normalise = schechter_grid_fine / schechter_sum

        # Build interpolator with extrapolation for edge cases (fill_value=None)
        schechter_prob_interp = RegularGridInterpolator(fine_mid_points, schechter_normalise, bounds_error=False, fill_value=None)
        
        # Check interpolator
        # max Z, min L, where the peak should be
        print(schechter_prob_interp((z_bins[-1], fine_mid_points[1][0])))

        # Sample
        sample_df = sample_schechter(schechter_prob_interp, min(z_bins), max(z_bins), 
                                    min(np.log10(lumin_bins)), max(np.log10(lumin_bins)),
                                    schechter_sum=schechter_sum, n_samp=5000)
        sample_df.to_csv(schechter_clust_path)
    

    flux_curve_sampled(sample_df, erosita_flux_hist, schechter_sum, flux_midpoints)
    return
    fit2d_alt_ppc(mid_points, histo2d, schechter_grid, z_bins, lumin_bins, sample_df)
    flux_curve_sampled(sample_df)
    
    plt.imshow(schechter_normalise.T, extent=[z_bins[0], z_bins[-1], np.log10(lumin_bins[0]), np.log10(lumin_bins[-1])], 
               aspect="auto", origin="lower")
    plt.ylabel("log(L_500)")
    plt.xlabel("Redshift")
    plt.colorbar(label="N(L, z) from Schechter and Sigmoid")
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
