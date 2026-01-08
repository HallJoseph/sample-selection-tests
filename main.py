# Code for trying to produce an appropriate model for the eRASS selection function
# Created: 2026-01-06, by: Joseph Hall

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from load_catalogue import load_catalogue


H0 = 70
COSMO = FlatLambdaCDM(H0, 0.3)
LITLLE_H = H0/100


def xlf_schechter(lx, phi_star, lx_star, alpha):
    return (phi_star * ((lx/lx_star)**-alpha) * np.exp(-lx/lx_star) * 1/lx_star)


def integrand(lx, z, phi_star, lx_star, alpha):
    # Get comoving volume
    comovol = COSMO.differential_comoving_volume(z)

    # Get Schechter func value
    phi_l = xlf_schechter(lx, phi_star, lx_star, alpha)

    # Return integrand
    return phi_l * comovol


def main(sample_path="data/emain_wen-han_final_20250328_1052"):
    # Using WARPS/REFLEX XLF 
    phi_star = 2.94e-7 * (u.Mpc ** -3)
    l_star = 2.64e44 * u.erg / u.second
    alpha = 1.69
    lumins = np.logspace(42, 45.3)*u.erg/u.second

    # Integrate Schechter with respect to L
    print(integrand(lumins[0], 0.14, phi_star, l_star, alpha))

    emain, wh = load_catalogue(sample_path)
    print(emain.columns)

    grid = np.histogram2d(emain["BEST_Z_1"], np.log10(emain["L500_1"])+42)
    print(grid)

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
