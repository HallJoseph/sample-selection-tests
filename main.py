# Code for trying to produce an appropriate model for the eRASS selection function
# Created: 2026-01-06, by: Joseph Hall

from astropy.modeling.models import Schechter1D
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np


def xlf_schechter(lx, phi_star, lx_star, alpha):
    return (phi_star * ((lx/lx_star)**-alpha) * np.exp(-lx/lx_star) * 1/lx_star)


def main():
    # Using WARPS XLF
    phi_star = 2.94e-7 * (u.Mpc ** -3)
    l_star = 2.64e44 * u.erg / u.second
    alpha = 1.69
    lumins = np.logspace(42, 45.3)*u.erg/u.second

    fig, ax = plt.subplots()
    ax.plot(lumins, xlf_schechter(lumins, phi_star, l_star, alpha) * 1e44)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.show()


if __name__ == "__main__":
    main()
