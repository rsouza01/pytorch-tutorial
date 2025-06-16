#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%H:%M:%S",
 )


def animate(i, fig, axes, rs, radius, ixr, ixl, v, fv, vs, bins):
    # logger.info(f">> CALL animate({i})")
    try:
        [ax.clear() for ax in axes]

        # Plot 1
        plot_scatter(i, axes[0], rs, radius, ixr, ixl)

        # Plot 2 - Histogram
        plot_histogram(i, axes[1], v, vs, fv, bins)

        fig.tight_layout()
    except Exception as e:
        logger.info("Exception on animate: '%s'", str(e))

def plot_scatter(i, ax, rs, radius, ixr, ixl):

    xred, yred = rs[i][0][ixr], rs[i][1][ixr]
    xblue, yblue = rs[i][0][ixl],rs[i][1][ixl]

    circles_red = [plt.Circle((xi, yi), radius=4*radius, linewidth=0) for xi,yi in zip(xred,yred)]
    circles_blue = [plt.Circle((xi, yi), radius=4*radius, linewidth=0) for xi,yi in zip(xblue,yblue)]

    cred = matplotlib.collections.PatchCollection(circles_red, facecolors="red")
    cblue = matplotlib.collections.PatchCollection(circles_blue, facecolors="blue")

    ax.add_collection(cred)
    ax.add_collection(cblue)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)


def plot_histogram(i, ax, v, vs, fv, bins):
    ax.hist(np.sqrt(np.sum(vs[i]**2, axis=0)), bins=bins, density=True)
    ax.plot(v,fv)
    ax.set_xlabel(f"Velocity [m/s], Frame {i}")
    ax.set_ylabel("# Particles")
    ax.set_xlim(0,1500)
    ax.set_ylim(0,0.006)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)

