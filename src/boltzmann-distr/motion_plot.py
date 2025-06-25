""" Module for animating the motion of particles in a billiards simulation.
This module provides functionality to animate the motion of particles,
plot their positions, and visualize their velocities using matplotlib.
"""

import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%H:%M:%S",
 )


def animate(i, fig, axes, rs, radius, ixr, ixl, v, fv, vs, bins, hist_x_min_value, hist_x_max_value, hist_y_min_value, hist_y_max_value):
    """
    Update the animation frame for the given index `i`.
    This function clears the axes and plots the current state of the particles
    and their velocities.
    Args:
        i (int): The index of the current frame.
        fig (matplotlib.figure.Figure): The figure object for the animation.
        axes (list): List of axes to plot on.
        rs (list): List of particle positions at each frame.
        radius (float): Radius of the particles.
        ixr (list): Indices of red particles.
        ixl (list): Indices of blue particles.
        v (np.ndarray): Velocity values for the histogram.
        fv (np.ndarray): Frequency values for the histogram.
        vs (list): List of velocities at each frame.
        bins (int): Number of bins for the histogram.
    """
    [ax.clear() for ax in axes]

    # Plot 1 - Particles
    plot_scatter(i, axes[0], rs, radius, ixr, ixl)

    # Plot 2 - Histogram
    plot_histogram(i, axes[1], v, vs, fv, bins, hist_x_min_value, hist_x_max_value, hist_y_min_value, hist_y_max_value)

    fig.tight_layout()

def plot_scatter(i, ax, rs, radius, ixr, ixl):
    """Plot the scatter plot of particle positions with circles representing particles."""

    xred, yred = rs[i][0][ixr], rs[i][1][ixr]
    xblue, yblue = rs[i][0][ixl],rs[i][1][ixl]

    circles_red = [
        plt.Circle((xi, yi),
                   radius=2*radius,
                   linewidth=0) for xi,yi in zip(xred,yred)
        ]
    circles_blue = [
        plt.Circle((xi, yi),
                   radius=2*radius,
                   linewidth=0) for xi,yi in zip(xblue,yblue)]

    cred = matplotlib.collections.PatchCollection(circles_red, facecolors="red")
    cblue = matplotlib.collections.PatchCollection(circles_blue, facecolors="blue")

    ax.add_collection(cred)
    ax.add_collection(cblue)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)


def plot_histogram(i, ax, v, vs, fv, bins, hist_x_min_value, hist_x_max_value, hist_y_min_value, hist_y_max_value):
    """Plot the histogram of particle velocities."""

    ax.hist(np.sqrt(np.sum(vs[i]**2, axis=0)), bins=bins, density=True)
    ax.plot(v,fv)
    ax.set_xlabel(f"Velocity [m/s], Frame {i}")
    ax.set_ylabel("# Particles")
    ax.set_xlim(hist_x_min_value, hist_x_max_value)
    ax.set_ylim(hist_y_min_value, hist_y_max_value)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
