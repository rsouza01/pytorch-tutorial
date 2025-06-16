#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from matplotlib.animation import PillowWriter
from itertools import combinations
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%Y-%m-%d %H:%M",
 )


def get_new_v(_v1, _v2, _r1, _r2):
    return _v1 - np.diag((_v1-_v2).T@(_r1-_r2))/np.sum((_r1-_r2)**2, axis=0) * (_r1-_r2)

def get_delta_pairs(x):
    return np.diff(np.asarray(list(combinations(x, 2))), axis=1).ravel()

def get_deltad_pairs(r):
    return np.sqrt(
        get_delta_pairs(r[0])**2 + get_delta_pairs(r[1])**2
    )

def compute_new_v(v1, v2, r1, r2):
    return get_new_v(v1, v2, r1, r2), get_new_v(v2, v1, r2, r1)

def motion(r, v, id_pairs, ts, dt, d_cutoff):
    rs = np.zeros((ts, r.shape[0], r.shape[1]))     
    vs = np.zeros((ts, v.shape[0], v.shape[1]))     
    # Initial state
    rs[0] = r.copy()
    vs[0] = v.copy()

    for i in range(1, ts):
        ic = id_pairs[get_deltad_pairs(r) < d_cutoff]
        v[:,ic[:,0]], v[:,ic[:,1]] = compute_new_v(
            v[:,ic[:,0]], v[:,ic[:,1]],
            r[:,ic[:,0]], r[:,ic[:,1]]
        )
        v[0, r[0] > 1] = -np.abs(v[0, r[0] > 1])
        v[0, r[0] < 0] = np.abs(v[0, r[0] < 0])
        v[1, r[1] > 1] = -np.abs(v[1, r[1] > 1])
        v[1, r[1] < 0] = np.abs(v[1, r[1] < 0])
        r = r + v * dt
        rs[i] = r.copy()
        vs[i] = v.copy()
    
    return rs, vs

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
        logger.info("Exception on animate: " + str(e))

def plot_scatter(i, ax, rs, radius, ixr, ixl):

        xred, yred = rs[i][0][ixr], rs[i][1][ixr]
        xblue, yblue = rs[i][0][ixl],rs[i][1][ixl]

        circles_red = [plt.Circle((xi, yi), radius=4*radius, linewidth=0) for xi,yi in zip(xred,yred)]
        circles_blue = [plt.Circle((xi, yi), radius=4*radius, linewidth=0) for xi,yi in zip(xblue,yblue)]

        cred = matplotlib.collections.PatchCollection(circles_red, facecolors='red')
        cblue = matplotlib.collections.PatchCollection(circles_blue, facecolors='blue')

        ax.add_collection(cred)
        ax.add_collection(cblue)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)


def plot_histogram(i, ax, v, vs, fv, bins):
        ax.hist(np.sqrt(np.sum(vs[i]**2, axis=0)), bins=bins, density=True)
        ax.plot(v,fv)
        ax.set_xlabel(f'Velocity [m/s], Frame {i}')
        ax.set_ylabel(f'# Particles')
        ax.set_xlim(0,1500)
        ax.set_ylim(0,0.006)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)


def main() -> int:

    time_start = datetime.now()

    n_particles = 6
    radius = 0.0015

    r = np.random.random((2,n_particles))
    logger.info("Initial X positions: %s", r[0])
    logger.info("Initial Y positions: %s", r[1])

    ixr = r[0]>0.5 
    ixl = r[0]<=0.5 
    logger.info("Red particles indices: %s", ixr)
    logger.info("Blue particles indices: %s", ixl)

    ids = np.arange(n_particles)
    logger.info("Particle IDs: %s", ids)
    ids_pairs = np.asarray(list(combinations(ids,2)))
    logger.info("Particle pairs: %s", ids_pairs)
    v = np.zeros((2,n_particles))
    v[0][ixr] = -500
    v[0][ixl] = 500
    

    logger.info("Calling motion...")
    # rs, vs = motion(r, v, ids_pairs, ts=1000, dt=0.000008, d_cutoff=2*radius)
    rs, vs = motion(r, v, ids_pairs, ts=10000, dt=0.000008, d_cutoff=2*radius)

    v = np.linspace(0, 2000, 1000)
    a = 2/500**2
    fv = a*v*np.exp(-a*v**2 / 2)

    bins = np.linspace(0,1500,50)
    
    fig, axes = plt.subplots(1, 2, figsize=(20,10))

    logger.info("About to call FuncAnimation...")
    ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50, fargs=(fig, axes, rs, radius, ixr, ixl, v, fv, vs, bins))

    logger.info("About to save the charts...")
    ani.save('ani.gif',writer='pillow',fps=30,dpi=100)
    logger.info("Done!")

    time_end = datetime.now()
    difference_seconds = (time_end - time_start).total_seconds()
    logger.info("Execution time: %d minutes", difference_seconds/60)

    return 0



if __name__ == '__main__':
    sys.exit(main())
    