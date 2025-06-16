#!/usr/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from itertools import combinations
from datetime import datetime
import logging
import yaml

import motion_plot as mp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%H:%M:%S",
 )

def get_config(filename="config.yaml"):
    with open(filename) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg["simulation"]["n_particles"], cfg["simulation"]["radius"], cfg["simulation"]["run_partial"], cfg["simulation"]["initial_velocity"]

def print_header(n_particles, radius, run_partial, initial_velocity):
    logger.info("="*100)
    logger.info("Billiards Simulation")
    logger.info("="*100)
    logger.info("n_particles: %s", n_particles)
    logger.info("radius: %s", radius)
    logger.info("run_partial: %s", run_partial)
    logger.info("="*100)


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

def simulate_motion(position, velocities, id_pairs, ts, dt, d_cutoff):
    """
    ts= frames, so 
    [
        [ts, some x, some y], for particle 0
        [ts + 1, some x, some y], for particle 1
        ...
        [ts + n, some x, some y], for particle n
        ...
    ]
    """

    rs = np.zeros((ts, position.shape[0], position.shape[1]))
    logger.info("position.shape[0]: %s", position.shape[0])
    logger.info("position.shape[1]: %s", position.shape[1])
    logger.info("rs size: %s", len(rs))
    logger.info("rs[0]: %s", rs[0])

    vs = np.zeros((ts, velocities.shape[0], velocities.shape[1]))
    logger.info("v.shape[0]: %s", velocities.shape[0])
    logger.info("v.shape[1]: %s", velocities.shape[1])
    # logger.info("vs: %s", vs)

    # Initial state
    rs[0] = position.copy()
    vs[0] = velocities.copy()

    for i in range(1, ts):
        ic = id_pairs[get_deltad_pairs(position) < d_cutoff]
        velocities[:,ic[:,0]], velocities[:,ic[:,1]] = compute_new_v(
            velocities[:,ic[:,0]], velocities[:,ic[:,1]],
            position[:,ic[:,0]], position[:,ic[:,1]]
        )
        velocities[0, position[0] > 1] = -np.abs(velocities[0, position[0] > 1])
        velocities[0, position[0] < 0] = np.abs(velocities[0, position[0] < 0])
        velocities[1, position[1] > 1] = -np.abs(velocities[1, position[1] > 1])
        velocities[1, position[1] < 0] = np.abs(velocities[1, position[1] < 0])
        position = position + velocities * dt
        rs[i] = position.copy()
        vs[i] = velocities.copy()
    
    return rs, vs

def main() -> int:

    time_start = datetime.now()

    n_particles, radius, run_partial, initial_velocity = get_config()

    print_header(n_particles, radius, run_partial, initial_velocity)

    # Array where element 0 is the X position and element 1 is the Y position
    p_positions = np.random.random((2,n_particles))
    logger.info("Initial X positions: %s", p_positions[0])
    logger.info("Initial Y positions: %s", p_positions[1])

    ixr = p_positions[0]>0.5
    ixl = p_positions[0]<=0.5
    logger.info("Red particles indices: %s", ixr)
    logger.info("Blue particles indices: %s", ixl)

    # Each particle has an ID, which is just its index in the array
    ids = np.arange(n_particles)
    logger.info("Particle IDs: %s", ids)

    for id in ids:
        logger.info("Particle %s: X=%s, Y=%s", id, p_positions[0][id], p_positions[1][id])

    # Generate all pairs of particle IDs, for collision detection
    ids_pairs = np.asarray(list(combinations(ids,2)))
    logger.info("Particle pairs: %s", ids_pairs)

    # Initial velocities.
    v = np.zeros((2,n_particles))
    v[0][ixr] = -initial_velocity
    v[0][ixl] = initial_velocity
    
    logger.info("Calling motion...")
    # rs, vs = motion(p_positions, v, ids_pairs, ts=10000, dt=0.000008, d_cutoff=2*radius)
    rs, vs = simulate_motion(p_positions, v, ids_pairs, ts=100, dt=0.000008, d_cutoff=2*radius)

    v = np.linspace(0, 2000, 1000)
    a = 2/500**2
    fv = a*v*np.exp(-a*v**2 / 2)

    bins = np.linspace(0,1500,50)
    
    fig, axes = plt.subplots(1, 2, figsize=(20,10))

    if run_partial:
        logger.info("Running partial, exiting now")
        return 0

    logger.info("About to call FuncAnimation...")
    ani = animation.FuncAnimation(fig, mp.animate, frames=1000, interval=50, fargs=(fig, axes, rs, radius, ixr, ixl, v, fv, vs, bins))

    logger.info("About to save the charts...")
    ani.save('ani.gif',writer='pillow',fps=30,dpi=100)
    logger.info("Done!")

    time_end = datetime.now()
    difference_seconds = (time_end - time_start).total_seconds()
    logger.info("Execution time: %d minutes", difference_seconds/60)

    return 0



if __name__ == '__main__':
    sys.exit(main())
    