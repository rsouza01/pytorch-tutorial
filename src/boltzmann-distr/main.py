#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Billiards simulation script.
This script simulates the motion of particles in a billiards-like environment.
It initializes particles with random positions and velocities, simulates their motion,
and visualizes the results using matplotlib.

Source: https://www.youtube.com/watch?v=65kl4eE9ovI&t=1309s&ab_channel=Mr.PSolver
"""

import sys
import logging
from datetime import datetime
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from sympy import pprint
import yaml

import motion_plot as mp

log_format = "{asctime} - {levelname} - {funcName} - {message}"
logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt="%H:%M:%S", style="{")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_config(filename="config.yaml"):
    """
    Load the configuration from a YAML file.
    Returns:
        n_particles (int): Number of particles in the simulation.
        radius (float): Radius of the particles.
        run_animation (bool): Whether to run a partial simulation.
        initial_velocity (float): Initial velocity of the particles.
    """
    with open(filename, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    return cfg["simulation"]["n_particles"], \
        cfg["simulation"]["radius"], \
        cfg["simulation"]["run_animation"], \
        cfg["simulation"]["initial_velocity"], \
        cfg["simulation"]["ts"]

def print_header(n_particles, radius, run_animation, initial_velocity):
    """
    Print the header information for the simulation.
    """
    logger.info("="*100)
    logger.info("Maxwell-Boltzmann Distribution Simulation")
    logger.info("="*100)
    logger.info("n_particles: %s", n_particles)
    logger.info("radius: %s", radius)
    logger.info("run_animation: %s", run_animation)
    logger.info("initial_velocity: %s", initial_velocity)
    logger.info("="*100)


def get_new_v(_v1, _v2, _r1, _r2):
    """
    Calculate the new velocity for a particle after a collision.
    """
    return _v1 - np.diag((_v1-_v2).T@(_r1-_r2))/np.sum((_r1-_r2)**2, axis=0) * (_r1-_r2)

def get_delta_pairs(x):
    """
    Calculate the difference between pairs of particles.
    """
    return np.diff(np.asarray(list(combinations(x, 2))), axis=1).ravel()

def get_delta_d_pairs(positions):
    """Calculate the distance between pairs of particles."""
    return np.sqrt(
        get_delta_pairs(positions[0])**2 + get_delta_pairs(positions[1])**2
    )

def compute_new_v(v1, v2, r1, r2):
    """
    Compute the new velocities for two particles after a collision.
    Args:
        v1 (np.ndarray): Velocity of the first particle.
        v2 (np.ndarray): Velocity of the second particle.
        r1 (np.ndarray): Position of the first particle.
        r2 (np.ndarray): Position of the second particle."""
    return get_new_v(v1, v2, r1, r2), get_new_v(v2, v1, r2, r1)

def simulate_motion(positions, velocities, id_pairs, ts, dt, d_cutoff):
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
    
    rs = np.zeros((ts, positions.shape[0], positions.shape[1]))
    vs = np.zeros((ts, velocities.shape[0], velocities.shape[1]))

    # Initial state
    # t = 0, first element of rs
    # Then rs[0][0] has ALL x positions of all particles for t=0 and rs[0][1] has ALL y positions of all particles for t=0.
    # The next two statements store a snapshot of the initial positions and velocities
    rs[0] = positions.copy()   # rs[t][0][0]=x, rs[t][0][1]=y
    vs[0] = velocities.copy()  # vs[t][0][0]=v_x, vs[t][0][1]=v_y
    logger.debug("Snapshot ts=0: (x=%s, y=%s, v_x=%s, v_y=%s)", rs[0][0][0], rs[0][1][0], vs[0][0][0], vs[0][1][0])
    logger.debug("rs.shape: %s", rs.shape)

    for i in range(1, ts):
        # ic contains the indices of pairs of particles that are within the cutoff distance
        # id_pairs is a 2D array where each row is a pair of particle IDs
        ic = id_pairs[get_delta_d_pairs(positions) < 2 * d_cutoff]
        # logger.info(">>>>> get_delta_d_pairs(positions)")
        # pprint(get_delta_d_pairs(positions))

        #logger.info(">>>>> ic")
        #pprint(ic)
        
        velocities[:,ic[:,0]], velocities[:,ic[:,1]] = compute_new_v(
            velocities[:,ic[:,0]], velocities[:,ic[:,1]],
            positions[:,ic[:,0]], positions[:,ic[:,1]]
        )
        velocities[0, positions[0] >= 1 - d_cutoff] = -np.abs(velocities[0, positions[0] >= 1 - d_cutoff])
        velocities[0, positions[0] <= 0 + d_cutoff] = np.abs(velocities[0, positions[0] <= 0 + d_cutoff])
        velocities[1, positions[1] >= 1 - d_cutoff] = -np.abs(velocities[1, positions[1] >= 1 - d_cutoff])
        velocities[1, positions[1] <= 0 + d_cutoff] = np.abs(velocities[1, positions[1] <= 0 + d_cutoff])
        positions = positions + velocities * dt
        rs[i] = positions.copy()
        vs[i] = velocities.copy()

    
        # logger.info("rs: %s", rs)

    # logger.info(10*">>>>> rs")
    # logger.info("rs: %s", rs)
    # pprint(rs)
    # logger.info(10*">>>>>")
    # logger.info("vs: %s", vs)
    # pprint(vs)

    return rs, vs

def main() -> int:
    """
    Main function to run the simulation.
    """

    time_start = datetime.now()

    n_particles, radius, run_animation, initial_velocity, ts = get_config()

    print_header(n_particles, radius, run_animation, initial_velocity)

    # Each particle has an ID, which is just its index in the array
    ids = np.arange(n_particles)
    logger.debug("Particle IDs: %s", ids)
    logger.debug("Particle IDs.shape: %s", ids.shape)

    # Generate all pairs of particle IDs, for collision detection
    ids_pairs = np.asarray(list(combinations(ids, 2)))
    logger.debug("Particle pairs: %s", [str(pair) for pair in ids_pairs])
    logger.debug("Particle pairs.shape: %s", ids_pairs.shape)

    # Array where element 0 is the X position and element 1 is the Y position
    positions = np.random.random((2, n_particles))
    # pprint(positions)
    logger.debug("Coordinate particle[0]: (%s, %s)", positions[0][0], positions[1][0])
    logger.debug("Initial X positions: %s", positions[0])
    logger.debug("Initial Y positions: %s", positions[1])
    logger.debug("-"*100)

    ixr = positions[0] > 0.5  # Vector operation, returns a boolean array with True for positions > 0.5
    ixl = positions[0] <= 0.5 # Vector operation, returns a boolean array with True for positions <= 0.5
    logger.debug("Blue(left) particles indices: %s", ids[ixl])
    logger.debug("Red(right) particles indices: %s", ids[ixr])
    logger.debug(">>> positions.shape: %s", positions.shape)
    logger.debug(">>> positions.shape[0](x OR y): %s", positions.shape[0])
    logger.debug(">>> positions.shape[1](x0, x1, x2, ..., or y0, y1, y2, ...): %s", positions.shape[1])
    logger.debug("-"*100)

    # Initial velocities.
    velocities = np.zeros((2, n_particles))
    velocities[0][ixr] = -initial_velocity # <= Numpy boolean indexing, one line assignment for the whole array
    velocities[0][ixl] =  initial_velocity

    # Obvious results, but useful for debugging
    logger.debug("Blue(left) particles velocities: %s", velocities[0][ixl])
    logger.debug("Red(right) particles velocities: %s", velocities[0][ixr])
    logger.debug("-"*100)

    logger.info("Calling motion...")

    # rs, vs = motion(positions, velocities, ids_pairs, ts=10000, dt=0.000008, d_cutoff=2*radius)
    rs, vs = simulate_motion(positions, velocities, ids_pairs, ts=ts, dt=0.000008, d_cutoff=2*radius)

    if not run_animation:
        logger.info("Animation generation disabled, exiting now.")
        return 0

    logger.info("Animation generation enabled, proceeding...")
    v = np.linspace(0, 2000, 1000)
    a = 2/500**2
    fv = a*v*np.exp(-a*v**2 / 2)

    bins = np.linspace(0,1500,50)

    fig, axes = plt.subplots(1, 2, figsize=(20,10))

    logger.info("Calling FuncAnimation...")
    ani = animation.FuncAnimation(fig,
                                  mp.animate,
                                  frames=1000,
                                  interval=50,
                                  fargs=(fig, axes, rs, radius, ixr, ixl, v, fv, vs, bins))

    logger.info("Saving the charts...")
    ani.save('ani.gif',writer='pillow',fps=30,dpi=100)
    logger.info("Done!")

    time_end = datetime.now()
    difference_seconds = (time_end - time_start).total_seconds()
    logger.info("Execution time: %d minutes", difference_seconds/60)

    return 0



if __name__ == '__main__':
    sys.exit(main())
