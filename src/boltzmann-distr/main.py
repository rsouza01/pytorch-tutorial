#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Billiards simulation script.
This script simulates the motion of particles in a billiards-like environment.
It initializes particles with random positions and velocities, simulates their motion,
and visualizes the results using matplotlib.
"""

import sys
import logging
from datetime import datetime
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import yaml

import motion_plot as mp

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%H:%M:%S",
 )

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
        cfg["simulation"]["initial_velocity"]

def print_header(n_particles, radius, run_animation, initial_velocity):
    """
    Print the header information for the simulation.
    """
    logger.info("="*100)
    logger.info("Billiards Simulation")
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

def get_deltad_pairs(positions):
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
    # ts = 10
    rs = np.zeros((ts, positions.shape[0], positions.shape[1]))
    # logger.info("position.shape[0]: %s", position.shape[0])
    # logger.info("position.shape[1]: %s", position.shape[1])
    # logger.info("rs size: %s", len(rs))

    vs = np.zeros((ts, velocities.shape[0], velocities.shape[1]))
    # logger.info("v.shape[0]: %s", velocities.shape[0])
    # logger.info("v.shape[1]: %s", velocities.shape[1])
    # logger.info("vs: %s", vs)

    # Initial state
    rs[0] = positions.copy()    # rs[0][0]=x, rs[0][1]=y
    vs[0] = velocities.copy()  # vs[0][0]=v_x, vs[0][1]=v_y

    # logger.info(80*">")
    # logger.info(">>>> rs[0]: %s", rs[0])
    # logger.info(">>>> vs[0]: %s", vs[0])
    # logger.info(80*">")

    for i in range(1, ts):
        logger.info(">>>> i: %d, positions[0]): %s", i, positions[0])
        logger.info(">>>> i: %d, get_deltad_pairs(position): %s", i, get_deltad_pairs(positions))
        
        ic = id_pairs[get_deltad_pairs(positions) < d_cutoff]
        
        # logger.info(">>>> i: %d, id: %s", i, str(ic))
        
        velocities[:,ic[:,0]], velocities[:,ic[:,1]] = compute_new_v(
            velocities[:,ic[:,0]], velocities[:,ic[:,1]],
            positions[:,ic[:,0]], positions[:,ic[:,1]]
        )
        velocities[0, positions[0] > 1] = -np.abs(velocities[0, positions[0] > 1])
        velocities[0, positions[0] < 0] = np.abs(velocities[0, positions[0] < 0])
        velocities[1, positions[1] > 1] = -np.abs(velocities[1, positions[1] > 1])
        velocities[1, positions[1] < 0] = np.abs(velocities[1, positions[1] < 0])
        positions = positions + velocities * dt
        rs[i] = positions.copy()
        vs[i] = velocities.copy()

    return rs, vs

def main() -> int:
    """
    Main function to run the simulation.
    """

    time_start = datetime.now()

    n_particles, radius, run_animation, initial_velocity = get_config()

    print_header(n_particles, radius, run_animation, initial_velocity)

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

    for _id in ids:
        logger.info("Particle %s: X=%s, Y=%s", _id, p_positions[0][_id], p_positions[1][_id])

    # Generate all pairs of particle IDs, for collision detection
    ids_pairs = np.asarray(list(combinations(ids,2)))
    # logger.info("Particle pairs: %s", ids_pairs)
    logger.info("# Particle pairs: %s", len(ids_pairs))

    # Initial velocities.
    v = np.zeros((2,n_particles))
    v[0][ixr] = -initial_velocity
    v[0][ixl] = initial_velocity

    logger.info("Calling motion...")
    # rs, vs = motion(p_positions, v, ids_pairs, ts=10000, dt=0.000008, d_cutoff=2*radius)
    rs, vs = simulate_motion(p_positions, v, ids_pairs, ts=10000, dt=0.000008, d_cutoff=2*radius)

    v = np.linspace(0, 2000, 1000)
    a = 2/500**2
    fv = a*v*np.exp(-a*v**2 / 2)

    bins = np.linspace(0,1500,50)

    fig, axes = plt.subplots(1, 2, figsize=(20,10))

    if not run_animation:
        logger.info("Animation generation disabled, exiting now")
        return 0

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
