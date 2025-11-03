#!/usr/bin/env python3

from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import matplotlib
from matplotlib import animation
from matplotlib.animation import PillowWriter
import scienceplots
import logging
from decorators import timer

logging.basicConfig(level=logging.INFO)
plt.style.use(['science', 'notebook'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SimulationSettings = namedtuple('SimulationSettings', [
	'particles',
	'speed',
	'radius',
	'total_simulation_frames',
	'delta_t'])

def get_deltad2_pairs(r, id_pairs):
    dx = torch.diff(torch.stack([r[0][id_pairs[:,0]], r[0][id_pairs[:,1]]]).T).squeeze()
    dy = torch.diff(torch.stack([r[1][id_pairs[:,0]], r[1][id_pairs[:,1]]]).T).squeeze()
    return dx**2 + dy**2

def compute_new_v(v1, v2, r1, r2):
    v1new = v1 - torch.sum((v1-v2)*(r1-r2), axis=0)/torch.sum((r1-r2)**2, axis=0) * (r1-r2)
    v2new = v2 - torch.sum((v1-v2)*(r1-r2), axis=0)/torch.sum((r2-r1)**2, axis=0) * (r2-r1)
    return v1new, v2new

def motion(r, v, id_pairs, ts, dt, d_cutoff):
    rs = torch.zeros((ts, r.shape[0], r.shape[1])).to(device)
    vs = torch.zeros((ts, v.shape[0], v.shape[1])).to(device)
    # Initial State
    rs[0] = r
    vs[0] = v
    for i in range(1,ts):
        ic = id_pairs[get_deltad2_pairs(r, id_pairs) < d_cutoff ** 2]
        v[:,ic[:,0]], v[:,ic[:,1]] = compute_new_v(v[:,ic[:,0]], v[:,ic[:,1]], r[:,ic[:,0]], r[:,ic[:,1]])

        v[0,r[0]>1] = -torch.abs(v[0,r[0]>1])
        v[0,r[0]<0] = torch.abs(v[0,r[0]<0])
        v[1,r[1]>1] = -torch.abs(v[1,r[1]>1])
        v[1,r[1]<0] = torch.abs(v[1,r[1]<0])

        r = r + v*dt
        rs[i] = r
        vs[i] = v
    return rs, vs

@timer
def main(settings):

    r = torch.rand((2,settings.particles)).to(device)
    ixr = r[0]>0.5
    ixl = r[0]<=0.5
    ids = torch.arange(settings.particles)
    ids_pairs = torch.combinations(ids,2).to(device)
    v = torch.zeros((2,settings.particles)).to(device)
    v[0][ixr] = -settings.speed
    v[0][ixl] = settings.speed
    # radius = 0.0005

    # rs, vs = motion.motion(r, v, ids_pairs, ts=2000, dt=0.000008, d_cutoff=2*radius)
    rs, vs = motion(r, v,
					ids_pairs,
					ts=settings.total_simulation_frames,
					dt=settings.delta_t,
					d_cutoff=2*settings.radius)

    fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    axes.clear()
    vmin = 0
    vmax = 1

    # Left side
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_xlabel('Velocity [m/s]')

    markersize = 2 * settings.radius * axes.get_window_extent().width / (vmax - vmin) * 72. / fig.dpi
    red, = axes.plot([], [], 'o', color='red', markersize=markersize)
    blue, = axes.plot([], [], 'o', color='blue', markersize=markersize)


    def animate(i):
        xred, yred = rs[i][0][ixr].cpu(), rs[i][1][ixr].cpu()
        xblue, yblue = rs[i][0][ixl].cpu(),rs[i][1][ixl].cpu()
        red.set_data(xred, yred)
        blue.set_data(xblue, yblue)
        return red, blue

    writer = animation.FFMpegWriter(fps=30)
    ani = animation.FuncAnimation(fig, animate, frames=settings.total_simulation_frames, interval=50, blit=True)
    ani.save(f'./simulations/plot_playground.p_{settings.particles}.v_{settings.speed}.mp4',writer=writer,dpi=100)

# https://matplotlib.org/stable/gallery/animation/simple_anim.html
if __name__ == "__main__":
    simulationSettings = SimulationSettings(
        particles=1,
        speed=1500,
        radius=0.05,
		total_simulation_frames=100,
		delta_t=0.000008,
		# delta_t = 1
	)
    logging.info("calling main()")
    main(simulationSettings)
