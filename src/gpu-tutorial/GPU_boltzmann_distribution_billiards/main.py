#!/usr/bin/env python3

import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from matplotlib.animation import PillowWriter
import scienceplots
plt.style.use(['science', 'notebook'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import time
import motion as motion

def main(n_particles):
	r = torch.rand((2,n_particles)).to(device)
	ixr = r[0]>0.5
	ixl = r[0]<=0.5
	ids = torch.arange(n_particles)
	ids_pairs = torch.combinations(ids,2).to(device)
	v = torch.zeros((2,n_particles)).to(device)
	v[0][ixr] = -500
	v[0][ixl] = 500
	radius = 0.0005
	# rs, vs = motion.motion(r, v, ids_pairs, ts=1000, dt=0.000008, d_cutoff=2*radius)
	rs, vs = motion.motion(r, v, ids_pairs, ts=1000, dt=0.000008, d_cutoff=2*radius)

	v = np.linspace(0, 2000, 1000)
	a = 2/500**2
	fv = a*v*np.exp(-a*v**2 / 2)


	bins = np.linspace(0,1500,50)
	plt.figure()
	plt.hist(torch.sqrt(torch.sum(vs[-1]**2, axis=0)).cpu(), bins=bins, density=True)
	plt.plot(v,fv)
	plt.xlabel('Velocity [m/s]')
	plt.ylabel('# Particles')

	fig, axes = plt.subplots(1, 2, figsize=(20,10))
	axes[0].clear()
	vmin = 0
	vmax = 1
	axes[0].set_xlim(0,1)
	axes[0].set_ylim(0,1)
	markersize = 2 * radius * axes[0].get_window_extent().width  / (vmax-vmin) * 72./fig.dpi
	red, = axes[0].plot([], [], 'o', color='red', markersize=markersize)
	blue, = axes[0].plot([], [], 'o', color='blue', markersize=markersize)
	n, bins, patches = axes[1].hist(torch.sqrt(torch.sum(vs[0]**2, axis=0)).cpu(), bins=bins, density=True)
	axes[1].plot(v,fv)
	axes[1].set_ylim(top=0.003)

	def animate(i):
		xred, yred = rs[i][0][ixr].cpu(), rs[i][1][ixr].cpu()
		xblue, yblue = rs[i][0][ixl].cpu(),rs[i][1][ixl].cpu()
		red.set_data(xred, yred)
		blue.set_data(xblue, yblue)
		hist, _ = np.histogram(torch.sqrt(torch.sum(vs[i]**2, axis=0)).cpu(), bins=bins, density=True)
		for i, patch in enumerate(patches):
			patch.set_height(hist[i])
		return red, blue

	writer = animation.FFMpegWriter(fps=30)
	ani = animation.FuncAnimation(fig, animate, frames=500, interval=50, blit=True)
	ani.save(f'./ani_gpu_{n_particles}.mp4',writer=writer,dpi=100)

if __name__ == "__main__":
	start_time = time.time()
	particles = 20000 # <= Limite da GPU
	# particles = 50
	main(particles)
	print("--- %s seconds ---" % (round(time.time() - start_time, 2)))
