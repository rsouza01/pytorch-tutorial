#!/usr/bin/env python3
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import logging
from decorators import timer
import random

import os
from matplotlib.animation import PillowWriter
import scienceplots
import sys
from collections import namedtuple

plt.style.use(['science', 'notebook'])

import time

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
					format='%(levelname)s: %(message)s')

Position = namedtuple('Position', ['x', 'y'])
Particle = namedtuple('Particle', ['radius', 'position'])

SimulationSettings = namedtuple('SimulationSettings', [
	'number_of_particles',
	'particle_v0',
	'particle_radius',
	'target',
	'fps'])


def setup_figure(simulation_settings: SimulationSettings, position_start, position_end: Position, ):
	color = 'red'

	# --- Setup the Figure and Axes ---
	fig, ax = plt.subplots()
	# Set the limits of the square plot
	ax.set_xlim(0, 2)
	ax.set_ylim(-0.5, 0.5)
	ax.set_aspect('equal', adjustable='box')  # Ensure it looks square
	ax.set_title(
		f"Uniform Motion from ({0}, {0}) to ({2}, {2})")
	ax.grid(True, linestyle='--', alpha=0.6)

	# A circle is a Patch object in matplotlib
	# It's initially placed at the starting position (x_start, y_start)
	circle = plt.Circle((0, 0), simulation_settings.particle_radius,
						fc=color, ec='black', lw=1.0)
	ax.add_patch(circle)

	# The target
	circle_target = plt.Circle((1, 0), simulation_settings.target.radius, fc='blue', ec='black', lw=1.0)
	ax.add_patch(circle_target)

	return fig, ax


def collided(particle, target: Particle) -> bool:
	distance = math.sqrt(
		(particle.position.x - target.position.x) ** 2 +
		(particle.position.y - target.position.y) ** 2)
	# logging.info('distance: %f' % distance)
	return distance <= (particle.radius + target.radius)


def simulation_particle(particle: Particle, target: Particle, v_0: float, t_0: float, delta_t: float):
	logging.info('position: (%f,%f,)' % (particle.position.x, particle.position.y))
	logging.info(
		f"(v_0: {target.radius:.2f}, t_0: {t_0:.2f}, delta_t: {delta_t:.2f})")

	logging.info(
		f"target: (radius: {target.radius:.2f}, position: ({target.position.x:.2f}, {target.position.y:.2f}))")

	x = particle.position.x
	t = t_0

	while x < target.position.x:
		t = t + delta_t
		x += v_0 * t
		particle = Particle(particle.radius, Position(x, particle.position.y))
		logging.info('new position: (%f,%f), collided: %r' % (x, particle.position.y,
															  collided(particle=particle,
																	   target=target)))


def animate(frame):
	logging.info('frame: (%d)' % (frame))
	return plt.Circle((1, 0), 0.2, fc='blue', ec='black', lw=1.0),


@timer
def main(simulation_settings: SimulationSettings):
	t_0 = 0.0
	v_0 = simulation_settings.particle_v0

	# Time step based on FPS, 1 second / frames per second
	interval_ms = 1000 / simulation_settings.fps
	delta_t = 1.0 / simulation_settings.fps

	fig, ax = setup_figure(simulation_settings, position_start=Position(0, 0), position_end=Position(0, 0))

	logging.info('simulation_settings: %s' % str(simulation_settings))
	for i in range(simulation_settings.number_of_particles):
		simulation_particle(Particle(radius=simulation_settings.particle_radius,
									 position=Position(x=0, y=0)), simulation_settings.target, v_0, t_0, delta_t)

	fig.savefig("./simulations/frame.png")

	n_frames = int(150)
	logging.info('n_frames: %d' % n_frames)

	# --- Create the Animation ---
	# FuncAnimation(fig, func, frames, init_func, interval, blit)
	ani = animation.FuncAnimation(
		fig,
		animate,
		frames=n_frames,
		interval=interval_ms,
		blit=True,  # Use blit for faster rendering (only redraws what changed)
		repeat=False  # The animation should stop after one cycle (5 seconds)
	)
	writer = animation.FFMpegWriter(fps=30)
	# Optional: To save the animation (requires a writer like 'ffmpeg' or 'pillow')
	# ani.save('linear_motion.mp4', writer=writer, fps=fps)
	ani.save(f'./simulations/my_linear_motion.mp4', writer=writer, dpi=100)


if __name__ == "__main__":
	number_of_particles: int = 1
	target_radius: float = 0.05
	target_position = Position(x=1, y=0)
	particle_radius: float = 0.01
	particle_v0: float = 0.2

	simulationSettings = SimulationSettings(
		number_of_particles=number_of_particles,
		particle_v0=particle_v0,
		target=Particle(target_radius, target_position),
		particle_radius=particle_radius,
		fps=50)

	main(simulationSettings)
