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
plt.style.use(['science', 'notebook'])

import time
from collections import namedtuple

logging.basicConfig(level=logging.INFO)

SimulationSettings = namedtuple('SimulationSettings', [
	'x_start',
	'y_start',
	'x_end',
	'y_end',
	'total_time',
	'target_radius',
	'particle_radius',
	'fps'])


def distance(circle, circle_target):
	x, y =  circle.center[0], circle.center[1]
	x_t, y_t =  circle_target.center[0], circle_target.center[1]
	result = math.sqrt((x - x_t) ** 2 + (y - y_t) ** 2)
	# logging.info('distance(): %f' % result)
	return result

@timer
def main(simulation_settings: SimulationSettings):
	# --- Parameters ---

	# Number of frames (N) and frame rate (FPS)
	# Using a common frame rate of 50 FPS for smooth motion
	# fps = 50
	n_frames = int(simulation_settings.total_time * simulation_settings.fps)
	logging.info('n_frames: %d' % n_frames)

	# Interval between frames in milliseconds (1000/FPS)
	interval_ms = 1000 / simulation_settings.fps
	logging.info('interval_ms: %d' % interval_ms)

	distance_x = simulation_settings.x_end - simulation_settings.x_start
	logging.info('Distance_x: %f' % distance_x)

	speed = distance_x/(interval_ms * 0.001)
	logging.info('speed (unit/s): %f' % speed)

	# Circle parameters
	#radius = 0.01
	color = 'red'


	# --- Setup the Figure and Axes ---
	fig, ax = plt.subplots()
	# Set the limits of the square plot
	ax.set_xlim(0, simulation_settings.x_end + 1)
	ax.set_ylim(-0.5, 0.5)
	ax.set_aspect('equal', adjustable='box')  # Ensure it looks square
	ax.set_title(f"Uniform Motion from ({simulation_settings.x_start}, {simulation_settings.y_start}) to ({simulation_settings.x_end}, {simulation_settings.y_end})")
	ax.grid(True, linestyle='--', alpha=0.6)

	# --- Create the initial artist (the circle patch) ---
	# A circle is a Patch object in matplotlib
	# It's initially placed at the starting position (x_start, y_start)
	circle = plt.Circle((simulation_settings.x_start, simulation_settings.y_start), simulation_settings.particle_radius, fc=color, ec='black', lw=1.0)
	ax.add_patch(circle)

	# The target
	circle_target = plt.Circle((1, 0), simulation_settings.target_radius, fc='blue', ec='black', lw=1.0)
	ax.add_patch(circle_target)


	# --- Initialization Function ---
	def init_func():
		"""Initializes the animation: sets the circle to the starting position."""
		circle.center = (simulation_settings.x_start, simulation_settings.y_start)
		return circle,


	# --- Animation/Update Function ---
	def animate(frame):
		"""
		Updates the position of the circle for each frame.
		'frame' is the current frame number, ranging from 0 to n_frames - 1.
		"""
		# Calculate the current time 't' corresponding to the frame number
		t = frame / simulation_settings.fps

		# Calculate the new x-position (linear interpolation)
		# x(t) = x_start + (x_end - x_start) * (t / total_time)
		# Since x_start = 0 and x_end = 1: x(t) = (1/5) * t
		x_now, y_now = circle.center

		dist = distance(circle, circle_target)
		# logging.info('distance(): %f' % dist)

		if dist <= simulation_settings.target_radius*2:
			x_new = -x_now
		else:
			x_new = simulation_settings.x_start + (simulation_settings.x_end - simulation_settings.x_start) * (t / simulation_settings.total_time)

		# The y-position is constant
		y_new = simulation_settings.y_start

		# Update the circle's center coordinates
		circle.center = (x_new, y_new)

		# Optional: Display current time in the title
		ax.set_title(f"Current Time: {t:.2f} s / {simulation_settings.total_time:.0f} s, Distance: {dist:.2f}")

		# Return the artist that was modified
		return circle,


	# --- Create the Animation ---
	# FuncAnimation(fig, func, frames, init_func, interval, blit)
	ani = animation.FuncAnimation(
		fig,
		animate,
		frames=n_frames,
		init_func=init_func,
		interval=interval_ms,
		blit=True,  # Use blitting for faster rendering (only redraws what changed)
		repeat=False  # The animation should stop after one cycle (5 seconds)
	)

	# --- Show the Plot ---
	# plt.show()

	writer = animation.FFMpegWriter(fps=30)
	# Optional: To save the animation (requires a writer like 'ffmpeg' or 'pillow')
	#ani.save('linear_motion.mp4', writer=writer, fps=fps)
	ani.save(f'./simulations/linear_motion.mp4', writer=writer, dpi=100)


if __name__ == "__main__":
	target_radius = 0.05
	particle_radius = 0.01
	y_0 = random.uniform(-target_radius, target_radius)
	simulationSettings = SimulationSettings( x_start=0.0,
											 y_start=y_0,
											 x_end=1.0,
											 y_end=y_0,
											 total_time=5,
											 target_radius=target_radius,
											 particle_radius=particle_radius,
											 fps=50)

	main(simulationSettings)
