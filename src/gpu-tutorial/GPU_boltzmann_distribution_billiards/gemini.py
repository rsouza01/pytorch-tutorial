#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import os
import numpy as np
from matplotlib.animation import PillowWriter
import scienceplots
plt.style.use(['science', 'notebook'])

import logging
import time

logging.basicConfig(level=logging.INFO)

def main():
	# --- Parameters ---
	# Start position (x0, y0)
	x_start, y_start = 0.0, 1.0

	# End position (x_end, y_end)
	x_end, y_end = 1.0, 1.0

	# Total animation time in seconds (T)
	total_time = 5.0

	# Number of frames (N) and frame rate (FPS)
	# Using a common frame rate of 50 FPS for smooth motion
	fps = 50
	n_frames = int(total_time * fps)
	logging.info('n_frames: %d' % n_frames)

	# Interval between frames in milliseconds (1000/FPS)
	interval_ms = 1000 / fps
	logging.info('interval_ms: %d' % interval_ms)

	distance_x = x_end - x_start
	logging.info('Distance: %f' % distance_x)

	speed = distance_x/(interval_ms * 0.001)
	logging.info('speed (unit/s): %f' % speed)

	# Circle parameters
	radius = 0.05
	color = 'red'


	# --- Setup the Figure and Axes ---
	fig, ax = plt.subplots()
	# Set the limits of the square plot
	ax.set_xlim(-0.5, 1.5)  # Extend slightly beyond [0, 1] for visibility
	ax.set_ylim(-0.5, 1.5)
	ax.set_aspect('equal', adjustable='box')  # Ensure it looks square
	ax.set_title(f"Uniform Motion from ({x_start}, {y_start}) to ({x_end}, {y_end})")
	ax.grid(True, linestyle='--', alpha=0.6)

	# --- Create the initial artist (the circle patch) ---
	# A circle is a Patch object in matplotlib
	# It's initially placed at the starting position (x_start, y_start)
	circle = plt.Circle((x_start, y_start), radius, fc=color, ec='black', lw=1.0)
	ax.add_patch(circle)


	# --- Initialization Function ---
	def init_func():
		"""Initializes the animation: sets the circle to the starting position."""
		circle.center = (x_start, y_start)
		return circle,


	# --- Animation/Update Function ---
	def animate(frame):
		"""
		Updates the position of the circle for each frame.
		'frame' is the current frame number, ranging from 0 to n_frames - 1.
		"""
		# Calculate the current time 't' corresponding to the frame number
		t = frame / fps

		# Calculate the new x-position (linear interpolation)
		# x(t) = x_start + (x_end - x_start) * (t / total_time)
		# Since x_start = 0 and x_end = 1: x(t) = (1/5) * t
		x_new = x_start + (x_end - x_start) * (t / total_time)

		# The y-position is constant
		y_new = y_start

		# Update the circle's center coordinates
		circle.center = (x_new, y_new)

		# Optional: Display current time in the title
		ax.set_title(f"Current Time: {t:.2f} s / {total_time:.0f} s")

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
	start_time = time.time()
	main()
	print("--- %s seconds ---" % (round(time.time() - start_time, 2)))
