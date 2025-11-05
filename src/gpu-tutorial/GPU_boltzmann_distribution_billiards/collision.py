#!/usr/bin/env python3


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# --- Constants and Setup ---
TARGET_RADIUS = 5.0
PARTICLE_RADIUS = 0.5
INITIAL_SPEED = 50.0  # Initial speed in the x-direction
INITIAL_X_OFFSET = -50.0  # Particles start far to the left
NUM_PARTICLES = 150

DT = 0.05  # Time step for simulation
ANIMATION_INTERVAL_MS = DT * 1000  # Milliseconds per frame for Matplotlib animation
TIME_DELAY_BETWEEN_SHOTS = 0.5  # Wait 2.0 seconds (simulation time) between launches


# --- Classes for Particles and Target ---
class Target:
	def __init__(self, position, radius, mass=np.inf):
		self.r = np.array(position, dtype=float)
		self.m = mass
		self.radius = radius


class Particle:
	def __init__(self, initial_position, initial_velocity, mass, radius, color='blue'):
		self.r = np.array(initial_position, dtype=float)
		self.v = np.array(initial_velocity, dtype=float)
		self.m = mass
		self.radius = radius
		self.color = color
		self.scattered = False  # Flag to track if particle has scattered

		# Store historical positions for plotting the trajectory
		self.trajectory = [self.r.copy()]

	def update_position(self, dt):
		"""Updates particle position based on its velocity and time step."""
		self.r += self.v * dt
		self.trajectory.append(self.r.copy())

	def handle_collision(self, target):
		"""Calculates new velocity after elastic collision with a stationary, infinite mass target."""
		if not self.scattered:  # Only scatter once
			# Vector from target center to particle center at collision
			r_vec = self.r - target.r
			distance = np.linalg.norm(r_vec)

			# Check for collision (should have happened slightly before due to discrete time steps,
			# but we assume the closest approach is the collision point)
			if distance <= (self.radius + target.radius):
				# Unit normal vector at collision point (from target to particle)
				n = r_vec / distance

				# Calculate reflected velocity: v' = v - 2 * (v . n) * n
				self.v = self.v - 2 * np.dot(self.v, n) * n
				self.scattered = True  # Mark as scattered
				print(f"Particle collided! New velocity: {self.v}")

	def is_out_of_bounds(self, x_lim, y_lim):
		"""Checks if particle has moved outside the animation view."""
		return not (x_lim[0] < self.r[0] < x_lim[1] and \
					y_lim[0] < self.r[1] < y_lim[1])


# Setup Particles and Target
target = Target(position=[0.0, 0.0], radius=TARGET_RADIUS)
particles = []
colors = ['red', 'green', 'blue', 'purple', 'orange']
current_sim_time = 0.0  # Initialize simulation time

for i in range(NUM_PARTICLES):
	# Calculate launch time for the current particle
	launch_time = i * TIME_DELAY_BETWEEN_SHOTS

	# Random initial y-position (impact parameter 'b') within target's effective radius
	# For simplicity, let's assume particles collide with the *center* of the target
	# if their y-position is within the target's radius.
	# A more rigorous check involves the sum of radii for collision.

	# For initial y position: random value between -(target_radius - particle_radius) and (target_radius - particle_radius)
	# This ensures the particle's center can actually pass through the target's visual bounds.
	effective_max_y = TARGET_RADIUS + PARTICLE_RADIUS
	initial_y = np.random.uniform(-effective_max_y, effective_max_y)

	initial_position = [INITIAL_X_OFFSET, initial_y]
	initial_velocity = [INITIAL_SPEED, 0.0]  # Pure x-direction velocity initially

	p = Particle(initial_position, initial_velocity, mass=1.0, radius=PARTICLE_RADIUS, color=colors[i % len(colors)])

	p.launch_time = launch_time
	p.active = False  # Start inactive
	particles.append(p)

# --- Matplotlib Animation Setup ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(INITIAL_X_OFFSET - 10, 50)  # Extend x-limit to show particles scattering away
ax.set_ylim(-TARGET_RADIUS * 3, TARGET_RADIUS * 3)  # Sufficient y-limit
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Elastic Scattering Simulation")
ax.grid(True)

# Plot the target
target_circle = plt.Circle(target.r, target.radius, color='gray', alpha=0.6, label='Target')
ax.add_patch(target_circle)

# Create particle plot objects (circles and trajectory lines)
particle_artists = []
trajectory_lines = []
for p in particles:
	circle = plt.Circle(p.r, p.radius, color=p.color, ec='black', lw=0.5, label=f'Particle {particles.index(p) + 1}')
	ax.add_patch(circle)
	particle_artists.append(circle)

	line, = ax.plot([], [], color=p.color, linestyle='--', linewidth=1)
	trajectory_lines.append(line)

# Keep track of which particles are still "active" for the animation loop
active_particles = list(particles)
finished_particles_count = 0  # Track how many have gone off screen

# --- Matplotlib Setup (unchanged) ---

# Global variable to track elapsed simulation time
elapsed_time = 0.0


# (The global 'active_particles' list is no longer strictly necessary,
# as we will iterate over *all* particles and check the 'active' flag.)

def animate(frame):
	global finished_particles_count, elapsed_time

	# 1. Advance the simulation time
	elapsed_time += DT

	new_active_particles = []

	for i, p in enumerate(particles):  # Iterate over ALL particles now

		# 2. Check Launch Condition
		if elapsed_time >= p.launch_time and not p.active:
			p.active = True
			# Make the particle visible when it launches
			particle_artists[i].set_visible(True)
			trajectory_lines[i].set_visible(True)

		# 3. Only update the particle if it is active (i.e., launched)
		if p.active:
			p.update_position(DT)
			p.handle_collision(target)

			# Update particle circle position
			particle_artists[i].set_center(p.r)

			# Update trajectory line
			traj_data = np.array(p.trajectory)
			trajectory_lines[i].set_data(traj_data[:, 0], traj_data[:, 1])

			# Check if particle is out of bounds
			if p.is_out_of_bounds(ax.get_xlim(), ax.get_ylim()):
				# This particle has finished its journey
				p.active = False  # Mark as inactive for the next frame
				finished_particles_count += 1
				particle_artists[i].set_visible(False)
				trajectory_lines[i].set_visible(False)
		else:
			# When inactive (before launch or after finishing), ensure it is off-screen/hidden
			particle_artists[i].set_visible(False)
			trajectory_lines[i].set_visible(False)

	# End condition for animation:
	if finished_particles_count == NUM_PARTICLES:
		print("All particles have finished their simulation.")
		ani.event_source.stop()

	return particle_artists + trajectory_lines


# Create the animation
ani = FuncAnimation(fig, animate, frames=range(2000),  # Max frames to prevent infinite loop
					interval=ANIMATION_INTERVAL_MS, blit=True, repeat=False)

# Optional: To save the animation (requires ffmpeg or imagemagick)
# ani.save('elastic_scattering.gif', writer='imagemagick', fps=30)
ani.save('./simulations/elastic_scattering.mp4', writer='ffmpeg', fps=30)

# plt.legend()
# plt.show()
