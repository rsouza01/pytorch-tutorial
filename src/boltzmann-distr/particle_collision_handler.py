"""A module for handling particle collisions in a billiards simulation."""

class ParticleCollisionHandler:
    """
    A class to handle particle collisions in a billiards simulation.
    This class is responsible for managing the number of particles and their collision radius.
    """

    def __init__(self, n_particles, radius):
        """
        Initialize the ParticleCollisionHandler with the number of particles and their radius.
        Args:
            n_particles (int): Number of particles in the simulation.
            radius (float): Radius of the particles.
        """
        self.n_particles = n_particles
        self.radius = radius

    def __repr__(self):
        return f"ParticleCollisionHandler(n_particles={self.n_particles}, radius={self.radius})"

    def get_collision_radius(self):
        """
        Get the collision radius for the particles.
        The collision radius is defined as the sum of the radii of two particles.
        Returns:
            float: The collision radius.
        """
        return 2 * self.radius
