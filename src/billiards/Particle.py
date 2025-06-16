"""A module defining the Particle class for a billiards simulation."""

class Particle:
    """A class representing a particle in a billiards simulation."""
    def __init__(self, p_id, x, y, vx, vy, radius):
        self.p_id = p_id  # Unique identifier for the particle
        self.x = x  # x-coordinate of the particle
        self.y = y  # y-coordinate of the particle
        self.vx = (1 if x > 0.5 else -1) * 500 # x-component of velocity
        self.vy = 0  # y-component of velocity
        self.radius = radius  # radius of the particle
        self.color = 'red' if x < 0 else 'blue'

    def __repr__(self):
        """Return a string representation of the particle."""
        return f"Particle(x={self.x}, y={self.y}, vx={self.vx}, vy={self.vy}, radius={self.radius})"

    def distance_to(self, other):
        """Calculate the distance to another particle."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def collided_with(self, other):
        """Check if this particle has collided with another particle."""
        distance = ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
        return distance < (self.radius + other.radius)
