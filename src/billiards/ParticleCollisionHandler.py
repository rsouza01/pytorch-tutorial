class ParticleCollisionHandler:
    def __init__(self, n_particles, radius):
        self.n_particles = n_particles
        self.radius = radius

    def __repr__(self):
        return f"ParticleCollisionHandler(n_particles={self.n_particles}, radius={self.radius})"
    