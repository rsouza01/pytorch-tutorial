"""Test cases for the Particle class in a billiards simulation."""

import logging
import pytest
from particle import Particle


logger = logging.getLogger(__name__)

# def test_default():
#     """
#     Test the default behavior of the Particle class.
#     """
#     # test_particle_distance()
#     # test_particle_collision()
#     test_particle_non_collision()

def test_particle_distance():
    """
    Test the distance calculation between two particles.
    """
    p1 = Particle(p_id=1, x=0.6, y=0.2, vx=0, vy=0, radius=0.05)
    p2 = Particle(p_id=2, x=0.7, y=0.2, vx=0, vy=0, radius=0.05)

    logger.info("Particle 1: %s", p1)
    logger.info("Particle 2: %s", p2)
    logger.info("Distance p1-p2: %s", p1.distance_to(p2))
    logger.info("Distance p2-p1: %s", p2.distance_to(p1))

    assert p1.distance_to(p2)  == pytest.approx(0.0999999999)
    assert p2.distance_to(p1)  == pytest.approx(0.0999999999)


def test_particle_collision():
    """
    Test the collision detection between two particles.
    """
    particle_radius = 0.05
    p1 = Particle(p_id=1, x=0.6, y=0.2, vx=0, vy=0, radius=particle_radius)
    p2 = Particle(p_id=2, x=0.7, y=0.2, vx=0, vy=0, radius=particle_radius)

    logger.info("Particle 1: %s", p1)
    logger.info("Particle 2: %s", p2)
    logger.info("Collision distance < %s", p1.radius + p2.radius)
    logger.info("Distance p1-p2: %s", p1.distance_to(p2))

    assert p1.collided_with(p2) is True

def test_particle_non_collision():
    """
    Test that two particles do not collide when they are far apart.
    """
    logger.info("Testing non-collision between two particles.")
    particle_radius = 0.05
    p1 = Particle(p_id=1, x=0.0, y=0.0, vx=0, vy=0, radius=particle_radius)
    p2 = Particle(p_id=2, x=1, y=1, vx=0, vy=0, radius=particle_radius)

    logger.info("Particle 1: %s", p1)
    logger.info("Particle 2: %s", p2)
    logger.info("Collision distance < %s", p1.radius + p2.radius)
    logger.info("Distance p1-p2: %s", p1.distance_to(p2))

    assert p1.collided_with(p2) is False

def test_init_x_greater_than_half():
    """
    Test the initialization of a Particle with x greater than 0.5.
    """
    p = Particle(p_id=1, x=0.6, y=0.2, vx=0, vy=0, radius=5)
    assert p.x == 0.6
    assert p.y == 0.2
    assert p.vx == 500
    assert p.vy == 0
    assert p.radius == 5
    assert p.color == 'blue'

def test_init_x_less_than_half():
    """
    Test the initialization of a Particle with x less than 0.5.
    """
    p = Particle(p_id=2, x=0.4, y=0.3, vx=0, vy=0, radius=3)
    assert p.x == 0.4
    assert p.y == 0.3
    assert p.vx == -500
    assert p.vy == 0
    assert p.radius == 3
    assert p.color == 'blue'

def test_init_x_negative_color_red():
    """
    Test the initialization of a Particle with x less than 0 and color red.
    """
    p = Particle( p_id=3, x=-0.1, y=0.0, vx=0, vy=0, radius=2)
    assert p.color == 'red'
    assert p.vx == -500  # since x < 0.5

def test_repr():
    """
    Test the string representation of a Particle.
    """
    p = Particle(p_id=4, x=0.7, y=0.8, vx=0, vy=0, radius=1)
    s = repr(p)
    assert "Particle(x=0.7, y=0.8, vx=500, vy=0, radius=1)" in s
