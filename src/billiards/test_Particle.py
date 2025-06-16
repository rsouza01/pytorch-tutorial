import pytest
from Particle import Particle

import logging

logger = logging.getLogger(__name__)

def test_default():
    # test_particle_distance()
    # test_particle_collision()
    test_particle_non_collision()

def test_particle_distance():
    p1 = Particle(id=1, x=0.6, y=0.2, vx=0, vy=0, radius=0.05)
    p2 = Particle(id=2, x=0.7, y=0.2, vx=0, vy=0, radius=0.05)

    logger.info(f"Particle 1: {p1}")
    logger.info(f"Particle 2: {p2}")
    logger.info(f"Distance p1-p2: {p1.distance_to(p2)}")
    logger.info(f"Distance p2-p1: {p2.distance_to(p1)}")

    assert p1.distance_to(p2)  == pytest.approx(0.0999999999)
    assert p2.distance_to(p1)  == pytest.approx(0.0999999999)


def test_particle_collision():
    particle_radius = 0.05
    p1 = Particle(id=1, x=0.6, y=0.2, vx=0, vy=0, radius=particle_radius)
    p2 = Particle(id=2, x=0.7, y=0.2, vx=0, vy=0, radius=particle_radius)

    logger.info(f"Particle 1: {p1}")
    logger.info(f"Particle 2: {p2}")
    logger.info(f"Collision distance < {p1.radius + p2.radius}")
    logger.info(f"Distance p1-p2: {p1.distance_to(p2)}")

    assert p1.collided_with(p2) == True

def test_particle_non_collision():
    particle_radius = 0.05
    p1 = Particle(id=1, x=0.0, y=0.0, vx=0, vy=0, radius=particle_radius)
    p2 = Particle(id=2, x=1, y=1, vx=0, vy=0, radius=particle_radius)

    logger.info(f"Particle 1: {p1}")
    logger.info(f"Particle 2: {p2}")
    logger.info(f"Collision distance < {p1.radius + p2.radius}")
    logger.info(f"Distance p1-p2: {p1.distance_to(p2)}")

    assert p1.collided_with(p2) == False

def test_init_x_greater_than_half():
    p = Particle(id=1, x=0.6, y=0.2, vx=0, vy=0, radius=5)
    assert p.x == 0.6
    assert p.y == 0.2
    assert p.vx == 500
    assert p.vy == 0
    assert p.radius == 5
    assert p.color == 'blue'

def test_init_x_less_than_half():
    p = Particle(id=2, x=0.4, y=0.3, vx=0, vy=0, radius=3)
    assert p.x == 0.4
    assert p.y == 0.3
    assert p.vx == -500
    assert p.vy == 0
    assert p.radius == 3
    assert p.color == 'blue'

def test_init_x_negative_color_red():
    p = Particle(id=3, x=-0.1, y=0.0, vx=0, vy=0, radius=2)
    assert p.color == 'red'
    assert p.vx == -500  # since x < 0.5

def test_repr():
    p = Particle(id=4, x=0.7, y=0.8, vx=0, vy=0, radius=1)
    s = repr(p)
    assert "Particle(x=0.7, y=0.8, vx=500, vy=0, radius=1)" in s