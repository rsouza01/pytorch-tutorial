import pytest
from Particle import Particle

# src/billiards/test_Particle.py

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