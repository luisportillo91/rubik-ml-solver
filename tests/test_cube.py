from rubik.cube import Face, Cube
import numpy as np
from random import randint


def get_random():
    return randint(0, 255)


def compute_distance(f: Face):
    u_line = f.value[0, :, :]
    r_line = f.value[:, 2, :]
    d_line = f.value[2, :, :]
    l_line = f.value[:, 0, :]
    u_neighbor_line = f.transversal_faces[0].value[0, :, :]
    r_neighbor_line = f.transversal_faces[1].value[0, :, :]
    d_neighbor_line = f.transversal_faces[2].value[0, :, :]
    l_neighbor_line = f.transversal_faces[3].value[0, :, :]
    u_distance = np.linalg.norm(u_line - u_neighbor_line)
    r_distance = np.linalg.norm(r_line - r_neighbor_line)
    d_distance = np.linalg.norm(d_line - d_neighbor_line)
    l_distance = np.linalg.norm(l_line - l_neighbor_line)
    return np.round(u_distance + r_distance + d_distance + l_distance)


def test_rotate():
    u_color = (get_random(), get_random(), get_random())  # white
    r_color = (get_random(), get_random(), get_random())  # orange
    f_color = (get_random(), get_random(), get_random())  # blue
    d_color = (get_random(), get_random(), get_random())  # yellow
    l_color = (get_random(), get_random(), get_random())  # red

    f = Face(color=f_color, alias='f')
    u = Face(color=u_color, alias='u')
    r = Face(color=r_color, alias='r')
    d = Face(color=d_color, alias='d')
    l = Face(color=l_color, alias='l')

    f.set_transversals([u, r, d, l])
    original_total_distance = compute_distance(f)
    for i in range(4):
        f.rotate(1)
        total_distance = compute_distance(f)
        assert original_total_distance == total_distance

    for i in range(4):
        f.rotate(-1)
        total_distance = compute_distance(f)
        assert original_total_distance == total_distance


def test_scramble():
    cube = Cube()
    original_facelet = cube.get_facelet()
    n_moves = randint(8, 32)
    cube.scramble(n_moves)
    facelet = cube.get_facelet()
    assert facelet != original_facelet
    for move in cube.solution:
        cube.move(move)
    facelet = cube.get_facelet()
    assert facelet == original_facelet

    n_moves = 4096
    cube.reset()
    cube.scramble(n_moves)
    len(cube.facelets) != len({facelet for facelet in cube.facelets})

    cube.reset()
    cube.scramble(n_moves, avoid_repeated=True)
    len(cube.facelets) == len({facelet for facelet in cube.facelets})
