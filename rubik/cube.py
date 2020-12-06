import numpy as np
from typing import List
from random import randint


class Face:
    def __init__(self, color, alias):
        self.color = color
        self.alias = alias
        self.value = np.array([[self.color] * 3 for _ in range(3)])
        self.transversal_faces = []
        self.reset_flag = True

    def set_transversals(self, transversal_faces: List):
        self.transversal_faces = transversal_faces

    def rotate(self, direction):
        if self.transversal_faces is None:
            raise Exception("You must first set the transversal facelets")
        if direction == 1:
            self.value = np.rot90(self.value, axes=(1, 0))
            last_face = self.transversal_faces[3].value[0, :, :].copy()
            for i in range(3, 0, -1):
                self.transversal_faces[i].value[0, :, :] = self.transversal_faces[i - 1].value[0, :, :].copy()
            self.transversal_faces[0].value[0, :, :] = last_face.copy()

        elif direction == -1:
            self.value = np.rot90(self.value, -1, axes=(1, 0))
            last_face = self.transversal_faces[0].value[0, :, :].copy()
            for i in range(3):
                self.transversal_faces[i].value[0, :, :] = self.transversal_faces[i + 1].value[0, :, :].copy()
            self.transversal_faces[3].value[0, :, :] = last_face.copy()
        else:
            raise Exception("Wrong direction")
        for face in self.transversal_faces:
            face.reset_flag = False
        self.reset_flag = False

    def reset(self):
        if not self.reset_flag:
            self.value = np.array([[self.color] * 3 for _ in range(3)])
            self.reset_flag = True
            for face in self.transversal_faces:
                face.reset()


class Cube:
    def __init__(self,
                 u_color=(255, 255, 255),
                 r_color=(255, 165, 0),
                 f_color=(0, 0, 255),
                 d_color=(255, 255, 0),
                 l_color=(255, 0, 0),
                 b_color=(0, 255, 0)
                 ):

        self.colors_lookup = {
            u_color: 'u',
            r_color: 'r',
            f_color: 'f',
            d_color: 'd',
            l_color: 'l',
            b_color: 'b'
        }
        self.F = Face(color=f_color, alias='f')
        self.U = Face(color=u_color, alias='u')
        self.R = Face(color=r_color, alias='r')
        self.D = Face(color=d_color, alias='d')
        self.L = Face(color=l_color, alias='l')
        self.B = Face(color=b_color, alias='b')

        # Assign the facelets in clockwise order the front value
        self.F.set_transversals([self.U, self.R, self.D, self.L])
        self.U.set_transversals([self.B, self.R, self.F, self.L])
        self.R.set_transversals([self.U, self.B, self.D, self.F])
        self.D.set_transversals([self.F, self.R, self.B, self.L])
        self.L.set_transversals([self.U, self.F, self.D, self.B])
        self.B.set_transversals([self.U, self.L, self.D, self.R])

        self.move_funcs = {
            "F": lambda: self.F.rotate(1),
            "F'": lambda: self.F.rotate(-1),
            "U": lambda: self.U.rotate(1),
            "U'": lambda: self.U.rotate(-1),
            "R": lambda: self.R.rotate(1),
            "R'": lambda: self.R.rotate(-1),
            "D": lambda: self.D.rotate(1),
            "D'": lambda: self.D.rotate(-1),
            "L": lambda: self.L.rotate(1),
            "L'": lambda: self.L.rotate(-1),
            "B": lambda: self.B.rotate(1),
            "B'": lambda: self.B.rotate(-1)
        }
        self.valid_moves = ["F", "F'", "U", "U'", "R", "R'", "D", "D'", "L", "L'", "B", "B'"]
        self.facelets = [self.get_facelet()]
        self.solution = []

    def reset(self):
        self.F.reset()
        del self.facelets[1:]
        del self.solution[:]

    def move(self, move):
        self.move_funcs[move]()

    def _get_facelet(self, face):
        return [self.colors_lookup[tuple(face.value[i, j, :])] for i in range(3) for j in range(3)]

    def get_facelet(self):
        """
        Return the cube order
        """
        f_facelet = self._get_facelet(self.F)
        u_facelet = self._get_facelet(self.U)
        r_facelet = self._get_facelet(self.R)
        d_facelet = self._get_facelet(self.D)
        l_facelet = self._get_facelet(self.L)
        b_facelet = self._get_facelet(self.B)
        facelet = f_facelet + u_facelet + r_facelet + d_facelet + l_facelet + b_facelet
        return ''.join(facelet)

    def scramble(self, n_moves: int, avoid_repeated: bool = False):
        """
        Scramble the cube with n_moves

        Args:
            avoid_repeated (bool): Exclude repeated facelets during the scramble
            n_moves (int): number of moves performed in the scramble
        """
        for i in range(n_moves):
            move = self.valid_moves[randint(0, 11)]
            self.move_funcs[move]()
            facelet = self.get_facelet()
            if avoid_repeated:
                while facelet in self.facelets:
                    undo_move = move.replace("'", '') if move[-1] == "'" else f"{move}'"
                    self.move(undo_move)
                    move = self.valid_moves[randint(0, 11)]
                    self.move(move)
                    facelet = self.get_facelet()
            self.solution.insert(0, move.replace("'", '') if move[-1] == "'" else move + "'")
            self.facelets.append(facelet)
