import argparse
from random import randint
from rubik.cube import Cube
from collections import Counter
import pandas as pd
import datetime


def cmd_lines():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scramble_min_moves',
                        type=int,
                        help="minmum number of moves in each scrambles",
                        default=20)
    parser.add_argument('--scramble_max_moves',
                        type=int,
                        help="maximum number of moves in each scrambles",
                        default=20)
    parser.add_argument('--n_scrambles',
                        type=int,
                        help="Number of scrambles allowed",
                        default=1024)
    return parser.parse_args()


def main(args):
    now = datetime.datetime.now()
    min_moves = args.scramble_min_moves
    max_moves = max(args.scramble_max_moves,args.scramble_min_moves)
    n = args.n_scrambles
    cube = Cube()
    facelets = []
    for _ in range(n):
        n_moves = randint(min_moves, max_moves)
        cube.scramble(n_moves, avoid_repeated=True)
        facelets.extend(cube.facelets)
        cube.reset()
    facelets_count = dict(Counter(facelets))
    total_facelets = len({facelet for facelet in facelets})
    data = {'facelet': [], 'prob': []}
    for facelet in facelets_count:
        prob = facelets_count[facelet] / total_facelets
        data['facelet'].append(facelet)
        data['prob'].append(prob)
    filename = f'results/{now}_scrambles={n}_min={min_moves}_max={max_moves}.csv'
    pd.DataFrame(data).sort_values('prob', ascending=False).to_csv(filename, index=False)


if __name__ == '__main__':
    main(cmd_lines())
