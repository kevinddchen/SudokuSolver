from time import time
from typing import Iterator

import numpy as np

from sudoku.solve import Puzzle, solve

# ------------------------------------------------------------------------------

'''
Test sudoku solver on 100 puzzles taken from
https://projecteuler.net/problem=96.

Usage:
    `python test_sudoku.py`
'''

# ------------------------------------------------------------------------------

def read_puzzles(filename: str = 'assets/raw_puzzles.txt') -> Iterator[np.ndarray]:
    '''Read sudoku puzzles from file.'''
    with open(filename) as f:
        while f.readline():     ## Simultaneously checks for EOF and removes header
            arr = []
            for _ in range(9):
                x = f.readline()
                x = x.strip('\n')
                x = [int(c) for c in x]
                arr.append(x)
            yield np.array(arr, dtype=np.uint8)

# ------------------------------------------------------------------------------

if __name__ == '__main__':

    S = 0   ## Keep track of sum of the 3-digit number in the top-left corner of each puzzle
    t = time()

    for i, arr in enumerate(read_puzzles()):
        puzzle = Puzzle.fromArray(arr)
        puzzle = solve(puzzle)
        assert puzzle.is_solved(), f'Grid {i+1} not solved.'

        S += puzzle._grid[0, 0]*100 + puzzle._grid[0, 1]*10 + puzzle._grid[0, 2]

    assert S == 24702, f'Sum {S} does not match expected value.'

    print('Test passed.')
    print(f'Total time: {time()-t:.3f} sec.')
