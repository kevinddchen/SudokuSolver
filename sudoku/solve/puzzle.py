from __future__ import annotations

from typing import Collection, Tuple

import numpy as np

# ------------------------------------------------------------------------------

'''
Note: `mode` refers to how the information in the puzzle is organized. When
mode=0, the major index (i.e. the first index) refers to a row. When mode=1, the
major index refers to a column. When mode=2, the major index refers to a 3 x 3
block of squares. The three modes are illustrated below:

    000000000       012345678       000111222
    111111111       012345678       000111222
    222222222       012345678       000111222
    333333333       012345678       333444555
    444444444       012345678       333444555
    555555555       012345678       333444555
    666666666       012345678       666777888
    777777777       012345678       666777888
    888888888       012345678       666777888
    
    mode=0          mode=1          mode=2
'''

# ------------------------------------------------------------------------------

class PuzzleInputError(Exception):
    pass

class PuzzleInconsistency(Exception):
    pass

# ------------------------------------------------------------------------------

class Puzzle(object):

    '''
    Class for storing Sudoku puzzles. Use `Puzzle.fromArray()` to initialize from
    an array.

    Params:
        _grid (ndarray<uint8>): 9x9 array representing the puzzle.
        _hints (dict): Keys are coordinates (i, j) where _grid[i, j] is not yet
            filled-in. _hints[(i, j)] is a set corresponding to possible values
            of _grid[i, j].
        _news (list): Stack of (i, j) coordinates of newly filled-in squares.
    '''

    def __init__(self):
        self._grid = None
        self._hints = None
        self._news = None
            
    @classmethod
    def fromArray(cls, array: np.ndarray) -> Puzzle:

        '''
        Initialize from array. Can throw `PuzzleInputError` and
        `PuzzleInconsistency`.

        Args:
            array (ndarray): 9x9 array of integers between 0 and 9 (inclusive),
                where 0 represents an empty square.

        Returns:
            (Puzzle): A class instance.
        '''

        obj = cls()

        ## Initialize grid
        obj._validate_input(array)
        obj._grid = array.astype(np.uint8)
        obj._validate_grid()

        ## Initialize hints and news
        obj._hints = {}
        obj._news = []
        for i, row in enumerate(obj._grid):
            for j, el in enumerate(row):
                if el == 0:
                    obj._hints[(i, j)] = set(range(1, 10))
                else:
                    obj._news.append((i, j))
        return obj

    def __repr__(self) -> str:
        return str(self._grid)

    def copy(self) -> Puzzle:
        '''Returns copy of puzzle.'''
        obj = Puzzle()
        obj._grid = self._grid.copy()
        obj._hints = {c: h.copy() for c, h in self._hints.items()}
        obj._news = self._news.copy()
        return obj

    def is_solved(self) -> bool:
        '''Returns True if puzzle is solved.'''
        return len(self._hints) == 0 and self.is_consistent()

    def is_consistent(self) -> bool:
        '''Returns True if puzzle is consistent so far.'''
        try:
            self._validate_grid()
            return True
        except PuzzleInconsistency:
            return False

    def reduce(self):
        ''' 
        Use simple logic to fill in squares. This is not guaranteed to solve the
        puzzle. Will throw `PuzzleInconsistency` if the puzzle is found to be
        inconsistent.
        '''
        def search_for_unique():
            for mode in range(3):
                for major_ix in range(9):
                    ignore = set()      ## Tracks filled-in values and duplicate hints
                    unique = dict()     ## Tracks unique hints
                    for c in get_major(mode, major_ix):
                        ## If not yet filled in
                        if self._grid[c] == 0:
                            for x in self._hints[c]:
                                if x in ignore:
                                    pass
                                elif x in unique:
                                    ignore.add(x)
                                    del unique[x]
                                else:
                                    unique[x] = c 
                        ## If already filled in
                        else:
                            x = self._grid[c]
                            ignore.add(x)
                            if x in unique:
                                del unique[x] 
                    for h, c in unique.items():
                        self._fill_in(c, h)
                        return

        while self._news:
            ## First, eliminate hints. If a square is encountered that only has
            ## one hint left, fill in the square.
            c1 = self._news.pop()
            el = self._grid[c1]
            for mode in range(3):
                major_ix = get_major_ix(mode, c1)
                for c2 in get_major(mode, major_ix):
                    ## If not yet filled in, eliminate hint
                    if self._grid[c2] == 0 and el in self._hints[c2]:
                        self._hints[c2].remove(el)
                        ## Check if only one hint left
                        if len(self._hints[c2]) == 1:
                            self._fill_in(c2, self._hints[c2].pop())
                    ## If already filled in, if equal then inconsistent
                    elif self._grid[c2] == el and c1 != c2:
                        err_msg = f'Found inconsistent entry {el} at ({c2[0]}, {c2[1]}).'
                        raise PuzzleInconsistency(err_msg)
            
            ## Second, look in each major and see if there are any squares with a
            ## unique hint. Each time a square is filled in, it is crucial to
            ## update `_hints`.
            search_for_unique()

    def _fill_in(self, coord: Tuple[int, int], n: int):
        '''Fill in a square of puzzle.'''
        self._grid[coord] = n
        self._news.append(coord)
        del self._hints[coord]

    @staticmethod
    def _validate_input(arr: np.ndarray):
        '''Basic input validation. Can throw `PuzzleInputError`.'''
        if not (isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.integer) and arr.shape == (9, 9)):
            err_msg = 'Expected a 9x9 numpy array of integers.'
            raise PuzzleInputError(err_msg)
        for i, row in enumerate(arr):
            for j, el in enumerate(row):
                if el < 0 or el > 9:
                    err_msg = f'Invalid entry at ({i}, {j}). Got {el}, but expected an integer between 0 and 9 (inclusive).'
                    raise PuzzleInputError(err_msg)

    def _validate_grid(self):
        '''Check if puzzle is consistent. Can throw `PuzzleInconsistency`.'''
        for mode in range(3):
            for major_ix in range(9):
                filled = set()      ## Tracks filled-in values
                for coord in get_major(mode, major_ix):
                    el = self._grid[coord]
                    if el > 0:
                        if el in filled:
                            err_msg = f'Invalid puzzle. Duplicate entry {el} at ({coord[0]}, {coord[1]}).'
                            raise PuzzleInconsistency(err_msg)
                        filled.add(el)

# ------------------------------------------------------------------------------

def get_major_ix(mode: int, coord: Tuple[int, int]) -> int:
    '''Get the major index of a coordinate (i, j) in a particular mode.'''
    if mode == 0:
        return coord[0]
    if mode == 1:
        return coord[1]
    if mode == 2:
        return 3 * (coord[0] // 3) + (coord[1] // 3)
    else:
        return None

# ------------------------------------------------------------------------------

def get_major(mode: int, major_ix: int) -> Collection[Tuple[int, int]]:
    '''Get the nine coordinates of a major.'''
    if mode == 0:
        irange = range(major_ix, major_ix+1)
        jrange = range(9)
    elif mode == 1:
        irange = range(9)
        jrange = range(major_ix, major_ix+1)
    else:
        imin = 3*(major_ix // 3)
        jmin = 3*(major_ix % 3)
        irange = range(imin, imin+3)
        jrange = range(jmin, jmin+3)
    for i in irange:
        for j in jrange:
            yield (i, j)
