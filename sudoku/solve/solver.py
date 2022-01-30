import heapq
from dataclasses import dataclass, field
from typing import Tuple

from .puzzle import Puzzle, PuzzleInconsistency

# ------------------------------------------------------------------------------

@dataclass(order=True)
class _PuzzlePriority(object):
    '''
    Puzzles are sorted by the minimum number of hints for a single square, and
    then by the number of empty squares. Some other fields are included for
    convenience when solving the puzzle.
    '''
    min_hints: int
    num_empty: int
    puzzle: Puzzle = field(compare=False)
    min_coord: Tuple[int, int] = field(compare=False)

# ------------------------------------------------------------------------------

def _get_priority(puzzle: Puzzle) -> _PuzzlePriority:
    '''Get priority for a puzzle.'''
    min_hints = None
    min_coord = None
    for c, h in puzzle._hints.items():
        if min_hints is None or len(h) < min_hints:
            min_hints = len(h)
            min_coord = c
    num_empty = len(puzzle._hints)
    pp = _PuzzlePriority(min_hints, num_empty, puzzle, min_coord)
    return pp

# ------------------------------------------------------------------------------

def solve(puzzle: Puzzle) -> Puzzle:

    '''
    Solve the puzzle by repeatedly reducing and making guesses. This is
    guaranteed to find a solution, if one exists. Otherwise, will throw
    `SudokuInconsistency` if there is no solution.
    
    Guesses are kept track of with a priority queue, and guesses are always
    make for hints with the fewest number of possibilites. When a puzzle is
    inconsistent (i.e. an earlier guess was incorrect), it is discarded.
    '''

    puzzle.reduce()
    if puzzle.is_solved():
        return puzzle

    heap = [_get_priority(puzzle)]
    while heap:
        priority = heapq.heappop(heap)
        parent = priority.puzzle
        coord = priority.min_coord
        poss = parent._hints[coord]
        for n in poss:
            child = parent.copy()
            child._fill_in(coord, n)    ## Make guess in child
            try:
                child.reduce()
            except PuzzleInconsistency:
                continue
            if child.is_solved():
                return child
            heapq.heappush(heap, _get_priority(child))
            
    err_msg = 'Puzzle has no solution.'
    raise PuzzleInconsistency(err_msg)
