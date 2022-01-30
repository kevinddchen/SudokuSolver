import time
from typing import Tuple

# ------------------------------------------------------------------------------

class TimeEstimator(object):

    '''Utility class that estimates time remaining.'''

    def __init__(self, num_iters: int):
        self.start_t = time.time()
        self.t = time.time()
        self.N = num_iters
        self.ave_delta_t = 0

    def update(self) -> Tuple[float, float]:

        '''
        Updates progress and returns time since last update and estimated time
        remaining (in seconds).
        '''

        self.N -= 1
        delta_t, self.t = time.time() - self.t, time.time()
        if self.ave_delta_t != 0:
            self.ave_delta_t = .9*self.ave_delta_t + .1*delta_t     ## Exponential moving average
        else:
            self.ave_delta_t = delta_t
        remaining_t = self.N * self.ave_delta_t
        return delta_t, remaining_t

    def reset(self):
        '''Resets timer keeping track of last update.'''
        self.t = time.time()

    def total(self) -> float:
        '''Returns total elapsed time since creation.'''
        return time.time() - self.start_t