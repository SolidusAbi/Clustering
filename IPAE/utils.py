from itertools import islice
from collections import deque

class Utils():
    @staticmethod
    def sliding_window_iter(iterable, size):
        '''
            Iterate through iterable using a sliding window of several elements.
            Important: It is a generator!.
            
            Creates an iterable where each element is a tuple of `size`
            consecutive elements from `iterable`, advancing by 1 element each
            time. For example:

            >>> list(sliding_window_iter([1, 2, 3, 4], 2))
            [(1, 2), (2, 3), (3, 4)]
            
            source: https://codereview.stackexchange.com/questions/239352/sliding-window-iteration-in-python
        '''
        iterable = iter(iterable)
        window = deque(islice(iterable, size), maxlen=size)
        for item in iterable:
            yield tuple(window)
            window.append(item)
        if window:  
            # needed because if iterable was already empty before the `for`,
            # then the window would be yielded twice.
            yield tuple(window)
