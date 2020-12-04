from itertools import islice
from collections import deque
from matplotlib import pyplot as plt
import matplotlib as mpl
import math
import numpy as np

from IPDL import MatrixBasedRenyisEntropy
from .InformationPlane import InformationPlane

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

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

def gen_log_space(limit: int, n: int) -> np.ndarray:
    '''
        code from: https://stackoverflow.com/questions/12418234/logarithmically-spaced-integers
    '''
    result = [1]
    if n>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)

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

    @staticmethod
    def gen_log_space(limit: int, n: int) -> np.ndarray:
        '''
            code from: https://stackoverflow.com/questions/12418234/logarithmically-spaced-integers
        '''
        result = [1]
        if n>1:  # just a check to avoid ZeroDivisionError
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
        while len(result)<n:
            next_value = result[-1]*ratio
            if next_value - result[-1] >= 1:
                # safe zone. next_value will be a different integer
                result.append(next_value)
            else:
                # problem! same integer. we need to find next_value by artificially incrementing previous value
                result.append(result[-1]+1)
                # recalculate the ratio so that the remaining values will scale correctly
                ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
        
        # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
        return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)

    @staticmethod
    def show_information_plane(ip: InformationPlane) -> mpl.figure.Figure:
        markers = "o^spdP*"
        cmap = mpl.cm.Blues
        reference = MatrixBasedRenyisEntropy.entropy(ip.get_input_matrix()).cpu()

        Ixt, Ity = ip.get_mi(moving_average_n=0)

        with plt.style.context('seaborn'):
            fig = plt.figure(constrained_layout=True, figsize=(16,8))
            gs1 = fig.add_gridspec(nrows=10, ncols=2, left=0.05, right=0.84, wspace=0.05, hspace=10)

            f8_ax1 = fig.add_subplot(gs1[0:9, 0])
            f8_ax1.set_title("Encoder")
            f8_ax1.set_xlabel("I(X, T)")
            f8_ax1.set_ylabel("I(T, Y)")
            f8_ax1.set(xlim=(0, reference), ylim=(0, reference))
            f8_ax1.plot([0, 1], [0, 1], transform=f8_ax1.transAxes, linestyle='dashed')

            for idx in range((len(Ixt)//2)+1):
                if idx == (len(Ixt)//2):
                    label = "Bottleneck"
                else:
                    label = "Encoder {}".format(idx+1)
                current_Ixt = np.array(Ixt[idx])
                current_Ity = np.array(Ity[idx])

                log_spaced = Utils.gen_log_space(len(current_Ixt), math.ceil(len(current_Ixt)*0.1))
                iterations = np.arange(len(log_spaced))

                f8_ax1.scatter(current_Ixt[log_spaced], current_Ity[log_spaced], c=iterations, vmin=0, vmax=iterations.max(), label=label, marker=markers[idx], cmap=cmap, edgecolors='black')
            f8_ax1.legend()

            f8_ax2 = fig.add_subplot(gs1[0:9, 1])
            f8_ax2.set_title("Decoder")
            f8_ax2.set_xlabel("I(X, T)")
            f8_ax2.set_ylabel("I(T, Y)")
            f8_ax2.yaxis.tick_right()
            f8_ax2.yaxis.set_label_position("right")
            f8_ax2.set(xlim=(0, reference), ylim=(0, reference))
            f8_ax2.plot([0, 1], [0, 1], transform=f8_ax2.transAxes, linestyle='dashed')

            decode_markers = markers[:idx+1]
            decode_markers = decode_markers[::-1]
            for marker_idx, idx in enumerate(range((len(Ixt)//2), len(Ixt))):
                if idx == (len(Ixt)//2):
                    label = "Bottleneck"
                else:
                    label = "Decoder {}".format(idx+1)
                current_Ixt = np.array(Ixt[idx])
                current_Ity = np.array(Ity[idx])
                log_spaced = Utils.gen_log_space(len(current_Ixt), math.ceil(len(current_Ixt)*0.1))

                marker = decode_markers[marker_idx]
                f8_ax2.scatter(current_Ixt[log_spaced], current_Ity[log_spaced], c=iterations, vmin=0, vmax=iterations.max(), label=label, marker=marker, cmap=cmap, edgecolors='black')
            
            f8_ax2.legend()

            f8_ax3 = fig.add_subplot(gs1[9, :])
            f8_ax3.set_title("Iterations")
            norm = mpl.colors.Normalize(vmin=0, vmax=len(current_Ixt))
            cb1 = mpl.colorbar.ColorbarBase(f8_ax3, cmap=cmap,
                                            norm=norm,
                                            orientation='horizontal')

        return fig

