from abc import ABC, abstractmethod
from jllutils import miscutils
import numpy as np

class PlumeDayFilter(ABC):
    """Base class for determining if a given set of columns has a plume in it.
    """
    @abstractmethod
    def day_has_plume(self, columns: np.ndarray) -> bool:
        """Return True if the columns given contain a plume
        """
        pass

    @abstractmethod
    def description(self) -> str:
        pass


class QuantilePlumeDayFilter(PlumeDayFilter):
    """Filter that detects plumes by the difference between two quantiles in the scene.

    The defaults (1st to 99th percentile, minimum delta of 5e-5) were chosen for NO2.
    For other gases, the minimum delta would need modified. That is the minimum difference
    between the given quantiles ``q`` a scene must have to be considered to have a plume.
    """
    def __init__(self, min_delta=5e-5, q=[0.01, 0.99]):
        self.min_delta = min_delta
        self.q = q

    def day_has_plume(self, columns: np.ndarray) -> bool:
        if hasattr(columns, 'mask'):
            columns = columns.filled(np.nan)
        v1, v2 = np.nanquantile(columns, self.q)
        return bool((v2 - v1) >= self.min_delta)

    def description(self) -> str:
        q1, q2 = self.q
        return f'if {q1*100:.0f} to {q2*100:.0f} percentile difference exceeds {self.min_delta:.5g}'


class PlumeStartSelector(ABC):
    """Base class for determining which grid cells to start the floodfill algorithm from.
    """
    @abstractmethod
    def get_starting_indices(self, columns: np.ndarray) -> tuple[tuple[int]]:
        """Return the indices of ``columns`` from which to begin the floodfill algorithm.

        ``columns`` is expected to be 2D. The returned value will be a sequence of
        two-element tuples that index columns. That is, if the return value is ``xx``,
        then ``columns[xx[0]]`` is the first grid cell to start from, ``columns[xx[1]]``
        the second, and so on.
        """
        pass

    def description(self) -> str:
        pass


class QuantilePlumeStartSelector(PlumeStartSelector):
    """Selector that identifies grid cells to floodfill from based on a quantile.

    This will identify grid cells to start from as those that have a column amount
    above the given quantile, ``q_threshold``. The default of 0.995 will therefore
    take the grid cells with the highest 0.5% of column amounts.
    """
    def __init__(self, q_threshold=0.995):
        self.q_threshold = q_threshold

    def get_starting_indices(self, columns: np.ndarray) -> tuple[tuple[int]]:
        if hasattr(columns, 'mask'):
            columns = columns.filled(np.nan)

        orig_shape = columns.shape
        columns = columns.ravel()
        sorted_inds = np.argsort(columns, descending=True)
        sorted_vals = columns[sorted_inds]
        threshold_val = np.nanquantile(columns, self.q_threshold)
        vv = sorted_vals >= threshold_val
        ii, jj = np.unravel_index(sorted_inds[vv], orig_shape)
        return tuple((int(i),int(j)) for i,j in zip(ii,jj))

    def description(self) -> str:
        return f'grid cells above the {self.q_threshold*100:.0f} percentile'


class PlumeThreshold(ABC):
    """Base class for determining how far the floodfill algorithm should go for each plume.
    """
    @abstractmethod
    def get_plume_column_threshold(self, starting_col: float, scene_columns: np.ndarray) -> float:
        """Return the smallest column value that should be considered part of the plume.

        ``starting_col`` must be the column value of the grid cell from which the floodfill
        algorithm begins. ``scene_columns`` is the 2D array of all column amounts in the scene.
        """
        pass

    def description(self) -> str:
        pass

class DecayPlumeThreshold(PlumeThreshold):
    """Thresholder that uses a fraction of the maximum as the threshold.

    This will return a threshold equal to ``fraction_of_max * starting_col``.
    The default of :math:`1/e^2` worked well for a few test NO2 plumes. A factor
    of e was chosen to represent chemical or transport decay of the plume.
    """
    def __init__(self, fraction_of_max: float = np.exp(-2)):
        self.fraction_of_max = fraction_of_max

    def get_plume_column_threshold(self, starting_col: float, scene_columns: np.ndarray) -> float:
        return starting_col * self.fraction_of_max

    def description(self) -> str:
        return f"when column amount decreases to {self.fraction_of_max:.5f} of the initial grid cell's value"


class StddevPlumeThreshold(PlumeThreshold):
    """Thresholder that uses the standard deviation of the scene to determine the plume threshold.

    This will return a threshold equal to the starting column amount minus the standard deviation
    of the column amounts in the scene times ``sigma``. The default sigma of 2.0 was still very
    conservative on the NO2 scenes tested, so this class is generally not used.
    """
    def __init__(self, sigma: float = 2.0):
        self.sigma = sigma

    def get_plume_column_threshold(self, starting_col: float, scene_columns: np.ndarray) -> float:
        if hasattr(scene_columns, 'mask'):
            scene_columns = scene_columns.filled(np.nan)
        scene_sd = np.nanstd(scene_columns, ddof=1)
        t = starting_col - scene_sd * self.sigma
        return t

    def description(self) -> str:
        return f"when column amount decreases to the initial grid cell value minus {self.sigma} times the scene's standard deviation"


class QuantilePlumeThreshold(PlumeThreshold):
    """Thresholder that uses the quantile of the scene columns as the threshold.

    The quantile used is set by ``q``. The default of 0.95 was chosen assuming that plumes should
    be about the top 5% of data, and it worked decently well on a few test NO2 scenes.
    """
    def __init__(self, q: float = 0.95):
        self.q = q

    def get_plume_column_threshold(self, starting_col: float, scene_columns: np.ndarray) -> float:
        if hasattr(scene_columns, 'mask'):
            scene_columns = scene_columns.filled(np.nan)
        return float(np.nanquantile(scene_columns, self.q))

    def description(self) -> str:
        return f"when column amount decreases to the {self.q*100:.0f} percentile of the scene"


def make_plume_mask(columns: np.ndarray, start_selector: PlumeStartSelector, thresholder: PlumeThreshold) -> np.ndarray:
    """Generate a mask that is ``True`` for all grid cells containing a plume.

    Parameters
    ----------
    columns
        The 2D array of column amounts in which we are searching for a plume.

    start_selector
        The instance of a :class:`PlumeStartSelector` subclass that determines which grid cells to begin
        the floodfills from.

    thresholder
        The instance of a :class:`PlumeThreshold` subclass that determines how far the plumes extend.

    Returns
    -------
    mask
        A boolean array that will be ``True`` for grid cells in a plume. If there are multiple plumes,
        there will be more than one ``True`` region in the mask.
    """
    if hasattr(columns, 'mask'):
        columns = columns.filled(np.nan)
    ii_start = start_selector.get_starting_indices(columns)
    overall_mask = np.zeros(columns.shape, dtype=bool)
    for ii in ii_start:
        if overall_mask[ii]:
            # This point was already in a plume, no need to redo the floodfill
            continue
        starting_val = columns[ii]
        thresh = thresholder.get_plume_column_threshold(starting_val, columns)
        this_mask = miscutils.flood_fill_threshold(columns, ii, thresh, allow_diagonals=True)
        overall_mask |= this_mask

    return overall_mask
