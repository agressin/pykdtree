# pykdtree, Fast kd-tree implementation with OpenMP-enabled queries
#
# Copyright (C) 2013 - present  Esben S. Nielsen
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

from typing_extensions import disjoint_base
from typing import Any
import numpy as np


class DESC:
    """Named constants for compute_descriptors() feature indices."""
    EIGENVALUE_1: int
    EIGENVALUE_2: int
    EIGENVALUE_3: int
    NORMAL_X: int
    NORMAL_Y: int
    NORMAL_Z: int
    VERTICALITY: int
    LINEARITY: int
    PLANARITY: int
    SPHERICITY: int
    OMNIVARIANCE: int
    ANISOTROPY: int
    EIGENENTROPY: int
    SURFACE_VARIATION: int
    Z_RANGE: int
    Z_ABOVE: int
    Z_BELOW: int
    Z_STD: int
    DENSITY: int
    ROUGHNESS: int
    COUNT: int
    NAMES: list[str]


@disjoint_base
class KDTree:
    """kd-tree for fast nearest-neighbour lookup.
    The interface is made to resemble the scipy.spatial kd-tree except
    only Euclidean distance measure is supported.

    :Parameters:
    data_pts : numpy array
        Data points with shape (n , dims)
    leafsize : int, optional
        Maximum number of data points in tree leaf
    """

    @property
    def data_pts(self) -> np.ndarray:
        """Data points used to construct the kd-tree."""

    @property
    def data(self) -> np.ndarray:
        """Data points used to construct the kd-tree."""

    @property
    def n(self) -> int:
        """Number of data points."""

    @property
    def ndim(self) -> int:
        """Number of dimensions."""

    @property
    def leafsize(self) -> int:
        """Maximum number of data points in tree leaf."""

    def __init__(self, data_pts: np.ndarray, leafsize: int = 16): ...
    def query(
        self,
        query_pts: np.ndarray,
        k: int = 1,
        eps: float = 0,
        distance_upper_bound: float | None = None,
        sqr_dists: bool = False,
        mask: np.ndarray | None = None,
    ):
        """Query the kd-tree for nearest neighbors

        :Parameters:
        query_pts : numpy array
            Query points with shape (m, dims)
        k : int
            The number of nearest neighbours to return
        eps : non-negative float
            Return approximate nearest neighbours; the k-th returned value
            is guaranteed to be no further than (1 + eps) times the distance
            to the real k-th nearest neighbour
        distance_upper_bound : non-negative float
            Return only neighbors within this distance.
            This is used to prune tree searches.
        sqr_dists : bool, optional
            Internally pykdtree works with squared distances.
            Determines if the squared or Euclidean distances are returned.
        mask : numpy array, optional
            Array of booleans where neighbors are considered invalid and
            should not be returned. A mask value of True represents an
            invalid pixel. Mask should have shape (n,) to match data points.
            By default all points are considered valid.

        """
        ...

    def compute_descriptors(
        self,
        query_pts: np.ndarray,
        k: int | list[int] = 20,
        eps: float = 0,
        distance_upper_bound: float | None = None,
        mask: np.ndarray | None = None,
    ):
        """Compute comprehensive point descriptors for 3D query points.

        Computes 20 features per point per scale in a single k-NN pass.
        Use DESC.* constants for indexing (e.g. desc[:, DESC.LINEARITY]).

        Features:
          0-2: eigenvalues, 3-5: normal, 6: verticality,
          7: linearity, 8: planarity, 9: sphericity,
          10: omnivariance, 11: anisotropy, 12: eigenentropy,
          13: surface_variation, 14: z_range, 15: z_above,
          16: z_below, 17: z_std, 18: density, 19: roughness

        Supports multi-scale: pass k as a list (e.g. [5, 10, 20]).

        :Parameters:
        query_pts : numpy array
            Query points with shape (m, 3)
        k : int or list of ints
            Number of nearest neighbours. If a list, multi-scale.
        eps : non-negative float
            Return approximate nearest neighbours
        distance_upper_bound : non-negative float, optional
            Return only neighbors within this distance
        mask : numpy array, optional
            Boolean mask for invalid data points, shape (n,)

        :Returns:
        descriptors : numpy array
            Shape (m, 20) if k is int, (m, num_scales, 20) if k is list.
        """
        ...

    def statistical_outlier_removal(
        self,
        k: int = 20,
        std_ratio: float = 2.0,
    ):
        """Statistical Outlier Removal (SOR).

        Removes points whose mean distance to k neighbors exceeds
        ``global_mean + std_ratio * global_std``.

        :Parameters:
        k : int
            Number of neighbors to consider
        std_ratio : float
            Number of standard deviations for threshold

        :Returns:
        inlier_mask : numpy boolean array
            True for inlier points, shape (n,)
        mean_distances : numpy array
            Mean Euclidean distance to k neighbors, shape (n,)
        threshold : float
            Distance threshold used
        """
        ...

# These are generated by Cython.
# Just in here to avoid errors in mypy tests.
__reduce_cython__: Any
__setstate_cython__: Any
__test__: Any
