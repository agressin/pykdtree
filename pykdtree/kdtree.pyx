#pykdtree, Fast kd-tree implementation with OpenMP-enabled queries
#
#Copyright (C) 2013 - present  Esben S. Nielsen
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
cimport numpy as np
from libc.stdint cimport uint64_t, uint32_t, int32_t, int8_t, uint8_t, UINT32_MAX
cimport cython

np.import_array()


class DESC:
    """Named constants for compute_descriptors() feature indices."""
    EIGENVALUE_1 = 0
    EIGENVALUE_2 = 1
    EIGENVALUE_3 = 2
    NORMAL_X = 3
    NORMAL_Y = 4
    NORMAL_Z = 5
    VERTICALITY = 6
    LINEARITY = 7
    PLANARITY = 8
    SPHERICITY = 9
    OMNIVARIANCE = 10
    ANISOTROPY = 11
    EIGENENTROPY = 12
    SURFACE_VARIATION = 13
    Z_RANGE = 14
    Z_ABOVE = 15
    Z_BELOW = 16
    Z_STD = 17
    DENSITY = 18
    ROUGHNESS = 19
    COUNT = 20
    NAMES = [
        'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',
        'normal_x', 'normal_y', 'normal_z',
        'verticality',
        'linearity', 'planarity', 'sphericity', 'omnivariance',
        'anisotropy', 'eigenentropy', 'surface_variation',
        'z_range', 'z_above', 'z_below', 'z_std',
        'density', 'roughness',
    ]


# Node structure
cdef struct node_float_int32_t:
    float cut_val
    int8_t cut_dim
    uint32_t start_idx
    uint32_t n
    float cut_bounds_lv
    float cut_bounds_hv
    node_float_int32_t *left_child
    node_float_int32_t *right_child

cdef struct tree_float_int32_t:
    float *bbox
    int8_t no_dims
    uint32_t *pidx
    node_float_int32_t *root

cdef struct node_double_int32_t:
    double cut_val
    int8_t cut_dim
    uint32_t start_idx
    uint32_t n
    double cut_bounds_lv
    double cut_bounds_hv
    node_double_int32_t *left_child
    node_double_int32_t *right_child

cdef struct tree_double_int32_t:
    double *bbox
    int8_t no_dims
    uint32_t *pidx
    node_double_int32_t *root

cdef struct node_float_int64_t:
    float cut_val
    int8_t cut_dim
    uint64_t start_idx
    uint64_t n
    float cut_bounds_lv
    float cut_bounds_hv
    node_float_int64_t *left_child
    node_float_int64_t *right_child

cdef struct tree_float_int64_t:
    float *bbox
    int8_t no_dims
    uint64_t *pidx
    node_float_int64_t *root

cdef struct node_double_int64_t:
    double cut_val
    int8_t cut_dim
    uint64_t start_idx
    uint64_t n
    double cut_bounds_lv
    double cut_bounds_hv
    node_double_int64_t *left_child
    node_double_int64_t *right_child

cdef struct tree_double_int64_t:
    double *bbox
    int8_t no_dims
    uint64_t *pidx
    node_double_int64_t *root

cdef extern tree_float_int32_t* construct_tree_float_int32_t(float *pa, int8_t no_dims, uint32_t n, uint32_t bsp) nogil
cdef extern void search_tree_float_int32_t(tree_float_int32_t *kdtree, float *pa, float *point_coords, uint32_t num_points, uint32_t k, float distance_upper_bound, float eps_fac, uint8_t *mask, uint32_t *closest_idxs, float *closest_dists) nogil
cdef extern void delete_tree_float_int32_t(tree_float_int32_t *kdtree)

cdef extern tree_double_int32_t* construct_tree_double_int32_t(double *pa, int8_t no_dims, uint32_t n, uint32_t bsp) nogil
cdef extern void search_tree_double_int32_t(tree_double_int32_t *kdtree, double *pa, double *point_coords, uint32_t num_points, uint32_t k, double distance_upper_bound, double eps_fac, uint8_t *mask, uint32_t *closest_idxs, double *closest_dists) nogil
cdef extern void delete_tree_double_int32_t(tree_double_int32_t *kdtree)

cdef extern tree_float_int64_t* construct_tree_float_int64_t(float *pa, int8_t no_dims, uint64_t n, uint64_t bsp) nogil
cdef extern void search_tree_float_int64_t(tree_float_int64_t *kdtree, float *pa, float *point_coords, uint64_t num_points, uint64_t k, float distance_upper_bound, float eps_fac, uint8_t *mask, uint64_t *closest_idxs, float *closest_dists) nogil
cdef extern void delete_tree_float_int64_t(tree_float_int64_t *kdtree)

cdef extern tree_double_int64_t* construct_tree_double_int64_t(double *pa, int8_t no_dims, uint64_t n, uint64_t bsp) nogil
cdef extern void search_tree_double_int64_t(tree_double_int64_t *kdtree, double *pa, double *point_coords, uint64_t num_points, uint64_t k, double distance_upper_bound, double eps_fac, uint8_t *mask, uint64_t *closest_idxs, double *closest_dists) nogil
cdef extern void delete_tree_double_int64_t(tree_double_int64_t *kdtree)

cdef extern void compute_descriptors_multiscale_float_int32_t(tree_float_int32_t *tree, float *pa, float *point_coords, uint32_t num_points, uint32_t k_max, int32_t *k_scales, int32_t num_scales, float distance_upper_bound, float eps, uint8_t *mask, float *descriptors_out) nogil
cdef extern void compute_descriptors_multiscale_double_int32_t(tree_double_int32_t *tree, double *pa, double *point_coords, uint32_t num_points, uint32_t k_max, int32_t *k_scales, int32_t num_scales, double distance_upper_bound, double eps, uint8_t *mask, double *descriptors_out) nogil
cdef extern void compute_descriptors_multiscale_float_int64_t(tree_float_int64_t *tree, float *pa, float *point_coords, uint64_t num_points, uint64_t k_max, int32_t *k_scales, int32_t num_scales, float distance_upper_bound, float eps, uint8_t *mask, float *descriptors_out) nogil
cdef extern void compute_descriptors_multiscale_double_int64_t(tree_double_int64_t *tree, double *pa, double *point_coords, uint64_t num_points, uint64_t k_max, int32_t *k_scales, int32_t num_scales, double distance_upper_bound, double eps, uint8_t *mask, double *descriptors_out) nogil

cdef extern void sor_mean_dists_float_int32_t(tree_float_int32_t *tree, float *pa, uint32_t num_points, uint32_t k, float *mean_dists_out) nogil
cdef extern void sor_mean_dists_double_int32_t(tree_double_int32_t *tree, double *pa, uint32_t num_points, uint32_t k, double *mean_dists_out) nogil
cdef extern void sor_mean_dists_float_int64_t(tree_float_int64_t *tree, float *pa, uint64_t num_points, uint64_t k, float *mean_dists_out) nogil
cdef extern void sor_mean_dists_double_int64_t(tree_double_int64_t *tree, double *pa, uint64_t num_points, uint64_t k, double *mean_dists_out) nogil

cdef extern void radius_filter_float_int32_t(tree_float_int32_t *tree, float *pa, uint32_t num_points, uint32_t k_min, float radius, uint8_t *inlier_mask_out, uint32_t *count_out) nogil
cdef extern void radius_filter_double_int32_t(tree_double_int32_t *tree, double *pa, uint32_t num_points, uint32_t k_min, double radius, uint8_t *inlier_mask_out, uint32_t *count_out) nogil
cdef extern void radius_filter_float_int64_t(tree_float_int64_t *tree, float *pa, uint64_t num_points, uint64_t k_min, float radius, uint8_t *inlier_mask_out, uint64_t *count_out) nogil
cdef extern void radius_filter_double_int64_t(tree_double_int64_t *tree, double *pa, uint64_t num_points, uint64_t k_min, double radius, uint8_t *inlier_mask_out, uint64_t *count_out) nogil

cdef extern void estimate_normals_float_int32_t(tree_float_int32_t *tree, float *pa, float *point_coords, uint32_t num_points, uint32_t k, float distance_upper_bound, float eps, uint8_t *mask, float *normals_out, float *curvatures_out) nogil
cdef extern void estimate_normals_double_int32_t(tree_double_int32_t *tree, double *pa, double *point_coords, uint32_t num_points, uint32_t k, double distance_upper_bound, double eps, uint8_t *mask, double *normals_out, double *curvatures_out) nogil
cdef extern void estimate_normals_float_int64_t(tree_float_int64_t *tree, float *pa, float *point_coords, uint64_t num_points, uint64_t k, float distance_upper_bound, float eps, uint8_t *mask, float *normals_out, float *curvatures_out) nogil
cdef extern void estimate_normals_double_int64_t(tree_double_int64_t *tree, double *pa, double *point_coords, uint64_t num_points, uint64_t k, double distance_upper_bound, double eps, uint8_t *mask, double *normals_out, double *curvatures_out) nogil

cdef class KDTree:
    """kd-tree for fast nearest-neighbour lookup.
    The interface is made to resemble the scipy.spatial kd-tree except
    only Euclidean distance measure is supported.

    :Parameters:
    data_pts : numpy array
        Data points with shape (n , dims)
    leafsize : int, optional
        Maximum number of data points in tree leaf
    """

    cdef tree_float_int32_t *_kdtree_float_int32_t
    cdef tree_double_int32_t *_kdtree_double_int32_t
    cdef tree_float_int64_t *_kdtree_float_int64_t
    cdef tree_double_int64_t *_kdtree_double_int64_t
    cdef readonly bint _use_int32_t
    cdef readonly np.ndarray data_pts
    cdef readonly np.ndarray data
    cdef float *_data_pts_data_float
    cdef double *_data_pts_data_double
    cdef readonly uint64_t n
    cdef readonly int8_t ndim
    cdef readonly uint32_t leafsize

    def __cinit__(KDTree self):
        self._kdtree_float_int32_t = NULL
        self._kdtree_double_int32_t = NULL
        self._kdtree_float_int64_t = NULL
        self._kdtree_double_int64_t = NULL

    def __init__(KDTree self, np.ndarray data_pts not None, int leafsize=16):

        # Check arguments
        if leafsize < 1:
            raise ValueError('leafsize must be greater than zero')
        if data_pts.ndim != 2:
            raise ValueError('data_pts array should have exactly 2 dimensions')
        if data_pts.size == 0:
            raise ValueError('data_pts should be non-empty')

        # Get data content
        cdef np.ndarray[float, ndim=1] data_array_float
        cdef np.ndarray[double, ndim=1] data_array_double

        if data_pts.dtype == np.float32:
            data_array_float = np.ascontiguousarray(data_pts.ravel(), dtype=np.float32)
            self._data_pts_data_float = <float *>data_array_float.data
            self.data_pts = data_array_float
        else:
            data_array_double = np.ascontiguousarray(data_pts.ravel(), dtype=np.float64)
            self._data_pts_data_double = <double *>data_array_double.data
            self.data_pts = data_array_double

        # scipy interface compatibility
        self.data = self.data_pts

        # Get tree info
        self.n = <uint64_t>data_pts.shape[0]
        self._use_int32_t = self.n * data_pts.shape[1] < UINT32_MAX
        self.leafsize = <uint32_t>leafsize
        if data_pts.ndim == 1:
            self.ndim = 1
        elif data_pts.shape[1] > 127:
            raise ValueError('Max 127 dimensions allowed')
        else:
            self.ndim = <int8_t>data_pts.shape[1]

        # Release GIL and construct tree
        if data_pts.dtype == np.float32:
            if self._use_int32_t:
                with nogil:
                    self._kdtree_float_int32_t = construct_tree_float_int32_t(self._data_pts_data_float, self.ndim,
                                                              <uint32_t>self.n, self.leafsize)
            else:
                with nogil:
                    self._kdtree_float_int64_t = construct_tree_float_int64_t(self._data_pts_data_float, self.ndim,
                                                              self.n, self.leafsize)
        else:
            if self._use_int32_t:
                with nogil:
                    self._kdtree_double_int32_t = construct_tree_double_int32_t(self._data_pts_data_double, self.ndim,
                                                                <uint32_t>self.n, self.leafsize)
            else:
                with nogil:
                    self._kdtree_double_int64_t = construct_tree_double_int64_t(self._data_pts_data_double, self.ndim,
                                                                self.n, self.leafsize)


    def query(KDTree self, np.ndarray query_pts not None, k=1, eps=0,
              distance_upper_bound=None, sqr_dists=False, mask=None):
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

        # Check arguments
        if k < 1:
            raise ValueError('Number of neighbours must be greater than zero')
        elif eps < 0:
            raise ValueError('eps must be non-negative')
        elif distance_upper_bound is not None:
            if distance_upper_bound < 0:
                raise ValueError('distance_upper_bound must be non negative')

        # Check dimensions
        if query_pts.ndim == 1:
            q_ndim = 1
        else:
            q_ndim = query_pts.shape[1]

        if self.ndim != q_ndim:
            raise ValueError('Data and query points must have same dimensions')

        if self.data_pts.dtype == np.float32 and query_pts.dtype != np.float32:
            raise TypeError('Type mismatch. query points must be of type float32 when data points are of type float32')

        # Get query info
        cdef uint64_t num_qpoints = query_pts.shape[0]
        cdef uint64_t num_n = k
        cdef np.ndarray[uint32_t, ndim=1] closest_idxs_int32_t
        cdef np.ndarray[uint64_t, ndim=1] closest_idxs_int64_t
        cdef np.ndarray[float, ndim=1] closest_dists_float
        cdef np.ndarray[double, ndim=1] closest_dists_double

        # Set up return arrays
        cdef uint32_t *closest_idxs_data_int32_t
        cdef uint64_t *closest_idxs_data_int64_t
        cdef float *closest_dists_data_float
        cdef double *closest_dists_data_double
        if self._use_int32_t:
            closest_idxs_int32_t = np.empty(num_qpoints * k, dtype=np.uint32)
            closest_idxs = closest_idxs_int32_t
            closest_idxs_data_int32_t = <uint32_t *>closest_idxs_int32_t.data
        else:
            closest_idxs_int64_t = np.empty(num_qpoints * k, dtype=np.uint64)
            closest_idxs = closest_idxs_int64_t
            closest_idxs_data_int64_t = <uint64_t *>closest_idxs_int64_t.data

        # Get query points data      
        cdef np.ndarray[float, ndim=1] query_array_float 
        cdef np.ndarray[double, ndim=1] query_array_double 
        cdef float *query_array_data_float 
        cdef double *query_array_data_double
        cdef np.ndarray[np.uint8_t, ndim=1] query_mask
        cdef np.uint8_t *query_mask_data

        if mask is not None and mask.size != self.n:
            raise ValueError('Mask must have the same size as data points')
        elif mask is not None:
            query_mask = np.ascontiguousarray(mask.ravel(), dtype=np.uint8)
            query_mask_data = <uint8_t *>query_mask.data
        else:
            query_mask_data = NULL


        if query_pts.dtype == np.float32 and self.data_pts.dtype == np.float32:
            closest_dists_float = np.empty(num_qpoints * k, dtype=np.float32)
            closest_dists = closest_dists_float
            closest_dists_data_float = <float *>closest_dists_float.data
            query_array_float = np.ascontiguousarray(query_pts.ravel(), dtype=np.float32)
            query_array_data_float = <float *>query_array_float.data
        else:
            closest_dists_double = np.empty(num_qpoints * k, dtype=np.float64)
            closest_dists = closest_dists_double
            closest_dists_data_double = <double *>closest_dists_double.data
            query_array_double = np.ascontiguousarray(query_pts.ravel(), dtype=np.float64)
            query_array_data_double = <double *>query_array_double.data

        # Setup distance_upper_bound
        cdef float dub_float
        cdef double dub_double
        if distance_upper_bound is None:
            if self.data_pts.dtype == np.float32:
                dub_float = <float>np.finfo(np.float32).max
            else:
                dub_double = <double>np.finfo(np.float64).max
        else:
            if self.data_pts.dtype == np.float32:
                dub_float = <float>(distance_upper_bound * distance_upper_bound)
            else:
                dub_double = <double>(distance_upper_bound * distance_upper_bound)

        # Set epsilon
        cdef double epsilon_float = <float>eps
        cdef double epsilon_double = <double>eps

        # Release GIL and query tree
        if self.data_pts.dtype == np.float32:
            if self._use_int32_t:
                with nogil:
                    search_tree_float_int32_t(self._kdtree_float_int32_t, self._data_pts_data_float,
                                      query_array_data_float, <uint32_t>num_qpoints, <uint32_t>num_n, dub_float, epsilon_float,
                                      query_mask_data, closest_idxs_data_int32_t, closest_dists_data_float)
            else:
                with nogil:
                    search_tree_float_int64_t(self._kdtree_float_int64_t, self._data_pts_data_float,
                                      query_array_data_float, num_qpoints, num_n, dub_float, epsilon_float,
                                      query_mask_data, closest_idxs_data_int64_t, closest_dists_data_float)
        else:
            if self._use_int32_t:
                with nogil:
                    search_tree_double_int32_t(self._kdtree_double_int32_t, self._data_pts_data_double,
                                      query_array_data_double, <uint32_t>num_qpoints, <uint32_t>num_n, dub_double, epsilon_double,
                                       query_mask_data, closest_idxs_data_int32_t, closest_dists_data_double)
            else:
                with nogil:
                    search_tree_double_int64_t(self._kdtree_double_int64_t, self._data_pts_data_double,
                                      query_array_data_double, num_qpoints, num_n, dub_double, epsilon_double,
                                       query_mask_data, closest_idxs_data_int64_t, closest_dists_data_double)

        # Shape result
        if k > 1:
            closest_dists_res = closest_dists.reshape(num_qpoints, k)
            closest_idxs_res = closest_idxs.reshape(num_qpoints, k)
        else:
            closest_dists_res = closest_dists
            closest_idxs_res = closest_idxs

        if distance_upper_bound is not None: # Mark out of bounds results
            if self.data_pts.dtype == np.float32:
                idx_out = (closest_dists_res >= dub_float)
            else:
                idx_out = (closest_dists_res >= dub_double)

            closest_dists_res[idx_out] = np.inf
            closest_idxs_res[idx_out] = self.n

        if not sqr_dists: # Return actual cartesian distances
            closest_dists_res = np.sqrt(closest_dists_res)

        return closest_dists_res, closest_idxs_res

    def compute_descriptors(KDTree self, np.ndarray query_pts not None, k=16, eps=0,
                            distance_upper_bound=None, mask=None):
        """Compute comprehensive point descriptors in a single k-NN pass.

        Outputs 20 features per point per scale, suitable for deep learning.
        Supports multi-scale: pass k as a list (e.g. [5, 10, 20]).
        Only works for 3D data (ndim=3).

        Features (20 per point per scale) - use DESC.* constants for indexing:
          0-2: eigenvalues (lambda1 >= lambda2 >= lambda3)
          3-5: normal vector (nx, ny, nz)
          6:   verticality (1 - |nz|)
          7:   linearity = (l1 - l2) / l1
          8:   planarity = (l2 - l3) / l1
          9:   sphericity = l3 / l1
          10:  omnivariance = (l1*l2*l3)^(1/3)
          11:  anisotropy = (l1 - l3) / l1
          12:  eigenentropy = -sum(li/S * ln(li/S))
          13:  surface_variation = l3 / (l1+l2+l3)
          14:  z_range (height range of neighbors)
          15:  z_above (max neighbor z - query z)
          16:  z_below (query z - min neighbor z)
          17:  z_std (height std of neighbors)
          18:  density (k / bounding box volume)
          19:  roughness (point-to-plane distance)

        :Parameters:
        query_pts : numpy array, shape (m, 3)
        k : int or list of ints
        eps : non-negative float
        distance_upper_bound : non-negative float, optional
        mask : numpy array, optional

        :Returns:
        descriptors : numpy array
            Shape (m, 20) if k is int, (m, num_scales, 20) if k is list.
        """

        if self.ndim != 3:
            raise ValueError('compute_descriptors only supports 3D data (ndim=3)')
        if eps < 0:
            raise ValueError('eps must be non-negative')
        if distance_upper_bound is not None and distance_upper_bound < 0:
            raise ValueError('distance_upper_bound must be non-negative')

        if query_pts.ndim == 1:
            q_ndim = 1
        else:
            q_ndim = query_pts.shape[1]
        if self.ndim != q_ndim:
            raise ValueError('Data and query points must have same dimensions')
        if self.data_pts.dtype == np.float32 and query_pts.dtype != np.float32:
            raise TypeError('Type mismatch. query points must be of type float32 when data points are of type float32')

        cdef bint multiscale = isinstance(k, (list, tuple))
        cdef np.ndarray[int32_t, ndim=1] k_scales_arr
        cdef int32_t *k_scales_data
        cdef int32_t c_num_scales
        cdef int32_t NUM_DESC = 20

        if multiscale:
            k_scales_arr = np.array(sorted(k), dtype=np.int32)
            k_scales_data = <int32_t *>k_scales_arr.data
            c_num_scales = <int32_t>len(k_scales_arr)
            if c_num_scales == 0:
                raise ValueError('k must be a non-empty list')
            if k_scales_arr[0] < 2:
                raise ValueError('All k values must be >= 2')
            k_max_val = int(k_scales_arr[c_num_scales - 1])
        else:
            if k < 2:
                raise ValueError('k must be >= 2')
            k_max_val = int(k)
            k_scales_arr = np.array([k], dtype=np.int32)
            k_scales_data = <int32_t *>k_scales_arr.data
            c_num_scales = 1

        cdef uint64_t num_qpoints = query_pts.shape[0]
        cdef uint64_t num_k_max = k_max_val
        cdef uint64_t total_out = num_qpoints * c_num_scales * NUM_DESC

        cdef np.ndarray[float, ndim=1] desc_float
        cdef np.ndarray[double, ndim=1] desc_double
        cdef float *desc_data_float
        cdef double *desc_data_double

        cdef np.ndarray[float, ndim=1] query_array_float
        cdef np.ndarray[double, ndim=1] query_array_double
        cdef float *query_array_data_float
        cdef double *query_array_data_double

        cdef np.ndarray[np.uint8_t, ndim=1] query_mask
        cdef np.uint8_t *query_mask_data

        if mask is not None and mask.size != self.n:
            raise ValueError('Mask must have the same size as data points')
        elif mask is not None:
            query_mask = np.ascontiguousarray(mask.ravel(), dtype=np.uint8)
            query_mask_data = <uint8_t *>query_mask.data
        else:
            query_mask_data = NULL

        cdef float dub_float
        cdef double dub_double
        if distance_upper_bound is None:
            if self.data_pts.dtype == np.float32:
                dub_float = <float>np.finfo(np.float32).max
            else:
                dub_double = <double>np.finfo(np.float64).max
        else:
            if self.data_pts.dtype == np.float32:
                dub_float = <float>(distance_upper_bound * distance_upper_bound)
            else:
                dub_double = <double>(distance_upper_bound * distance_upper_bound)

        cdef float epsilon_float = <float>eps
        cdef double epsilon_double = <double>eps

        if query_pts.dtype == np.float32 and self.data_pts.dtype == np.float32:
            desc_float = np.empty(total_out, dtype=np.float32)
            desc_data_float = <float *>desc_float.data
            query_array_float = np.ascontiguousarray(query_pts.ravel(), dtype=np.float32)
            query_array_data_float = <float *>query_array_float.data

            if self._use_int32_t:
                with nogil:
                    compute_descriptors_multiscale_float_int32_t(self._kdtree_float_int32_t, self._data_pts_data_float,
                        query_array_data_float, <uint32_t>num_qpoints, <uint32_t>num_k_max,
                        k_scales_data, c_num_scales, dub_float, epsilon_float,
                        query_mask_data, desc_data_float)
            else:
                with nogil:
                    compute_descriptors_multiscale_float_int64_t(self._kdtree_float_int64_t, self._data_pts_data_float,
                        query_array_data_float, num_qpoints, num_k_max,
                        k_scales_data, c_num_scales, dub_float, epsilon_float,
                        query_mask_data, desc_data_float)

            if multiscale:
                return desc_float.reshape(num_qpoints, c_num_scales, NUM_DESC)
            else:
                return desc_float.reshape(num_qpoints, NUM_DESC)
        else:
            desc_double = np.empty(total_out, dtype=np.float64)
            desc_data_double = <double *>desc_double.data
            query_array_double = np.ascontiguousarray(query_pts.ravel(), dtype=np.float64)
            query_array_data_double = <double *>query_array_double.data

            if self._use_int32_t:
                with nogil:
                    compute_descriptors_multiscale_double_int32_t(self._kdtree_double_int32_t, self._data_pts_data_double,
                        query_array_data_double, <uint32_t>num_qpoints, <uint32_t>num_k_max,
                        k_scales_data, c_num_scales, dub_double, epsilon_double,
                        query_mask_data, desc_data_double)
            else:
                with nogil:
                    compute_descriptors_multiscale_double_int64_t(self._kdtree_double_int64_t, self._data_pts_data_double,
                        query_array_data_double, num_qpoints, num_k_max,
                        k_scales_data, c_num_scales, dub_double, epsilon_double,
                        query_mask_data, desc_data_double)

            if multiscale:
                return desc_double.reshape(num_qpoints, c_num_scales, NUM_DESC)
            else:
                return desc_double.reshape(num_qpoints, NUM_DESC)

    def statistical_outlier_removal(KDTree self, k=20, std_ratio=2.0):
        """Statistical Outlier Removal (SOR).

        Removes points whose mean distance to k neighbors exceeds
        ``global_mean + std_ratio * global_std``.
        Same algorithm as CloudCompare/Open3D SOR.

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

        if k < 1:
            raise ValueError('k must be >= 1')
        if std_ratio < 0:
            raise ValueError('std_ratio must be non-negative')

        cdef uint64_t num_points = self.n
        cdef uint64_t num_k = k

        cdef np.ndarray[float, ndim=1] mean_dists_float
        cdef np.ndarray[double, ndim=1] mean_dists_double
        cdef float *mean_dists_data_float
        cdef double *mean_dists_data_double

        if self.data_pts.dtype == np.float32:
            mean_dists_float = np.empty(num_points, dtype=np.float32)
            mean_dists_data_float = <float *>mean_dists_float.data

            if self._use_int32_t:
                with nogil:
                    sor_mean_dists_float_int32_t(self._kdtree_float_int32_t, self._data_pts_data_float,
                        <uint32_t>num_points, <uint32_t>num_k, mean_dists_data_float)
            else:
                with nogil:
                    sor_mean_dists_float_int64_t(self._kdtree_float_int64_t, self._data_pts_data_float,
                        num_points, num_k, mean_dists_data_float)

            mean_dists = mean_dists_float
        else:
            mean_dists_double = np.empty(num_points, dtype=np.float64)
            mean_dists_data_double = <double *>mean_dists_double.data

            if self._use_int32_t:
                with nogil:
                    sor_mean_dists_double_int32_t(self._kdtree_double_int32_t, self._data_pts_data_double,
                        <uint32_t>num_points, <uint32_t>num_k, mean_dists_data_double)
            else:
                with nogil:
                    sor_mean_dists_double_int64_t(self._kdtree_double_int64_t, self._data_pts_data_double,
                        num_points, num_k, mean_dists_data_double)

            mean_dists = mean_dists_double

        global_mean = mean_dists.mean()
        global_std = mean_dists.std()
        threshold = float(global_mean + std_ratio * global_std)
        inlier_mask = mean_dists < threshold

        return inlier_mask, mean_dists, threshold

    def radius_filter(KDTree self, k_min=10, radius=1.0):
        """Radius-based outlier filter.

        Marks points as inliers if they have at least ``k_min`` neighbors
        within ``radius``. Useful for removing isolated noise points.

        :Parameters:
        k_min : int
            Minimum number of neighbors required within radius
        radius : float
            Search radius (Euclidean distance)

        :Returns:
        inlier_mask : numpy boolean array, shape (n,)
            True for inlier points
        n_inliers : int
            Number of inlier points
        """
        if k_min < 1:
            raise ValueError('k_min must be >= 1')
        if radius <= 0:
            raise ValueError('radius must be positive')

        cdef uint64_t num_points = self.n
        cdef np.ndarray[np.uint8_t, ndim=1] mask = np.empty(num_points, dtype=np.uint8)
        cdef uint8_t *mask_data = <uint8_t *>mask.data
        cdef uint32_t count32 = 0
        cdef uint64_t count64 = 0
        cdef uint64_t c_k_min = <uint64_t>k_min
        cdef float c_radius_float = <float>radius
        cdef double c_radius_double = <double>radius

        if self.data_pts.dtype == np.float32:
            if self._use_int32_t:
                with nogil:
                    radius_filter_float_int32_t(self._kdtree_float_int32_t, self._data_pts_data_float,
                        <uint32_t>num_points, <uint32_t>c_k_min, c_radius_float, mask_data, &count32)
                n_inliers = int(count32)
            else:
                with nogil:
                    radius_filter_float_int64_t(self._kdtree_float_int64_t, self._data_pts_data_float,
                        num_points, c_k_min, c_radius_float, mask_data, &count64)
                n_inliers = int(count64)
        else:
            if self._use_int32_t:
                with nogil:
                    radius_filter_double_int32_t(self._kdtree_double_int32_t, self._data_pts_data_double,
                        <uint32_t>num_points, <uint32_t>c_k_min, c_radius_double, mask_data, &count32)
                n_inliers = int(count32)
            else:
                with nogil:
                    radius_filter_double_int64_t(self._kdtree_double_int64_t, self._data_pts_data_double,
                        num_points, c_k_min, c_radius_double, mask_data, &count64)
                n_inliers = int(count64)

        return mask.view(np.bool_), n_inliers

    def estimate_normals(KDTree self, np.ndarray query_pts not None, k=20, eps=0,
                         distance_upper_bound=None, mask=None):
        """Estimate surface normals and curvatures from k-NN neighborhoods.

        Lightweight alternative to compute_descriptors when only normals
        and curvature are needed. Uses eigendecomposition of the local
        covariance matrix (Cardano's formula for 3x3).

        Only works for 3D data (ndim=3).

        :Parameters:
        query_pts : numpy array, shape (m, 3)
        k : int
            Number of neighbors
        eps : non-negative float
        distance_upper_bound : float, optional
        mask : numpy array, optional

        :Returns:
        normals : numpy array, shape (m, 3)
            Unit normal vectors (oriented upward)
        curvatures : numpy array, shape (m,)
            Surface variation: lambda3 / (lambda1 + lambda2 + lambda3)
        """
        if self.ndim != 3:
            raise ValueError('estimate_normals only supports 3D data (ndim=3)')
        if k < 2:
            raise ValueError('k must be >= 2')
        if eps < 0:
            raise ValueError('eps must be non-negative')
        if distance_upper_bound is not None and distance_upper_bound < 0:
            raise ValueError('distance_upper_bound must be non-negative')

        if query_pts.ndim == 1:
            q_ndim = 1
        else:
            q_ndim = query_pts.shape[1]
        if self.ndim != q_ndim:
            raise ValueError('Data and query points must have same dimensions')
        if self.data_pts.dtype == np.float32 and query_pts.dtype != np.float32:
            raise TypeError('Type mismatch. query points must be of type float32 when data points are of type float32')

        cdef uint64_t num_qpoints = query_pts.shape[0]

        cdef np.ndarray[float, ndim=1] query_array_float
        cdef np.ndarray[double, ndim=1] query_array_double
        cdef float *query_data_float
        cdef double *query_data_double

        cdef np.ndarray[np.uint8_t, ndim=1] query_mask
        cdef np.uint8_t *query_mask_data

        if mask is not None and mask.size != self.n:
            raise ValueError('Mask must have the same size as data points')
        elif mask is not None:
            query_mask = np.ascontiguousarray(mask.ravel(), dtype=np.uint8)
            query_mask_data = <uint8_t *>query_mask.data
        else:
            query_mask_data = NULL

        cdef float dub_float
        cdef double dub_double
        if distance_upper_bound is None:
            if self.data_pts.dtype == np.float32:
                dub_float = <float>np.finfo(np.float32).max
            else:
                dub_double = <double>np.finfo(np.float64).max
        else:
            if self.data_pts.dtype == np.float32:
                dub_float = <float>(distance_upper_bound * distance_upper_bound)
            else:
                dub_double = <double>(distance_upper_bound * distance_upper_bound)

        cdef float epsilon_float = <float>eps
        cdef double epsilon_double = <double>eps
        cdef uint64_t c_k = <uint64_t>k

        cdef np.ndarray[float, ndim=1] normals_float
        cdef np.ndarray[double, ndim=1] normals_double
        cdef np.ndarray[float, ndim=1] curvatures_float
        cdef np.ndarray[double, ndim=1] curvatures_double

        if query_pts.dtype == np.float32 and self.data_pts.dtype == np.float32:
            query_array_float = np.ascontiguousarray(query_pts.ravel(), dtype=np.float32)
            query_data_float = <float *>query_array_float.data
            normals_float = np.empty(num_qpoints * 3, dtype=np.float32)
            curvatures_float = np.empty(num_qpoints, dtype=np.float32)

            if self._use_int32_t:
                with nogil:
                    estimate_normals_float_int32_t(self._kdtree_float_int32_t, self._data_pts_data_float,
                        query_data_float, <uint32_t>num_qpoints, <uint32_t>c_k,
                        dub_float, epsilon_float, query_mask_data,
                        <float *>normals_float.data, <float *>curvatures_float.data)
            else:
                with nogil:
                    estimate_normals_float_int64_t(self._kdtree_float_int64_t, self._data_pts_data_float,
                        query_data_float, num_qpoints, c_k,
                        dub_float, epsilon_float, query_mask_data,
                        <float *>normals_float.data, <float *>curvatures_float.data)

            return normals_float.reshape(num_qpoints, 3), curvatures_float
        else:
            query_array_double = np.ascontiguousarray(query_pts.ravel(), dtype=np.float64)
            query_data_double = <double *>query_array_double.data
            normals_double = np.empty(num_qpoints * 3, dtype=np.float64)
            curvatures_double = np.empty(num_qpoints, dtype=np.float64)

            if self._use_int32_t:
                with nogil:
                    estimate_normals_double_int32_t(self._kdtree_double_int32_t, self._data_pts_data_double,
                        query_data_double, <uint32_t>num_qpoints, <uint32_t>c_k,
                        dub_double, epsilon_double, query_mask_data,
                        <double *>normals_double.data, <double *>curvatures_double.data)
            else:
                with nogil:
                    estimate_normals_double_int64_t(self._kdtree_double_int64_t, self._data_pts_data_double,
                        query_data_double, num_qpoints, c_k,
                        dub_double, epsilon_double, query_mask_data,
                        <double *>normals_double.data, <double *>curvatures_double.data)

            return normals_double.reshape(num_qpoints, 3), curvatures_double

    def __dealloc__(KDTree self):
        if self._kdtree_float_int32_t != NULL:
            delete_tree_float_int32_t(self._kdtree_float_int32_t)
        elif self._kdtree_double_int32_t != NULL:
            delete_tree_double_int32_t(self._kdtree_double_int32_t)
        if self._kdtree_float_int64_t != NULL:
            delete_tree_float_int64_t(self._kdtree_float_int64_t)
        elif self._kdtree_double_int64_t != NULL:
            delete_tree_double_int64_t(self._kdtree_double_int64_t)
