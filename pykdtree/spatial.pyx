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

"""Spatial operations for point cloud processing.

OpenMP-accelerated functions for voxelization, space-filling curves,
tiling, scatter/gather, and spatial filtering.
"""

import numpy as np
cimport numpy as np
from libc.stdint cimport uint64_t, uint32_t, int32_t, uint8_t
cimport cython

np.import_array()


# ---- C extern declarations ----

cdef extern void morton_encode_3d(uint64_t *x, uint64_t *y, uint64_t *z,
                                  uint64_t n, uint64_t *codes_out) nogil
cdef extern void morton_decode_3d(uint64_t *codes, uint64_t n,
                                  uint64_t *x_out, uint64_t *y_out, uint64_t *z_out) nogil
cdef extern void hilbert_encode_3d(uint64_t *x, uint64_t *y, uint64_t *z,
                                   uint64_t n, int32_t order, uint64_t *codes_out) nogil
cdef extern void voxel_downsample_float(float *points, uint64_t n, float voxel_size,
                                        uint64_t *selected_out, uint64_t *inverse_out,
                                        uint64_t *n_unique_out) nogil
cdef extern void voxel_downsample_double(double *points, uint64_t n, double voxel_size,
                                         uint64_t *selected_out, uint64_t *inverse_out,
                                         uint64_t *n_unique_out) nogil

cdef extern void assign_tiles_float(float *points, uint64_t n, float tile_size,
                                    int32_t *tile_x_out, int32_t *tile_y_out) nogil
cdef extern void assign_tiles_double(double *points, uint64_t n, double tile_size,
                                     int32_t *tile_x_out, int32_t *tile_y_out) nogil

cdef extern void scatter_minmax_float(float *points_xy, float *values, uint64_t n,
                                      uint32_t grid_h, uint32_t grid_w, float resolution,
                                      float origin_x, float origin_y,
                                      float *min_grid, float *max_grid, uint32_t *count_grid) nogil
cdef extern void scatter_minmax_double(double *points_xy, double *values, uint64_t n,
                                       uint32_t grid_h, uint32_t grid_w, double resolution,
                                       double origin_x, double origin_y,
                                       double *min_grid, double *max_grid, uint32_t *count_grid) nogil

cdef extern void grid_sample_nearest_float(float *grid, uint32_t grid_h, uint32_t grid_w,
                                           float *points_xy, uint64_t n, float resolution,
                                           float origin_x, float origin_y,
                                           float fill_value, float *values_out) nogil
cdef extern void grid_sample_nearest_double(double *grid, uint32_t grid_h, uint32_t grid_w,
                                            double *points_xy, uint64_t n, double resolution,
                                            double origin_x, double origin_y,
                                            double fill_value, double *values_out) nogil

cdef extern void filter_points_in_bbox_float(float *points, uint64_t n,
                                             float min_x, float min_y, float min_z,
                                             float max_x, float max_y, float max_z,
                                             uint8_t *mask_out, uint64_t *count_out) nogil
cdef extern void filter_points_in_bbox_double(double *points, uint64_t n,
                                              double min_x, double min_y, double min_z,
                                              double max_x, double max_y, double max_z,
                                              uint8_t *mask_out, uint64_t *count_out) nogil


# ---- Python API ----


def morton_encode(np.ndarray x not None, np.ndarray y not None, np.ndarray z not None):
    """Encode 3D integer coordinates to Morton (Z-order) codes.

    Each coordinate must fit in 21 bits (0 to 2097151).

    :Parameters:
    x, y, z : numpy uint64 arrays, shape (n,)

    :Returns:
    codes : numpy uint64 array, shape (n,)
    """
    cdef np.ndarray[uint64_t, ndim=1] xa = np.ascontiguousarray(x.ravel(), dtype=np.uint64)
    cdef np.ndarray[uint64_t, ndim=1] ya = np.ascontiguousarray(y.ravel(), dtype=np.uint64)
    cdef np.ndarray[uint64_t, ndim=1] za = np.ascontiguousarray(z.ravel(), dtype=np.uint64)
    cdef uint64_t n = <uint64_t>xa.shape[0]
    cdef np.ndarray[uint64_t, ndim=1] codes = np.empty(n, dtype=np.uint64)

    with nogil:
        morton_encode_3d(<uint64_t *>xa.data, <uint64_t *>ya.data, <uint64_t *>za.data,
                         n, <uint64_t *>codes.data)
    return codes


def morton_decode(np.ndarray codes not None):
    """Decode Morton codes back to 3D integer coordinates.

    :Parameters:
    codes : numpy uint64 array, shape (n,)

    :Returns:
    x, y, z : numpy uint64 arrays, shape (n,)
    """
    cdef np.ndarray[uint64_t, ndim=1] ca = np.ascontiguousarray(codes.ravel(), dtype=np.uint64)
    cdef uint64_t n = <uint64_t>ca.shape[0]
    cdef np.ndarray[uint64_t, ndim=1] xo = np.empty(n, dtype=np.uint64)
    cdef np.ndarray[uint64_t, ndim=1] yo = np.empty(n, dtype=np.uint64)
    cdef np.ndarray[uint64_t, ndim=1] zo = np.empty(n, dtype=np.uint64)

    with nogil:
        morton_decode_3d(<uint64_t *>ca.data, n,
                         <uint64_t *>xo.data, <uint64_t *>yo.data, <uint64_t *>zo.data)
    return xo, yo, zo


def hilbert_encode(np.ndarray x not None, np.ndarray y not None, np.ndarray z not None,
                   int order=21):
    """Encode 3D integer coordinates to Hilbert curve codes.

    Uses the Hamilton & Rau-Chaplin algorithm.
    Coordinates must fit in ``order`` bits (0 to ``2**order - 1``).

    :Parameters:
    x, y, z : numpy uint64 arrays, shape (n,)
    order : int, optional
        Number of bits per dimension (default 21)

    :Returns:
    codes : numpy uint64 array, shape (n,)
    """
    cdef np.ndarray[uint64_t, ndim=1] xa = np.ascontiguousarray(x.ravel(), dtype=np.uint64)
    cdef np.ndarray[uint64_t, ndim=1] ya = np.ascontiguousarray(y.ravel(), dtype=np.uint64)
    cdef np.ndarray[uint64_t, ndim=1] za = np.ascontiguousarray(z.ravel(), dtype=np.uint64)
    cdef uint64_t n = <uint64_t>xa.shape[0]
    cdef np.ndarray[uint64_t, ndim=1] codes = np.empty(n, dtype=np.uint64)

    with nogil:
        hilbert_encode_3d(<uint64_t *>xa.data, <uint64_t *>ya.data, <uint64_t *>za.data,
                          n, <int32_t>order, <uint64_t *>codes.data)
    return codes


def voxel_downsample(np.ndarray points not None, voxel_size, bint return_inverse=True):
    """Downsample a point cloud using a voxel grid.

    For each occupied voxel, selects one representative point.
    Points are quantized to a voxel grid and deduplicated using
    Morton-coded keys.

    :Parameters:
    points : numpy array, shape (n, 3)
        Point coordinates (float32 or float64)
    voxel_size : float
        Voxel edge length
    return_inverse : bool, optional
        If True, also return the inverse map (default True)

    :Returns:
    selected : numpy uint64 array, shape (n_unique,)
        Indices of selected representative points
    inverse : numpy uint64 array, shape (n,)
        Maps each input point to its voxel index.
        Only returned if ``return_inverse=True``.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')
    if voxel_size <= 0:
        raise ValueError('voxel_size must be positive')

    cdef uint64_t n = <uint64_t>points.shape[0]
    cdef float c_vs_f = <float>voxel_size
    cdef double c_vs_d = <double>voxel_size
    cdef np.ndarray[uint64_t, ndim=1] selected = np.empty(n, dtype=np.uint64)
    cdef np.ndarray[uint64_t, ndim=1] inverse
    cdef uint64_t *inverse_ptr
    cdef uint64_t n_unique = 0

    if return_inverse:
        inverse = np.empty(n, dtype=np.uint64)
        inverse_ptr = <uint64_t *>inverse.data
    else:
        inverse = None
        inverse_ptr = NULL

    cdef np.ndarray[float, ndim=1] pts_float
    cdef np.ndarray[double, ndim=1] pts_double

    if points.dtype == np.float32:
        pts_float = np.ascontiguousarray(points.ravel(), dtype=np.float32)
        with nogil:
            voxel_downsample_float(<float *>pts_float.data, n, c_vs_f,
                                   <uint64_t *>selected.data, inverse_ptr, &n_unique)
    else:
        pts_double = np.ascontiguousarray(points.ravel(), dtype=np.float64)
        with nogil:
            voxel_downsample_double(<double *>pts_double.data, n, c_vs_d,
                                    <uint64_t *>selected.data, inverse_ptr, &n_unique)

    selected = selected[:n_unique]
    if return_inverse:
        return selected, inverse
    return selected


def assign_tiles(np.ndarray points not None, tile_size):
    """Assign each point to its tile using floor division on X, Y.

    :Parameters:
    points : numpy array, shape (n, 3)
        Point coordinates (float32 or float64)
    tile_size : float
        Tile edge length in coordinate units

    :Returns:
    tile_x : numpy int32 array, shape (n,)
    tile_y : numpy int32 array, shape (n,)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')
    if tile_size <= 0:
        raise ValueError('tile_size must be positive')

    cdef uint64_t n = <uint64_t>points.shape[0]
    cdef float c_ts_f = <float>tile_size
    cdef double c_ts_d = <double>tile_size
    cdef np.ndarray[int32_t, ndim=1] tx = np.empty(n, dtype=np.int32)
    cdef np.ndarray[int32_t, ndim=1] ty = np.empty(n, dtype=np.int32)

    cdef np.ndarray[float, ndim=1] pts_float
    cdef np.ndarray[double, ndim=1] pts_double

    if points.dtype == np.float32:
        pts_float = np.ascontiguousarray(points.ravel(), dtype=np.float32)
        with nogil:
            assign_tiles_float(<float *>pts_float.data, n, c_ts_f,
                               <int32_t *>tx.data, <int32_t *>ty.data)
    else:
        pts_double = np.ascontiguousarray(points.ravel(), dtype=np.float64)
        with nogil:
            assign_tiles_double(<double *>pts_double.data, n, c_ts_d,
                                <int32_t *>tx.data, <int32_t *>ty.data)

    return tx, ty


def scatter_minmax(np.ndarray points_xy not None, np.ndarray values not None,
                   grid_shape, resolution, origin):
    """Scatter values onto a 2D grid computing min, max, and count per cell.

    :Parameters:
    points_xy : numpy array, shape (n, 2)
        Point X, Y coordinates
    values : numpy array, shape (n,)
        Values to scatter
    grid_shape : tuple (height, width)
        Grid dimensions in cells
    resolution : float
        Cell size in coordinate units
    origin : tuple (x, y)
        Grid origin (lower-left corner)

    :Returns:
    min_grid : numpy array, shape (height, width)
        Minimum value per cell (inf where empty)
    max_grid : numpy array, shape (height, width)
        Maximum value per cell (-inf where empty)
    count_grid : numpy uint32 array, shape (height, width)
        Number of points per cell
    """
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError('points_xy must have shape (n, 2)')

    cdef uint32_t grid_h = <uint32_t>grid_shape[0]
    cdef uint32_t grid_w = <uint32_t>grid_shape[1]
    cdef uint64_t n = <uint64_t>points_xy.shape[0]
    cdef float c_res_f = <float>resolution
    cdef double c_res_d = <double>resolution
    cdef float c_ox_f = <float>origin[0]
    cdef float c_oy_f = <float>origin[1]
    cdef double c_ox_d = <double>origin[0]
    cdef double c_oy_d = <double>origin[1]

    cdef np.ndarray[float, ndim=1] xy_float, val_float
    cdef np.ndarray[double, ndim=1] xy_double, val_double
    cdef np.ndarray[float, ndim=1] min_f, max_f
    cdef np.ndarray[double, ndim=1] min_d, max_d
    cdef np.ndarray[uint32_t, ndim=1] count = np.zeros(grid_h * grid_w, dtype=np.uint32)

    if points_xy.dtype == np.float32:
        xy_float = np.ascontiguousarray(points_xy.ravel(), dtype=np.float32)
        val_float = np.ascontiguousarray(values.ravel(), dtype=np.float32)
        min_f = np.empty(grid_h * grid_w, dtype=np.float32)
        max_f = np.empty(grid_h * grid_w, dtype=np.float32)

        with nogil:
            scatter_minmax_float(<float *>xy_float.data, <float *>val_float.data, n,
                                 grid_h, grid_w, c_res_f, c_ox_f, c_oy_f,
                                 <float *>min_f.data, <float *>max_f.data,
                                 <uint32_t *>count.data)

        return (min_f.reshape(grid_h, grid_w),
                max_f.reshape(grid_h, grid_w),
                count.reshape(grid_h, grid_w))
    else:
        xy_double = np.ascontiguousarray(points_xy.ravel(), dtype=np.float64)
        val_double = np.ascontiguousarray(values.ravel(), dtype=np.float64)
        min_d = np.empty(grid_h * grid_w, dtype=np.float64)
        max_d = np.empty(grid_h * grid_w, dtype=np.float64)

        with nogil:
            scatter_minmax_double(<double *>xy_double.data, <double *>val_double.data, n,
                                  grid_h, grid_w, c_res_d, c_ox_d, c_oy_d,
                                  <double *>min_d.data, <double *>max_d.data,
                                  <uint32_t *>count.data)

        return (min_d.reshape(grid_h, grid_w),
                max_d.reshape(grid_h, grid_w),
                count.reshape(grid_h, grid_w))


def grid_sample_nearest(np.ndarray grid not None, np.ndarray points_xy not None,
                        resolution, origin, fill_value=0.0):
    """Sample a 2D grid at point locations using nearest-neighbor lookup.

    :Parameters:
    grid : numpy array, shape (height, width)
        Grid to sample from
    points_xy : numpy array, shape (n, 2)
        Point X, Y coordinates
    resolution : float
        Cell size in coordinate units
    origin : tuple (x, y)
        Grid origin (lower-left corner)
    fill_value : float, optional
        Value for out-of-bounds points (default 0.0)

    :Returns:
    values : numpy array, shape (n,)
    """
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError('points_xy must have shape (n, 2)')

    cdef uint32_t grid_h = <uint32_t>grid.shape[0]
    cdef uint32_t grid_w = <uint32_t>grid.shape[1]
    cdef uint64_t n = <uint64_t>points_xy.shape[0]
    cdef float c_res_f = <float>resolution
    cdef double c_res_d = <double>resolution
    cdef float c_ox_f = <float>origin[0]
    cdef float c_oy_f = <float>origin[1]
    cdef double c_ox_d = <double>origin[0]
    cdef double c_oy_d = <double>origin[1]
    cdef float c_fill_f = <float>fill_value
    cdef double c_fill_d = <double>fill_value

    cdef np.ndarray[float, ndim=1] grid_flat_f, xy_float, vals_f
    cdef np.ndarray[double, ndim=1] grid_flat_d, xy_double, vals_d

    if grid.dtype == np.float32:
        grid_flat_f = np.ascontiguousarray(grid.ravel(), dtype=np.float32)
        xy_float = np.ascontiguousarray(points_xy.ravel(), dtype=np.float32)
        vals_f = np.empty(n, dtype=np.float32)

        with nogil:
            grid_sample_nearest_float(<float *>grid_flat_f.data, grid_h, grid_w,
                                      <float *>xy_float.data, n, c_res_f,
                                      c_ox_f, c_oy_f, c_fill_f,
                                      <float *>vals_f.data)
        return vals_f
    else:
        grid_flat_d = np.ascontiguousarray(grid.ravel(), dtype=np.float64)
        xy_double = np.ascontiguousarray(points_xy.ravel(), dtype=np.float64)
        vals_d = np.empty(n, dtype=np.float64)

        with nogil:
            grid_sample_nearest_double(<double *>grid_flat_d.data, grid_h, grid_w,
                                       <double *>xy_double.data, n, c_res_d,
                                       c_ox_d, c_oy_d, c_fill_d,
                                       <double *>vals_d.data)
        return vals_d


def filter_bbox(np.ndarray points not None,
                min_x, min_y, min_z, max_x, max_y, max_z):
    """Filter points within an axis-aligned bounding box.

    :Parameters:
    points : numpy array, shape (n, 3)
    min_x, min_y, min_z : lower bounds
    max_x, max_y, max_z : upper bounds (exclusive)

    :Returns:
    mask : numpy bool array, shape (n,)
    count : int, number of points inside
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')

    cdef uint64_t n = <uint64_t>points.shape[0]
    cdef np.ndarray[uint8_t, ndim=1] mask = np.empty(n, dtype=np.uint8)
    cdef uint64_t count = 0
    cdef float c_min_x_f = <float>min_x, c_min_y_f = <float>min_y, c_min_z_f = <float>min_z
    cdef float c_max_x_f = <float>max_x, c_max_y_f = <float>max_y, c_max_z_f = <float>max_z
    cdef double c_min_x_d = <double>min_x, c_min_y_d = <double>min_y, c_min_z_d = <double>min_z
    cdef double c_max_x_d = <double>max_x, c_max_y_d = <double>max_y, c_max_z_d = <double>max_z

    cdef np.ndarray[float, ndim=1] pts_float
    cdef np.ndarray[double, ndim=1] pts_double

    if points.dtype == np.float32:
        pts_float = np.ascontiguousarray(points.ravel(), dtype=np.float32)
        with nogil:
            filter_points_in_bbox_float(<float *>pts_float.data, n,
                                        c_min_x_f, c_min_y_f, c_min_z_f,
                                        c_max_x_f, c_max_y_f, c_max_z_f,
                                        <uint8_t *>mask.data, &count)
    else:
        pts_double = np.ascontiguousarray(points.ravel(), dtype=np.float64)
        with nogil:
            filter_points_in_bbox_double(<double *>pts_double.data, n,
                                         c_min_x_d, c_min_y_d, c_min_z_d,
                                         c_max_x_d, c_max_y_d, c_max_z_d,
                                         <uint8_t *>mask.data, &count)

    return mask.view(np.bool_), int(count)
