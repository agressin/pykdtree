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
from libc.stdlib cimport malloc, free
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

cdef extern void voxelize_float(float *points, uint64_t n, float voxel_size,
                                 float *features, uint64_t n_feat, int method,
                                 float *centroids_out, float *features_out,
                                 uint64_t *inverse_out, uint64_t *n_unique_out) nogil
cdef extern void voxelize_double(double *points, uint64_t n, double voxel_size,
                                  double *features, uint64_t n_feat, int method,
                                  double *centroids_out, double *features_out,
                                  uint64_t *inverse_out, uint64_t *n_unique_out) nogil

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

cdef extern void concat_masked(void **data_ptrs, uint8_t **mask_ptrs,
                                uint64_t *sizes, uint64_t n_arrays,
                                uint64_t elem_size,
                                void *out, uint64_t *total_out) nogil

cdef extern void merge_tiles_float(float **tile_x_ptrs, float **tile_y_ptrs,
                                    float **tile_z_ptrs,
                                    uint64_t *tile_sizes,
                                    float *tile_offsets, uint64_t n_tiles,
                                    float roi_min_x, float roi_min_y,
                                    float roi_max_x, float roi_max_y,
                                    int use_roi,
                                    float *xyz_out, uint8_t **mask_ptrs,
                                    uint64_t *tile_counts_out, uint64_t *total_out) nogil
cdef extern void merge_tiles_double(double **tile_x_ptrs, double **tile_y_ptrs,
                                     double **tile_z_ptrs,
                                     uint64_t *tile_sizes,
                                     double *tile_offsets, uint64_t n_tiles,
                                     double roi_min_x, double roi_min_y,
                                     double roi_max_x, double roi_max_y,
                                     int use_roi,
                                     double *xyz_out, uint8_t **mask_ptrs,
                                     uint64_t *tile_counts_out, uint64_t *total_out) nogil


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


def merge_tiles(list tiles not None, np.ndarray offsets not None,
                roi=None, dict attributes=None):
    """Merge multiple point cloud tiles with ROI filtering and coordinate transform.

    Fuses ROI masking, coordinate transformation, and concatenation into a
    single C/OpenMP call, avoiding intermediate arrays.

    Accepts separate x, y, z columns per tile — directly compatible with
    PyArrow ``.to_numpy()`` output, no ``column_stack`` needed.

    :Parameters:
    tiles : list of tuples (x, y, z)
        Each element is a tuple of three 1D numpy arrays (float32 or float64).
        Can also be (n, 3) arrays for convenience (auto-split into columns).
    offsets : numpy array, shape (n_tiles, 3)
        Per-tile offset vectors (tile_origin - target_origin).
    roi : tuple (min_x, min_y, max_x, max_y) or None
        ROI bounds in target (scene-local) coordinates.
        If None, all points are included.
    attributes : dict of {name: list of arrays} or None
        Per-tile attribute columns. Each key maps to a list of N arrays
        (one per tile). Masks are applied and arrays concatenated automatically.
        Example: ``{"intensity": [i0, i1, ...], "class": [c0, c1, ...]}``

    :Returns:
    xyz : numpy array, shape (n_total, 3)
        Merged and transformed point coordinates (interleaved).
    attrs : dict of {name: array} or None
        Merged attribute arrays (only if ``attributes`` was provided).
    masks : list of numpy bool arrays
        Per-tile masks indicating which points survived ROI filtering.
    """
    cdef uint64_t n_tiles = <uint64_t>len(tiles)
    if n_tiles == 0:
        return np.empty((0, 3), dtype=np.float32), []
    if offsets.ndim != 2 or offsets.shape[0] != <int>n_tiles or offsets.shape[1] != 3:
        raise ValueError('offsets must have shape (n_tiles, 3)')

    # Normalize input: accept (x, y, z) tuples or (n, 3) arrays
    cdef list tile_x_list = []
    cdef list tile_y_list = []
    cdef list tile_z_list = []
    cdef uint64_t total_max = 0

    for t in range(n_tiles):
        item = tiles[t]
        if isinstance(item, np.ndarray) and item.ndim == 2 and item.shape[1] == 3:
            tile_x_list.append(item[:, 0])
            tile_y_list.append(item[:, 1])
            tile_z_list.append(item[:, 2])
            total_max += item.shape[0]
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            tile_x_list.append(np.asarray(item[0]))
            tile_y_list.append(np.asarray(item[1]))
            tile_z_list.append(np.asarray(item[2]))
            total_max += len(item[0])
        else:
            raise ValueError(f'tile {t}: expected (x, y, z) tuple or (n, 3) array')

    # Determine dtype from first tile
    cdef bint is_float32 = (tile_x_list[0].dtype == np.float32)

    # Ensure contiguous + right dtype, store references
    cdef list cx_list = []
    cdef list cy_list = []
    cdef list cz_list = []
    cdef np.ndarray[uint64_t, ndim=1] sizes = np.empty(n_tiles, dtype=np.uint64)

    for t in range(n_tiles):
        if is_float32:
            cx_list.append(np.ascontiguousarray(tile_x_list[t], dtype=np.float32))
            cy_list.append(np.ascontiguousarray(tile_y_list[t], dtype=np.float32))
            cz_list.append(np.ascontiguousarray(tile_z_list[t], dtype=np.float32))
        else:
            cx_list.append(np.ascontiguousarray(tile_x_list[t], dtype=np.float64))
            cy_list.append(np.ascontiguousarray(tile_y_list[t], dtype=np.float64))
            cz_list.append(np.ascontiguousarray(tile_z_list[t], dtype=np.float64))
        sizes[t] = <uint64_t>len(cx_list[t])

    # Allocate mask arrays
    cdef list mask_list = []
    for t in range(n_tiles):
        mask_list.append(np.empty(sizes[t], dtype=np.uint8))

    # Build C pointer arrays
    cdef float **xptrs_f = NULL
    cdef float **yptrs_f = NULL
    cdef float **zptrs_f = NULL
    cdef double **xptrs_d = NULL
    cdef double **yptrs_d = NULL
    cdef double **zptrs_d = NULL
    cdef uint8_t **mask_ptrs_c = NULL
    cdef np.ndarray[uint64_t, ndim=1] counts = np.empty(n_tiles, dtype=np.uint64)
    cdef uint64_t total_out = 0

    # ROI params
    cdef int c_use_roi = 1 if roi is not None else 0
    cdef float c_roi_min_x_f = 0, c_roi_min_y_f = 0, c_roi_max_x_f = 0, c_roi_max_y_f = 0
    cdef double c_roi_min_x_d = 0, c_roi_min_y_d = 0, c_roi_max_x_d = 0, c_roi_max_y_d = 0
    if roi is not None:
        c_roi_min_x_f = <float>roi[0]
        c_roi_min_y_f = <float>roi[1]
        c_roi_max_x_f = <float>roi[2]
        c_roi_max_y_f = <float>roi[3]
        c_roi_min_x_d = <double>roi[0]
        c_roi_min_y_d = <double>roi[1]
        c_roi_max_x_d = <double>roi[2]
        c_roi_max_y_d = <double>roi[3]

    cdef np.ndarray[float, ndim=1] offsets_f, xyz_out_f, arr_f
    cdef np.ndarray[double, ndim=1] offsets_d, xyz_out_d, arr_d
    cdef np.ndarray[uint8_t, ndim=1] mask_arr
    cdef uint64_t t_idx

    if is_float32:
        offsets_f = np.ascontiguousarray(offsets.ravel(), dtype=np.float32)
        xyz_out_f = np.empty(total_max * 3, dtype=np.float32)

        xptrs_f = <float **>malloc(n_tiles * sizeof(float *))
        yptrs_f = <float **>malloc(n_tiles * sizeof(float *))
        zptrs_f = <float **>malloc(n_tiles * sizeof(float *))
        mask_ptrs_c = <uint8_t **>malloc(n_tiles * sizeof(uint8_t *))
        for t_idx in range(n_tiles):
            arr_f = cx_list[t_idx]; xptrs_f[t_idx] = <float *>arr_f.data
            arr_f = cy_list[t_idx]; yptrs_f[t_idx] = <float *>arr_f.data
            arr_f = cz_list[t_idx]; zptrs_f[t_idx] = <float *>arr_f.data
            mask_arr = mask_list[t_idx]; mask_ptrs_c[t_idx] = <uint8_t *>mask_arr.data

        with nogil:
            merge_tiles_float(xptrs_f, yptrs_f, zptrs_f,
                              <uint64_t *>sizes.data,
                              <float *>offsets_f.data, n_tiles,
                              c_roi_min_x_f, c_roi_min_y_f,
                              c_roi_max_x_f, c_roi_max_y_f,
                              c_use_roi,
                              <float *>xyz_out_f.data, mask_ptrs_c,
                              <uint64_t *>counts.data, &total_out)

        free(xptrs_f); free(yptrs_f); free(zptrs_f); free(mask_ptrs_c)
        result_xyz = xyz_out_f[:total_out * 3].reshape(total_out, 3)
    else:
        offsets_d = np.ascontiguousarray(offsets.ravel(), dtype=np.float64)
        xyz_out_d = np.empty(total_max * 3, dtype=np.float64)

        xptrs_d = <double **>malloc(n_tiles * sizeof(double *))
        yptrs_d = <double **>malloc(n_tiles * sizeof(double *))
        zptrs_d = <double **>malloc(n_tiles * sizeof(double *))
        mask_ptrs_c = <uint8_t **>malloc(n_tiles * sizeof(uint8_t *))
        for t_idx in range(n_tiles):
            arr_d = cx_list[t_idx]; xptrs_d[t_idx] = <double *>arr_d.data
            arr_d = cy_list[t_idx]; yptrs_d[t_idx] = <double *>arr_d.data
            arr_d = cz_list[t_idx]; zptrs_d[t_idx] = <double *>arr_d.data
            mask_arr = mask_list[t_idx]; mask_ptrs_c[t_idx] = <uint8_t *>mask_arr.data

        with nogil:
            merge_tiles_double(xptrs_d, yptrs_d, zptrs_d,
                               <uint64_t *>sizes.data,
                               <double *>offsets_d.data, n_tiles,
                               c_roi_min_x_d, c_roi_min_y_d,
                               c_roi_max_x_d, c_roi_max_y_d,
                               c_use_roi,
                               <double *>xyz_out_d.data, mask_ptrs_c,
                               <uint64_t *>counts.data, &total_out)

        free(xptrs_d); free(yptrs_d); free(zptrs_d); free(mask_ptrs_c)
        result_xyz = xyz_out_d[:total_out * 3].reshape(total_out, 3)

    # Convert masks to bool views
    result_masks = [mask_list[t][:sizes[t]].view(np.bool_) for t in range(n_tiles)]

    # Apply masks to attributes if provided
    result_attrs = None
    if attributes is not None:
        result_attrs = {}
        for attr_name, attr_arrays in attributes.items():
            result_attrs[attr_name] = apply_masks(list(attr_arrays), result_masks)

    return result_xyz, result_attrs, result_masks


def apply_masks(list arrays not None, list masks not None):
    """Concatenate arrays keeping only elements where mask is True.

    Applies the masks from :func:`merge_tiles` to attribute columns,
    producing a single merged array per attribute. Dtype-agnostic
    (works with float32, float64, uint8, int32, etc.).

    :Parameters:
    arrays : list of numpy arrays
        One 1D array per tile for a single attribute column.
    masks : list of numpy bool arrays
        Per-tile masks (as returned by :func:`merge_tiles`).

    :Returns:
    result : numpy array, 1D
        Concatenated values from all tiles, filtered by masks.

    :Example:
    >>> xyz, masks = merge_tiles(tiles, offsets, roi=roi)
    >>> intensity = apply_masks([attrs[i]["intensity"] for i in range(n)], masks)
    >>> rgb = apply_masks([attrs[i]["rgb"] for i in range(n)], masks)
    """
    cdef uint64_t n_arrays = <uint64_t>len(arrays)
    if n_arrays == 0:
        return np.empty(0)
    if len(masks) != <int>n_arrays:
        raise ValueError('arrays and masks must have the same length')

    # Ensure contiguous, keep references
    cdef list c_arrays = []
    cdef list c_masks = []
    cdef np.ndarray[uint64_t, ndim=1] sizes = np.empty(n_arrays, dtype=np.uint64)
    cdef uint64_t total_max = 0

    cdef object dtype = arrays[0].dtype
    cdef uint64_t elem_size = <uint64_t>dtype.itemsize

    for t in range(n_arrays):
        arr = np.ascontiguousarray(arrays[t])
        c_arrays.append(arr)
        m = np.ascontiguousarray(masks[t].view(np.uint8))
        c_masks.append(m)
        sizes[t] = <uint64_t>len(arr)
        total_max += len(arr)

    # Pre-allocate output
    cdef np.ndarray out = np.empty(total_max, dtype=dtype)
    cdef uint64_t total_out = 0

    # Build pointer arrays
    cdef void **data_ptrs_c = <void **>malloc(n_arrays * sizeof(void *))
    cdef uint8_t **mask_ptrs_c = <uint8_t **>malloc(n_arrays * sizeof(uint8_t *))
    cdef np.ndarray tmp_arr
    cdef np.ndarray[uint8_t, ndim=1] tmp_mask

    cdef uint64_t t_idx
    for t_idx in range(n_arrays):
        tmp_arr = c_arrays[t_idx]
        data_ptrs_c[t_idx] = <void *>tmp_arr.data
        tmp_mask = c_masks[t_idx]
        mask_ptrs_c[t_idx] = <uint8_t *>tmp_mask.data

    with nogil:
        concat_masked(data_ptrs_c, mask_ptrs_c,
                      <uint64_t *>sizes.data, n_arrays,
                      elem_size,
                      <void *>out.data, &total_out)

    free(data_ptrs_c)
    free(mask_ptrs_c)

    return out[:total_out]


def voxelize(np.ndarray points not None, voxel_size,
             np.ndarray features=None, str method='mean'):
    """Voxelize a point cloud with aggregation.

    For each occupied voxel, computes the centroid (mean position) and
    optionally aggregates feature columns using mean, max, or sum.

    :Parameters:
    points : numpy array, shape (n, 3)
        Point coordinates (float32 or float64).
    voxel_size : float
        Voxel edge length.
    features : numpy array, shape (n, n_feat), optional
        Feature columns to aggregate. Same dtype as points.
    method : str, optional
        Aggregation method: ``'mean'``, ``'max'``, or ``'sum'`` (default ``'mean'``).

    :Returns:
    centroids : numpy array, shape (n_unique, 3)
        Voxel centroids (mean of all points in each voxel).
    agg_features : numpy array, shape (n_unique, n_feat) or None
        Aggregated features per voxel (None if no features provided).
    inverse : numpy uint64 array, shape (n,)
        Maps each input point to its voxel index.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (n, 3)')
    if voxel_size <= 0:
        raise ValueError('voxel_size must be positive')

    cdef int c_method = 0
    if method == 'max':
        c_method = 1
    elif method == 'sum':
        c_method = 2
    elif method != 'mean':
        raise ValueError("method must be 'mean', 'max', or 'sum'")

    cdef uint64_t n = <uint64_t>points.shape[0]
    cdef uint64_t n_feat = 0
    cdef float c_vs_f = <float>voxel_size
    cdef double c_vs_d = <double>voxel_size
    cdef uint64_t n_unique = 0

    cdef np.ndarray[uint64_t, ndim=1] inverse = np.empty(n, dtype=np.uint64)
    cdef np.ndarray[float, ndim=1] centroids_f, features_f, feat_out_f
    cdef np.ndarray[double, ndim=1] centroids_d, features_d, feat_out_d
    cdef np.ndarray[float, ndim=1] pts_flat_f
    cdef np.ndarray[double, ndim=1] pts_flat_d
    cdef float *feat_ptr_f = NULL
    cdef float *feat_out_ptr_f = NULL
    cdef double *feat_ptr_d = NULL
    cdef double *feat_out_ptr_d = NULL

    if features is not None:
        if features.ndim == 1:
            n_feat = 1
        elif features.ndim == 2:
            n_feat = <uint64_t>features.shape[1]
        else:
            raise ValueError('features must be 1D or 2D')

    if points.dtype == np.float32:
        pts_flat_f = np.ascontiguousarray(points.ravel(), dtype=np.float32)
        centroids_f = np.empty(n * 3, dtype=np.float32)

        if n_feat > 0:
            features_f = np.ascontiguousarray(features.ravel(), dtype=np.float32)
            feat_out_f = np.empty(n * n_feat, dtype=np.float32)
            feat_ptr_f = <float *>features_f.data
            feat_out_ptr_f = <float *>feat_out_f.data

        with nogil:
            voxelize_float(<float *>pts_flat_f.data, n, c_vs_f,
                           feat_ptr_f, n_feat, c_method,
                           <float *>centroids_f.data, feat_out_ptr_f,
                           <uint64_t *>inverse.data, &n_unique)

        result_centroids = centroids_f[:n_unique * 3].reshape(n_unique, 3)
        if n_feat > 0:
            result_features = feat_out_f[:n_unique * n_feat].reshape(n_unique, n_feat)
        else:
            result_features = None
    else:
        pts_flat_d = np.ascontiguousarray(points.ravel(), dtype=np.float64)
        centroids_d = np.empty(n * 3, dtype=np.float64)

        if n_feat > 0:
            features_d = np.ascontiguousarray(features.ravel(), dtype=np.float64)
            feat_out_d = np.empty(n * n_feat, dtype=np.float64)
            feat_ptr_d = <double *>features_d.data
            feat_out_ptr_d = <double *>feat_out_d.data

        with nogil:
            voxelize_double(<double *>pts_flat_d.data, n, c_vs_d,
                            feat_ptr_d, n_feat, c_method,
                            <double *>centroids_d.data, feat_out_ptr_d,
                            <uint64_t *>inverse.data, &n_unique)

        result_centroids = centroids_d[:n_unique * 3].reshape(n_unique, 3)
        if n_feat > 0:
            result_features = feat_out_d[:n_unique * n_feat].reshape(n_unique, n_feat)
        else:
            result_features = None

    return result_centroids, result_features, inverse
