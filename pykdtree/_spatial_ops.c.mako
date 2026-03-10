/*
pykdtree, Fast kd-tree implementation with OpenMP-enabled queries

Copyright (C) 2013 - present  Esben S. Nielsen

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation, either version 3 of the License, or
 (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
Spatial operations for point cloud processing.
OpenMP-parallelized where applicable.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include <string.h>

#define DIST_MAX_float FLT_MAX
#define DIST_MAX_double DBL_MAX

/* =========================================================
   Morton (Z-order) encoding / decoding — uint64 only
   ========================================================= */

static inline uint64_t _spread_bits_3(uint64_t v)
{
    /* Spread bits of a 21-bit value into every third bit position */
    v &= 0x1FFFFFULL;
    v = (v | (v << 32)) & 0x1F00000000FFFFULL;
    v = (v | (v << 16)) & 0x1F0000FF0000FFULL;
    v = (v | (v <<  8)) & 0x100F00F00F00F00FULL;
    v = (v | (v <<  4)) & 0x10C30C30C30C30C3ULL;
    v = (v | (v <<  2)) & 0x1249249249249249ULL;
    return v;
}

static inline uint64_t _compact_bits_3(uint64_t v)
{
    /* Compact every third bit into contiguous low bits */
    v &= 0x1249249249249249ULL;
    v = (v | (v >>  2)) & 0x10C30C30C30C30C3ULL;
    v = (v | (v >>  4)) & 0x100F00F00F00F00FULL;
    v = (v | (v >>  8)) & 0x1F0000FF0000FFULL;
    v = (v | (v >> 16)) & 0x1F00000000FFFFULL;
    v = (v | (v >> 32)) & 0x1FFFFFULL;
    return v;
}

void morton_encode_3d(uint64_t *x, uint64_t *y, uint64_t *z,
                      uint64_t n, uint64_t *codes_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;

    #pragma omp parallel for schedule(static)
    for (i = 0; i < local_n; i++)
    {
        codes_out[i] = _spread_bits_3(x[i])
                     | (_spread_bits_3(y[i]) << 1)
                     | (_spread_bits_3(z[i]) << 2);
    }
}

void morton_decode_3d(uint64_t *codes, uint64_t n,
                      uint64_t *x_out, uint64_t *y_out, uint64_t *z_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;

    #pragma omp parallel for schedule(static)
    for (i = 0; i < local_n; i++)
    {
        x_out[i] = _compact_bits_3(codes[i]);
        y_out[i] = _compact_bits_3(codes[i] >> 1);
        z_out[i] = _compact_bits_3(codes[i] >> 2);
    }
}

/* =========================================================
   Hilbert encoding — Hamilton & Rau-Chaplin algorithm, 3D
   ========================================================= */

void hilbert_encode_3d(uint64_t *x, uint64_t *y, uint64_t *z,
                       uint64_t n, int32_t order, uint64_t *codes_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;

    #pragma omp parallel for schedule(static)
    for (i = 0; i < local_n; i++)
    {
        uint64_t coords[3];
        int b, d;
        uint64_t q, p, t;
        uint64_t m = 1ULL << (order - 1);
        uint64_t code;

        coords[0] = x[i]; coords[1] = y[i]; coords[2] = z[i];

        /* Inverse undo — axes to transpose */
        for (q = m; q > 1; q >>= 1)
        {
            p = q - 1;
            for (d = 0; d < 3; d++)
            {
                if (coords[d] & q)
                    coords[0] ^= p;
                else
                {
                    t = (coords[0] ^ coords[d]) & p;
                    coords[0] ^= t;
                    coords[d] ^= t;
                }
            }
        }

        /* Gray encode */
        for (d = 1; d < 3; d++)
            coords[d] ^= coords[d - 1];

        t = 0;
        for (q = m; q > 1; q >>= 1)
        {
            if (coords[2] & q)
                t ^= (q - 1);
        }
        for (d = 0; d < 3; d++)
            coords[d] ^= t;

        /* Interleave bits → single index */
        code = 0;
        for (b = order - 1; b >= 0; b--)
        {
            for (d = 0; d < 3; d++)
                code = (code << 1) | ((coords[d] >> b) & 1);
        }

        codes_out[i] = code;
    }
}

/* =========================================================
   DTYPE-templated spatial operations
   ========================================================= */

% for DTYPE in ['float', 'double']:

/************************************************
Voxel grid downsampling.
For each point computes a Morton-coded voxel key, sorts by key,
picks the first point in each unique voxel.

Params:
    points      : (n * 3) contiguous point coords
    n           : number of points
    voxel_size  : size of each voxel
    selected_out: output indices of representative points (size n, only first *n_unique used)
    inverse_out : (n,) maps each original point to its voxel index [0, n_unique)
                  May be NULL if not needed
    n_unique_out: number of unique voxels (return)
************************************************/
void voxel_downsample_${DTYPE}(${DTYPE} *points, uint64_t n,
                               ${DTYPE} voxel_size,
                               uint64_t *selected_out,
                               uint64_t *inverse_out,
                               uint64_t *n_unique_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;

    if (n == 0) { *n_unique_out = 0; return; }

    /* 1. Find bounding box minimum */
    ${DTYPE} min_x = DIST_MAX_${DTYPE};
    ${DTYPE} min_y = DIST_MAX_${DTYPE};
    ${DTYPE} min_z = DIST_MAX_${DTYPE};

    #pragma omp parallel for reduction(min:min_x,min_y,min_z) schedule(static)
    for (i = 0; i < local_n; i++)
    {
        ${DTYPE} px = points[3 * i], py = points[3 * i + 1], pz = points[3 * i + 2];
        if (px < min_x) min_x = px;
        if (py < min_y) min_y = py;
        if (pz < min_z) min_z = pz;
    }

    /* 2. Quantize and pack voxel keys (concat ix|iy|iz, 21 bits each) */
    ${DTYPE} inv_vs = 1 / voxel_size;
    uint64_t *voxel_keys = (uint64_t *)malloc(n * sizeof(uint64_t));

    #pragma omp parallel for schedule(static)
    for (i = 0; i < local_n; i++)
    {
        uint64_t ix = (uint64_t)floor((points[3 * i]     - min_x) * inv_vs);
        uint64_t iy = (uint64_t)floor((points[3 * i + 1] - min_y) * inv_vs);
        uint64_t iz = (uint64_t)floor((points[3 * i + 2] - min_z) * inv_vs);
        /* Clamp to 21-bit range */
        if (ix > 0x1FFFFFULL) ix = 0x1FFFFFULL;
        if (iy > 0x1FFFFFULL) iy = 0x1FFFFFULL;
        if (iz > 0x1FFFFFULL) iz = 0x1FFFFFULL;
        voxel_keys[i] = ix | (iy << 21) | (iz << 42);
    }

    /* 3. Open-addressing hash table to deduplicate voxel keys.
       Table size = next power of 2 >= 2*n for ~50% load factor.
       Each slot stores: key (uint64) and voxel_id (uint64).
       Empty slots marked with UINT64_MAX key. */
    uint64_t table_size = 1;
    while (table_size < 2 * n) table_size <<= 1;
    uint64_t ht_mask = table_size - 1;

    uint64_t *ht_keys = (uint64_t *)malloc(table_size * sizeof(uint64_t));
    uint64_t *ht_vid  = (uint64_t *)malloc(table_size * sizeof(uint64_t));
    memset(ht_keys, 0xFF, table_size * sizeof(uint64_t)); /* UINT64_MAX = empty */

    uint64_t n_unique = 0;
    #define HT_EMPTY 0xFFFFFFFFFFFFFFFFULL

    for (i = 0; i < local_n; i++)
    {
        uint64_t key = voxel_keys[i];
        /* splitmix64 finalizer for good distribution of small integer keys */
        uint64_t h = key;
        h ^= h >> 30;
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 27;
        h *= 0x94D049BB133111EBULL;
        h ^= h >> 31;
        h &= ht_mask;

        /* Linear probe */
        while (1)
        {
            if (ht_keys[h] == HT_EMPTY)
            {
                /* New voxel */
                ht_keys[h] = key;
                ht_vid[h] = n_unique;
                selected_out[n_unique] = (uint64_t)i;
                if (inverse_out) inverse_out[i] = n_unique;
                n_unique++;
                break;
            }
            else if (ht_keys[h] == key)
            {
                /* Existing voxel */
                if (inverse_out) inverse_out[i] = ht_vid[h];
                break;
            }
            h = (h + 1) & ht_mask;
        }
    }
    #undef HT_EMPTY

    *n_unique_out = n_unique;

    free(ht_keys);
    free(ht_vid);
    free(voxel_keys);
}

/************************************************
Assign each point to its tile based on floor division.

Params:
    points     : (n * 3) contiguous point coords (only x, y used)
    n          : number of points
    tile_size  : tile size in coordinate units
    tile_x_out : (n,) output tile X indices
    tile_y_out : (n,) output tile Y indices
************************************************/
void assign_tiles_${DTYPE}(${DTYPE} *points, uint64_t n,
                           ${DTYPE} tile_size,
                           int32_t *tile_x_out, int32_t *tile_y_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    ${DTYPE} inv_ts = 1 / tile_size;

    #pragma omp parallel for schedule(static)
    for (i = 0; i < local_n; i++)
    {
        tile_x_out[i] = (int32_t)floor(points[3 * i]     * inv_ts);
        tile_y_out[i] = (int32_t)floor(points[3 * i + 1] * inv_ts);
    }
}

/************************************************
Scatter values onto a 2D grid with min, max, and count.

Grid cell (row, col) corresponds to:
    row = floor((y - origin_y) / resolution)
    col = floor((x - origin_x) / resolution)

Params:
    points_xy   : (n * 2) contiguous point X, Y coords
    values      : (n,) values to scatter
    n           : number of points
    grid_h      : grid height (rows)
    grid_w      : grid width (columns)
    resolution  : grid cell size
    origin_x/y  : grid origin (lower-left corner)
    min_grid    : (grid_h * grid_w) output, pre-initialized to FLT/DBL_MAX
    max_grid    : (grid_h * grid_w) output, pre-initialized to -FLT/DBL_MAX
    count_grid  : (grid_h * grid_w) output, pre-initialized to 0
************************************************/
void scatter_minmax_${DTYPE}(${DTYPE} *points_xy, ${DTYPE} *values, uint64_t n,
                             uint32_t grid_h, uint32_t grid_w,
                             ${DTYPE} resolution,
                             ${DTYPE} origin_x, ${DTYPE} origin_y,
                             ${DTYPE} *min_grid, ${DTYPE} *max_grid,
                             uint32_t *count_grid)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    ${DTYPE} inv_res = 1 / resolution;
    uint64_t grid_size = (uint64_t)grid_h * grid_w;

    /* Initialize grids */
    for (i = 0; i < (int64_t)grid_size; i++)
    {
        min_grid[i] = DIST_MAX_${DTYPE};
        max_grid[i] = -DIST_MAX_${DTYPE};
        count_grid[i] = 0;
    }

    /* Scatter (single-threaded for correctness with non-atomic float ops) */
    for (i = 0; i < local_n; i++)
    {
        int32_t col = (int32_t)floor((points_xy[2 * i]     - origin_x) * inv_res);
        int32_t row = (int32_t)floor((points_xy[2 * i + 1] - origin_y) * inv_res);

        if (row >= 0 && row < (int32_t)grid_h && col >= 0 && col < (int32_t)grid_w)
        {
            uint64_t idx = (uint64_t)row * grid_w + col;
            ${DTYPE} v = values[i];
            if (v < min_grid[idx]) min_grid[idx] = v;
            if (v > max_grid[idx]) max_grid[idx] = v;
            count_grid[idx]++;
        }
    }
}

/************************************************
Sample a 2D grid at point locations (nearest-neighbor).

Params:
    grid        : (grid_h * grid_w) input grid values
    grid_h/w    : grid dimensions
    points_xy   : (n * 2) contiguous point X, Y coords
    n           : number of points
    resolution  : grid cell size
    origin_x/y  : grid origin (lower-left corner)
    fill_value  : value for out-of-bounds points
    values_out  : (n,) output sampled values
************************************************/
void grid_sample_nearest_${DTYPE}(${DTYPE} *grid, uint32_t grid_h, uint32_t grid_w,
                                  ${DTYPE} *points_xy, uint64_t n,
                                  ${DTYPE} resolution,
                                  ${DTYPE} origin_x, ${DTYPE} origin_y,
                                  ${DTYPE} fill_value,
                                  ${DTYPE} *values_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    ${DTYPE} inv_res = 1 / resolution;

    #pragma omp parallel for schedule(static)
    for (i = 0; i < local_n; i++)
    {
        int32_t col = (int32_t)floor((points_xy[2 * i]     - origin_x) * inv_res);
        int32_t row = (int32_t)floor((points_xy[2 * i + 1] - origin_y) * inv_res);

        if (row >= 0 && row < (int32_t)grid_h && col >= 0 && col < (int32_t)grid_w)
            values_out[i] = grid[(uint64_t)row * grid_w + col];
        else
            values_out[i] = fill_value;
    }
}

/************************************************
Filter points within an axis-aligned bounding box.

Params:
    points    : (n * 3) contiguous point coords
    n         : number of points
    min_x/y/z : lower bounds
    max_x/y/z : upper bounds
    mask_out  : (n,) output boolean mask (1 = inside)
    count_out : number of points inside (return)
************************************************/
void filter_points_in_bbox_${DTYPE}(${DTYPE} *points, uint64_t n,
                                    ${DTYPE} min_x, ${DTYPE} min_y, ${DTYPE} min_z,
                                    ${DTYPE} max_x, ${DTYPE} max_y, ${DTYPE} max_z,
                                    uint8_t *mask_out, uint64_t *count_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    uint64_t count = 0;

    #pragma omp parallel for reduction(+:count) schedule(static)
    for (i = 0; i < local_n; i++)
    {
        ${DTYPE} px = points[3 * i], py = points[3 * i + 1], pz = points[3 * i + 2];
        if (px >= min_x && px < max_x && py >= min_y && py < max_y && pz >= min_z && pz < max_z)
        {
            mask_out[i] = 1;
            count++;
        }
        else
        {
            mask_out[i] = 0;
        }
    }
    if (count_out) *count_out = count;
}

% endfor
