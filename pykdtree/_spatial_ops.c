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
void voxel_downsample_float(float *points, uint64_t n,
                               float voxel_size,
                               uint64_t *selected_out,
                               uint64_t *inverse_out,
                               uint64_t *n_unique_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;

    if (n == 0) { *n_unique_out = 0; return; }

    /* 1. Find bounding box minimum */
    float min_x = DIST_MAX_float;
    float min_y = DIST_MAX_float;
    float min_z = DIST_MAX_float;

    #pragma omp parallel for reduction(min:min_x,min_y,min_z) schedule(static)
    for (i = 0; i < local_n; i++)
    {
        float px = points[3 * i], py = points[3 * i + 1], pz = points[3 * i + 2];
        if (px < min_x) min_x = px;
        if (py < min_y) min_y = py;
        if (pz < min_z) min_z = pz;
    }

    /* 2. Quantize and pack voxel keys (concat ix|iy|iz, 21 bits each) */
    float inv_vs = 1 / voxel_size;
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
Voxelize with aggregation: compute voxel centroids and aggregate features.

For each occupied voxel, computes:
- centroid (mean x, y, z)
- aggregated features (mean, max, or sum)
- point-to-voxel inverse map

Params:
    points        : (n * 3) contiguous point coords
    n             : number of points
    voxel_size    : voxel edge length
    features      : (n * n_feat) contiguous features, or NULL
    n_feat        : number of feature columns (0 if no features)
    method        : aggregation method (0=mean, 1=max, 2=sum)
    centroids_out : output (n_unique * 3) voxel centroids
    features_out  : output (n_unique * n_feat) aggregated features, or NULL
    inverse_out   : (n,) maps each point to its voxel index
    n_unique_out  : number of unique voxels (return)
************************************************/
void voxelize_float(float *points, uint64_t n,
                        float voxel_size,
                        float *features, uint64_t n_feat,
                        int method,
                        float *centroids_out,
                        float *features_out,
                        uint64_t *inverse_out,
                        uint64_t *n_unique_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    uint64_t j;

    if (n == 0) { *n_unique_out = 0; return; }

    /* 1. Find bounding box minimum */
    float min_x = DIST_MAX_float;
    float min_y = DIST_MAX_float;
    float min_z = DIST_MAX_float;

    #pragma omp parallel for reduction(min:min_x,min_y,min_z) schedule(static)
    for (i = 0; i < local_n; i++)
    {
        float px = points[3 * i], py = points[3 * i + 1], pz = points[3 * i + 2];
        if (px < min_x) min_x = px;
        if (py < min_y) min_y = py;
        if (pz < min_z) min_z = pz;
    }

    /* 2. Quantize to voxel keys */
    float inv_vs = 1 / voxel_size;
    uint64_t *voxel_keys = (uint64_t *)malloc(n * sizeof(uint64_t));

    #pragma omp parallel for schedule(static)
    for (i = 0; i < local_n; i++)
    {
        uint64_t ix = (uint64_t)floor((points[3 * i]     - min_x) * inv_vs);
        uint64_t iy = (uint64_t)floor((points[3 * i + 1] - min_y) * inv_vs);
        uint64_t iz = (uint64_t)floor((points[3 * i + 2] - min_z) * inv_vs);
        if (ix > 0x1FFFFFULL) ix = 0x1FFFFFULL;
        if (iy > 0x1FFFFFULL) iy = 0x1FFFFFULL;
        if (iz > 0x1FFFFFULL) iz = 0x1FFFFFULL;
        voxel_keys[i] = ix | (iy << 21) | (iz << 42);
    }

    /* 3. Hash table: deduplicate + accumulate */
    uint64_t table_size = 1;
    while (table_size < 2 * n) table_size <<= 1;
    uint64_t ht_mask = table_size - 1;

    uint64_t *ht_keys = (uint64_t *)malloc(table_size * sizeof(uint64_t));
    uint64_t *ht_vid  = (uint64_t *)malloc(table_size * sizeof(uint64_t));
    memset(ht_keys, 0xFF, table_size * sizeof(uint64_t));

    /* Accumulators for centroids */
    uint64_t max_voxels = n;  /* upper bound */
    float *sum_x = (float *)calloc(max_voxels, sizeof(float));
    float *sum_y = (float *)calloc(max_voxels, sizeof(float));
    float *sum_z = (float *)calloc(max_voxels, sizeof(float));
    uint64_t *counts = (uint64_t *)calloc(max_voxels, sizeof(uint64_t));

    /* Feature accumulators */
    float *feat_acc = NULL;
    if (features && n_feat > 0 && features_out)
    {
        feat_acc = (float *)calloc(max_voxels * n_feat, sizeof(float));
        if (method == 1) /* max: init to -inf */
        {
            for (i = 0; i < (int64_t)(max_voxels * n_feat); i++)
                feat_acc[i] = -DIST_MAX_float;
        }
    }

    uint64_t n_unique = 0;
    #define HT_EMPTY 0xFFFFFFFFFFFFFFFFULL

    for (i = 0; i < local_n; i++)
    {
        uint64_t key = voxel_keys[i];
        uint64_t h = key;
        h ^= h >> 30;
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 27;
        h *= 0x94D049BB133111EBULL;
        h ^= h >> 31;
        h &= ht_mask;

        uint64_t vid;
        while (1)
        {
            if (ht_keys[h] == HT_EMPTY)
            {
                ht_keys[h] = key;
                vid = n_unique;
                ht_vid[h] = vid;
                n_unique++;
                break;
            }
            else if (ht_keys[h] == key)
            {
                vid = ht_vid[h];
                break;
            }
            h = (h + 1) & ht_mask;
        }

        inverse_out[i] = vid;

        /* Accumulate centroid */
        sum_x[vid] += points[3 * i];
        sum_y[vid] += points[3 * i + 1];
        sum_z[vid] += points[3 * i + 2];
        counts[vid]++;

        /* Accumulate features */
        if (feat_acc)
        {
            for (j = 0; j < n_feat; j++)
            {
                float fval = features[i * n_feat + j];
                if (method == 1) /* max */
                {
                    if (fval > feat_acc[vid * n_feat + j])
                        feat_acc[vid * n_feat + j] = fval;
                }
                else /* mean or sum */
                {
                    feat_acc[vid * n_feat + j] += fval;
                }
            }
        }
    }
    #undef HT_EMPTY

    /* 4. Finalize: compute centroids and mean features */
    #pragma omp parallel for schedule(static)
    for (i = 0; i < (int64_t)n_unique; i++)
    {
        float inv_cnt = 1.0 / counts[i];
        centroids_out[3 * i]     = sum_x[i] * inv_cnt;
        centroids_out[3 * i + 1] = sum_y[i] * inv_cnt;
        centroids_out[3 * i + 2] = sum_z[i] * inv_cnt;

        if (feat_acc && method == 0) /* mean */
        {
            for (j = 0; j < n_feat; j++)
                features_out[i * n_feat + j] = feat_acc[i * n_feat + j] * inv_cnt;
        }
        else if (feat_acc) /* max or sum: already final */
        {
            for (j = 0; j < n_feat; j++)
                features_out[i * n_feat + j] = feat_acc[i * n_feat + j];
        }
    }

    *n_unique_out = n_unique;

    free(voxel_keys);
    free(ht_keys);
    free(ht_vid);
    free(sum_x); free(sum_y); free(sum_z);
    free(counts);
    if (feat_acc) free(feat_acc);
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
void assign_tiles_float(float *points, uint64_t n,
                           float tile_size,
                           int32_t *tile_x_out, int32_t *tile_y_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    float inv_ts = 1 / tile_size;

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
void scatter_minmax_float(float *points_xy, float *values, uint64_t n,
                             uint32_t grid_h, uint32_t grid_w,
                             float resolution,
                             float origin_x, float origin_y,
                             float *min_grid, float *max_grid,
                             uint32_t *count_grid)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    float inv_res = 1 / resolution;
    uint64_t grid_size = (uint64_t)grid_h * grid_w;

    /* Initialize grids */
    for (i = 0; i < (int64_t)grid_size; i++)
    {
        min_grid[i] = DIST_MAX_float;
        max_grid[i] = -DIST_MAX_float;
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
            float v = values[i];
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
void grid_sample_nearest_float(float *grid, uint32_t grid_h, uint32_t grid_w,
                                  float *points_xy, uint64_t n,
                                  float resolution,
                                  float origin_x, float origin_y,
                                  float fill_value,
                                  float *values_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    float inv_res = 1 / resolution;

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
void filter_points_in_bbox_float(float *points, uint64_t n,
                                    float min_x, float min_y, float min_z,
                                    float max_x, float max_y, float max_z,
                                    uint8_t *mask_out, uint64_t *count_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    uint64_t count = 0;

    #pragma omp parallel for reduction(+:count) schedule(static)
    for (i = 0; i < local_n; i++)
    {
        float px = points[3 * i], py = points[3 * i + 1], pz = points[3 * i + 2];
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

/************************************************
Merge multiple tiles: ROI filter + coordinate transform + write to single buffer.

Accepts separate x, y, z columns per tile (matches Parquet column layout).
Two-pass approach:
  Pass 1: compute per-point mask and per-tile survivor count (parallel across tiles)
  Pass 2: prefix-sum offsets, then scatter surviving points with transform (parallel)

Params:
    tile_x_ptrs     : array of n_tiles pointers to x column arrays
    tile_y_ptrs     : array of n_tiles pointers to y column arrays
    tile_z_ptrs     : array of n_tiles pointers to z column arrays
    tile_sizes      : (n_tiles,) number of points per tile
    tile_offsets    : (n_tiles * 3) per-tile offset (tile_origin - target_origin)
    n_tiles         : number of tiles
    roi_min_x/y, roi_max_x/y : ROI bounds in scene-local coords
                                Set use_roi=0 to skip filtering
    use_roi         : 1 = apply ROI filter, 0 = take all points
    xyz_out         : pre-allocated output buffer (sum(tile_sizes) * 3), interleaved
    mask_ptrs       : array of n_tiles pointers to (tile_sizes[t],) uint8 mask arrays
    tile_counts_out : (n_tiles,) number of survivors per tile
    total_out       : total number of points written
************************************************/
void merge_tiles_float(float **tile_x_ptrs, float **tile_y_ptrs,
                           float **tile_z_ptrs,
                           uint64_t *tile_sizes,
                           float *tile_offsets, uint64_t n_tiles,
                           float roi_min_x, float roi_min_y,
                           float roi_max_x, float roi_max_y,
                           int use_roi,
                           float *xyz_out, uint8_t **mask_ptrs,
                           uint64_t *tile_counts_out, uint64_t *total_out)
{
    uint64_t t;

    /* Pass 1: compute masks and counts per tile */
    #pragma omp parallel for schedule(dynamic)
    for (t = 0; t < n_tiles; t++)
    {
        float *tx = tile_x_ptrs[t];
        float *ty = tile_y_ptrs[t];
        uint8_t *mask = mask_ptrs[t];
        uint64_t n = tile_sizes[t];
        float off_x = tile_offsets[3 * t];
        float off_y = tile_offsets[3 * t + 1];
        uint64_t count = 0;
        uint64_t i;

        if (use_roi)
        {
            /* ROI in tile-local coords: subtract offset to go from scene-local to tile-local */
            float local_min_x = roi_min_x - off_x;
            float local_max_x = roi_max_x - off_x;
            float local_min_y = roi_min_y - off_y;
            float local_max_y = roi_max_y - off_y;

            for (i = 0; i < n; i++)
            {
                if (tx[i] >= local_min_x && tx[i] < local_max_x &&
                    ty[i] >= local_min_y && ty[i] < local_max_y)
                {
                    mask[i] = 1;
                    count++;
                }
                else
                {
                    mask[i] = 0;
                }
            }
        }
        else
        {
            for (i = 0; i < n; i++) mask[i] = 1;
            count = n;
        }
        tile_counts_out[t] = count;
    }

    /* Prefix sum to compute write offsets */
    uint64_t *write_offsets = (uint64_t *)malloc((n_tiles + 1) * sizeof(uint64_t));
    write_offsets[0] = 0;
    for (t = 0; t < n_tiles; t++)
        write_offsets[t + 1] = write_offsets[t] + tile_counts_out[t];
    *total_out = write_offsets[n_tiles];

    /* Pass 2: transform + scatter surviving points to interleaved output */
    #pragma omp parallel for schedule(dynamic)
    for (t = 0; t < n_tiles; t++)
    {
        float *tx = tile_x_ptrs[t];
        float *ty = tile_y_ptrs[t];
        float *tz = tile_z_ptrs[t];
        uint8_t *mask = mask_ptrs[t];
        uint64_t n = tile_sizes[t];
        float off_x = tile_offsets[3 * t];
        float off_y = tile_offsets[3 * t + 1];
        float off_z = tile_offsets[3 * t + 2];
        uint64_t write_pos = write_offsets[t];
        uint64_t i;

        for (i = 0; i < n; i++)
        {
            if (mask[i])
            {
                xyz_out[3 * write_pos]     = tx[i] + off_x;
                xyz_out[3 * write_pos + 1] = ty[i] + off_y;
                xyz_out[3 * write_pos + 2] = tz[i] + off_z;
                write_pos++;
            }
        }
    }

    free(write_offsets);
}

/************************************************
Compute overlap weights by distance to ROI center.

Points in the center zone (d <= center_ratio) get weight 1.0.
Weight decreases linearly to 0.0 at ROI edge (d = 1.0).

Params:
    points_xy     : (n * 2) contiguous XY coords
    n             : number of points
    roi_cx/cy     : ROI center
    roi_half_sx/sy: ROI half-sizes
    center_ratio  : fraction of ROI considered "center" (0-1)
    weights_out   : (n,) output weights in [0, 1]
************************************************/
void compute_overlap_weights_float(float *points_xy, uint64_t n,
                                       float roi_cx, float roi_cy,
                                       float roi_half_sx, float roi_half_sy,
                                       float center_ratio,
                                       float *weights_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    float inv_falloff = 1.0 / (1.0 - center_ratio + 1e-12);

    #pragma omp parallel for schedule(static)
    for (i = 0; i < local_n; i++)
    {
        float dx = fabs(points_xy[2 * i]     - roi_cx) / roi_half_sx;
        float dy = fabs(points_xy[2 * i + 1] - roi_cy) / roi_half_sy;
        float d = dx > dy ? dx : dy;
        float w = 1.0 - (d - center_ratio) * inv_falloff;
        if (w < 0.0) w = 0.0;
        if (w > 1.0) w = 1.0;
        weights_out[i] = w;
    }
}

/************************************************
Accumulate predictions with overlap weights.

Strategies:
  0 = "center": keep prediction with highest weight per point
  1 = "mean":   weighted sum of logits/probas
  2 = "vote":   weighted vote counting per class

For "center" (strategy=0):
    preds: (m,) int32 class predictions
    acc_preds: (n_points,) int32 accumulator (init to default_value)
    acc_weights: (n_points,) float accumulator (init to 0)

For "mean" (strategy=1):
    preds: (m * n_classes) float predictions (row-major)
    acc_sum: (n_points * n_classes) float accumulator (init to 0)
    acc_weights: (n_points,) float accumulator (init to 0)

For "vote" (strategy=2):
    preds: (m,) int32 class predictions
    acc_votes: (n_points * n_classes) float accumulator (init to 0)
    acc_has_pred: (n_points,) uint8 flag (init to 0)

Params:
    row_ids       : (m,) global point indices
    weights       : (m,) overlap weights
    m             : number of points in this ROI
    strategy      : 0=center, 1=mean, 2=vote
    n_classes     : number of classes (for mean/vote)
    preds_int     : (m,) int32 predictions (center/vote)
    preds_float   : (m * n_classes) float predictions (mean)
    acc_preds_int : (n_points,) int32 accumulator (center)
    acc_weights   : (n_points,) float accumulator (center/mean)
    acc_sum       : (n_points * n_classes) float accumulator (mean)
    acc_votes     : (n_points * n_classes) float accumulator (vote)
    acc_has_pred  : (n_points,) uint8 flag (vote)
************************************************/
void accumulate_predictions_float(uint64_t *row_ids, float *weights,
                                      uint64_t m, int strategy,
                                      uint64_t n_classes,
                                      int32_t *preds_int,
                                      float *preds_float,
                                      int32_t *acc_preds_int,
                                      float *acc_weights,
                                      float *acc_sum,
                                      float *acc_votes,
                                      uint8_t *acc_has_pred)
{
    uint64_t i, j;

    if (strategy == 0)
    {
        /* center: keep highest weight — sequential due to potential conflicts */
        for (i = 0; i < m; i++)
        {
            uint64_t idx = row_ids[i];
            if (weights[i] > acc_weights[idx])
            {
                acc_weights[idx] = weights[i];
                acc_preds_int[idx] = preds_int[i];
            }
        }
    }
    else if (strategy == 1)
    {
        /* mean: weighted sum — sequential scatter */
        for (i = 0; i < m; i++)
        {
            uint64_t idx = row_ids[i];
            float w = weights[i];
            acc_weights[idx] += w;
            for (j = 0; j < n_classes; j++)
                acc_sum[idx * n_classes + j] += preds_float[i * n_classes + j] * w;
        }
    }
    else /* strategy == 2: vote */
    {
        for (i = 0; i < m; i++)
        {
            uint64_t idx = row_ids[i];
            int32_t cls = preds_int[i];
            if (cls >= 0 && (uint64_t)cls < n_classes)
            {
                acc_votes[idx * n_classes + cls] += weights[i];
                acc_has_pred[idx] = 1;
            }
        }
    }
}

/************************************************
Finalize accumulated predictions.

For "center" (strategy=0): no-op (already done during accumulate)
For "mean" (strategy=1): divide sum by weight, normalize if probas
For "vote" (strategy=2): argmax over vote counts

Params:
    n_points      : total number of points
    strategy      : 0=center, 1=mean, 2=vote
    n_classes     : number of classes
    is_probas     : 1 if output should be re-normalized as probabilities
    default_value : fill value for points with no predictions
    acc_preds_int : (n_points,) int32 (center result, or vote output)
    acc_weights   : (n_points,) float weights (mean)
    acc_sum       : (n_points * n_classes) float sums (mean input)
    result_float  : (n_points * n_classes) float output (mean)
    acc_votes     : (n_points * n_classes) float vote counts
    acc_has_pred  : (n_points,) uint8 flags (vote)
************************************************/
void finalize_predictions_float(uint64_t n_points, int strategy,
                                    uint64_t n_classes, int is_probas,
                                    float default_value,
                                    int32_t *acc_preds_int,
                                    float *acc_weights,
                                    float *acc_sum,
                                    float *result_float,
                                    float *acc_votes,
                                    uint8_t *acc_has_pred)
{
    int64_t i;
    int64_t local_n = (int64_t)n_points;
    uint64_t j;

    if (strategy == 1) /* mean */
    {
        #pragma omp parallel for schedule(static) private(j)
        for (i = 0; i < local_n; i++)
        {
            if (acc_weights[i] > 0)
            {
                float inv_w = 1.0 / acc_weights[i];
                float row_sum = 0;
                for (j = 0; j < n_classes; j++)
                {
                    result_float[i * n_classes + j] = acc_sum[i * n_classes + j] * inv_w;
                    row_sum += result_float[i * n_classes + j];
                }
                if (is_probas && row_sum > 1e-8)
                {
                    float inv_sum = 1.0 / row_sum;
                    for (j = 0; j < n_classes; j++)
                        result_float[i * n_classes + j] *= inv_sum;
                }
            }
            else
            {
                for (j = 0; j < n_classes; j++)
                    result_float[i * n_classes + j] = default_value;
            }
        }
    }
    else if (strategy == 2) /* vote */
    {
        #pragma omp parallel for schedule(static) private(j)
        for (i = 0; i < local_n; i++)
        {
            if (acc_has_pred[i])
            {
                int32_t best_cls = 0;
                float best_val = acc_votes[i * n_classes];
                for (j = 1; j < n_classes; j++)
                {
                    if (acc_votes[i * n_classes + j] > best_val)
                    {
                        best_val = acc_votes[i * n_classes + j];
                        best_cls = (int32_t)j;
                    }
                }
                acc_preds_int[i] = best_cls;
            }
            /* else: already default_value */
        }
    }
    /* strategy == 0 (center): nothing to do */
}

/************************************************
Filter points to core bbox and scatter results to original indices.

Fuses:
  1. Core mask computation (2D bbox filter on XY)
  2. Scatter attributes by original row indices

Params:
    points_xy     : (n * 2) contiguous XY coords of all points (buffer+core)
    n             : number of points
    core_min_x/y  : core bbox lower bounds
    core_max_x/y  : core bbox upper bounds (exclusive)
    row_indices   : (n,) original row indices in the output array
    src_values    : (n * n_cols) source values to scatter (contiguous, row-major)
    n_cols        : number of columns to scatter
    dst_values    : (n_dst * n_cols) output array (pre-allocated)
    core_mask_out : (n,) output boolean mask (1 = inside core)
    n_core_out    : number of points in core (return)
************************************************/
void unbuffer_and_scatter_float(float *points_xy, uint64_t n,
                                    float core_min_x, float core_min_y,
                                    float core_max_x, float core_max_y,
                                    uint64_t *row_indices,
                                    float *src_values, uint64_t n_cols,
                                    float *dst_values,
                                    uint8_t *core_mask_out,
                                    uint64_t *n_core_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    uint64_t j;
    uint64_t n_core = 0;

    /* Pass 1: compute core mask (parallel) */
    #pragma omp parallel for reduction(+:n_core) schedule(static)
    for (i = 0; i < local_n; i++)
    {
        float px = points_xy[2 * i];
        float py = points_xy[2 * i + 1];
        if (px >= core_min_x && px < core_max_x &&
            py >= core_min_y && py < core_max_y)
        {
            core_mask_out[i] = 1;
            n_core++;
        }
        else
        {
            core_mask_out[i] = 0;
        }
    }

    *n_core_out = n_core;

    /* Pass 2: scatter core values to destination (parallel) */
    if (src_values && dst_values && row_indices)
    {
        #pragma omp parallel for schedule(static) private(j)
        for (i = 0; i < local_n; i++)
        {
            if (core_mask_out[i])
            {
                uint64_t dst_idx = row_indices[i];
                for (j = 0; j < n_cols; j++)
                    dst_values[dst_idx * n_cols + j] = src_values[i * n_cols + j];
            }
        }
    }
}


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
void voxel_downsample_double(double *points, uint64_t n,
                               double voxel_size,
                               uint64_t *selected_out,
                               uint64_t *inverse_out,
                               uint64_t *n_unique_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;

    if (n == 0) { *n_unique_out = 0; return; }

    /* 1. Find bounding box minimum */
    double min_x = DIST_MAX_double;
    double min_y = DIST_MAX_double;
    double min_z = DIST_MAX_double;

    #pragma omp parallel for reduction(min:min_x,min_y,min_z) schedule(static)
    for (i = 0; i < local_n; i++)
    {
        double px = points[3 * i], py = points[3 * i + 1], pz = points[3 * i + 2];
        if (px < min_x) min_x = px;
        if (py < min_y) min_y = py;
        if (pz < min_z) min_z = pz;
    }

    /* 2. Quantize and pack voxel keys (concat ix|iy|iz, 21 bits each) */
    double inv_vs = 1 / voxel_size;
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
Voxelize with aggregation: compute voxel centroids and aggregate features.

For each occupied voxel, computes:
- centroid (mean x, y, z)
- aggregated features (mean, max, or sum)
- point-to-voxel inverse map

Params:
    points        : (n * 3) contiguous point coords
    n             : number of points
    voxel_size    : voxel edge length
    features      : (n * n_feat) contiguous features, or NULL
    n_feat        : number of feature columns (0 if no features)
    method        : aggregation method (0=mean, 1=max, 2=sum)
    centroids_out : output (n_unique * 3) voxel centroids
    features_out  : output (n_unique * n_feat) aggregated features, or NULL
    inverse_out   : (n,) maps each point to its voxel index
    n_unique_out  : number of unique voxels (return)
************************************************/
void voxelize_double(double *points, uint64_t n,
                        double voxel_size,
                        double *features, uint64_t n_feat,
                        int method,
                        double *centroids_out,
                        double *features_out,
                        uint64_t *inverse_out,
                        uint64_t *n_unique_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    uint64_t j;

    if (n == 0) { *n_unique_out = 0; return; }

    /* 1. Find bounding box minimum */
    double min_x = DIST_MAX_double;
    double min_y = DIST_MAX_double;
    double min_z = DIST_MAX_double;

    #pragma omp parallel for reduction(min:min_x,min_y,min_z) schedule(static)
    for (i = 0; i < local_n; i++)
    {
        double px = points[3 * i], py = points[3 * i + 1], pz = points[3 * i + 2];
        if (px < min_x) min_x = px;
        if (py < min_y) min_y = py;
        if (pz < min_z) min_z = pz;
    }

    /* 2. Quantize to voxel keys */
    double inv_vs = 1 / voxel_size;
    uint64_t *voxel_keys = (uint64_t *)malloc(n * sizeof(uint64_t));

    #pragma omp parallel for schedule(static)
    for (i = 0; i < local_n; i++)
    {
        uint64_t ix = (uint64_t)floor((points[3 * i]     - min_x) * inv_vs);
        uint64_t iy = (uint64_t)floor((points[3 * i + 1] - min_y) * inv_vs);
        uint64_t iz = (uint64_t)floor((points[3 * i + 2] - min_z) * inv_vs);
        if (ix > 0x1FFFFFULL) ix = 0x1FFFFFULL;
        if (iy > 0x1FFFFFULL) iy = 0x1FFFFFULL;
        if (iz > 0x1FFFFFULL) iz = 0x1FFFFFULL;
        voxel_keys[i] = ix | (iy << 21) | (iz << 42);
    }

    /* 3. Hash table: deduplicate + accumulate */
    uint64_t table_size = 1;
    while (table_size < 2 * n) table_size <<= 1;
    uint64_t ht_mask = table_size - 1;

    uint64_t *ht_keys = (uint64_t *)malloc(table_size * sizeof(uint64_t));
    uint64_t *ht_vid  = (uint64_t *)malloc(table_size * sizeof(uint64_t));
    memset(ht_keys, 0xFF, table_size * sizeof(uint64_t));

    /* Accumulators for centroids */
    uint64_t max_voxels = n;  /* upper bound */
    double *sum_x = (double *)calloc(max_voxels, sizeof(double));
    double *sum_y = (double *)calloc(max_voxels, sizeof(double));
    double *sum_z = (double *)calloc(max_voxels, sizeof(double));
    uint64_t *counts = (uint64_t *)calloc(max_voxels, sizeof(uint64_t));

    /* Feature accumulators */
    double *feat_acc = NULL;
    if (features && n_feat > 0 && features_out)
    {
        feat_acc = (double *)calloc(max_voxels * n_feat, sizeof(double));
        if (method == 1) /* max: init to -inf */
        {
            for (i = 0; i < (int64_t)(max_voxels * n_feat); i++)
                feat_acc[i] = -DIST_MAX_double;
        }
    }

    uint64_t n_unique = 0;
    #define HT_EMPTY 0xFFFFFFFFFFFFFFFFULL

    for (i = 0; i < local_n; i++)
    {
        uint64_t key = voxel_keys[i];
        uint64_t h = key;
        h ^= h >> 30;
        h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 27;
        h *= 0x94D049BB133111EBULL;
        h ^= h >> 31;
        h &= ht_mask;

        uint64_t vid;
        while (1)
        {
            if (ht_keys[h] == HT_EMPTY)
            {
                ht_keys[h] = key;
                vid = n_unique;
                ht_vid[h] = vid;
                n_unique++;
                break;
            }
            else if (ht_keys[h] == key)
            {
                vid = ht_vid[h];
                break;
            }
            h = (h + 1) & ht_mask;
        }

        inverse_out[i] = vid;

        /* Accumulate centroid */
        sum_x[vid] += points[3 * i];
        sum_y[vid] += points[3 * i + 1];
        sum_z[vid] += points[3 * i + 2];
        counts[vid]++;

        /* Accumulate features */
        if (feat_acc)
        {
            for (j = 0; j < n_feat; j++)
            {
                double fval = features[i * n_feat + j];
                if (method == 1) /* max */
                {
                    if (fval > feat_acc[vid * n_feat + j])
                        feat_acc[vid * n_feat + j] = fval;
                }
                else /* mean or sum */
                {
                    feat_acc[vid * n_feat + j] += fval;
                }
            }
        }
    }
    #undef HT_EMPTY

    /* 4. Finalize: compute centroids and mean features */
    #pragma omp parallel for schedule(static)
    for (i = 0; i < (int64_t)n_unique; i++)
    {
        double inv_cnt = 1.0 / counts[i];
        centroids_out[3 * i]     = sum_x[i] * inv_cnt;
        centroids_out[3 * i + 1] = sum_y[i] * inv_cnt;
        centroids_out[3 * i + 2] = sum_z[i] * inv_cnt;

        if (feat_acc && method == 0) /* mean */
        {
            for (j = 0; j < n_feat; j++)
                features_out[i * n_feat + j] = feat_acc[i * n_feat + j] * inv_cnt;
        }
        else if (feat_acc) /* max or sum: already final */
        {
            for (j = 0; j < n_feat; j++)
                features_out[i * n_feat + j] = feat_acc[i * n_feat + j];
        }
    }

    *n_unique_out = n_unique;

    free(voxel_keys);
    free(ht_keys);
    free(ht_vid);
    free(sum_x); free(sum_y); free(sum_z);
    free(counts);
    if (feat_acc) free(feat_acc);
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
void assign_tiles_double(double *points, uint64_t n,
                           double tile_size,
                           int32_t *tile_x_out, int32_t *tile_y_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    double inv_ts = 1 / tile_size;

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
void scatter_minmax_double(double *points_xy, double *values, uint64_t n,
                             uint32_t grid_h, uint32_t grid_w,
                             double resolution,
                             double origin_x, double origin_y,
                             double *min_grid, double *max_grid,
                             uint32_t *count_grid)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    double inv_res = 1 / resolution;
    uint64_t grid_size = (uint64_t)grid_h * grid_w;

    /* Initialize grids */
    for (i = 0; i < (int64_t)grid_size; i++)
    {
        min_grid[i] = DIST_MAX_double;
        max_grid[i] = -DIST_MAX_double;
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
            double v = values[i];
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
void grid_sample_nearest_double(double *grid, uint32_t grid_h, uint32_t grid_w,
                                  double *points_xy, uint64_t n,
                                  double resolution,
                                  double origin_x, double origin_y,
                                  double fill_value,
                                  double *values_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    double inv_res = 1 / resolution;

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
void filter_points_in_bbox_double(double *points, uint64_t n,
                                    double min_x, double min_y, double min_z,
                                    double max_x, double max_y, double max_z,
                                    uint8_t *mask_out, uint64_t *count_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    uint64_t count = 0;

    #pragma omp parallel for reduction(+:count) schedule(static)
    for (i = 0; i < local_n; i++)
    {
        double px = points[3 * i], py = points[3 * i + 1], pz = points[3 * i + 2];
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

/************************************************
Merge multiple tiles: ROI filter + coordinate transform + write to single buffer.

Accepts separate x, y, z columns per tile (matches Parquet column layout).
Two-pass approach:
  Pass 1: compute per-point mask and per-tile survivor count (parallel across tiles)
  Pass 2: prefix-sum offsets, then scatter surviving points with transform (parallel)

Params:
    tile_x_ptrs     : array of n_tiles pointers to x column arrays
    tile_y_ptrs     : array of n_tiles pointers to y column arrays
    tile_z_ptrs     : array of n_tiles pointers to z column arrays
    tile_sizes      : (n_tiles,) number of points per tile
    tile_offsets    : (n_tiles * 3) per-tile offset (tile_origin - target_origin)
    n_tiles         : number of tiles
    roi_min_x/y, roi_max_x/y : ROI bounds in scene-local coords
                                Set use_roi=0 to skip filtering
    use_roi         : 1 = apply ROI filter, 0 = take all points
    xyz_out         : pre-allocated output buffer (sum(tile_sizes) * 3), interleaved
    mask_ptrs       : array of n_tiles pointers to (tile_sizes[t],) uint8 mask arrays
    tile_counts_out : (n_tiles,) number of survivors per tile
    total_out       : total number of points written
************************************************/
void merge_tiles_double(double **tile_x_ptrs, double **tile_y_ptrs,
                           double **tile_z_ptrs,
                           uint64_t *tile_sizes,
                           double *tile_offsets, uint64_t n_tiles,
                           double roi_min_x, double roi_min_y,
                           double roi_max_x, double roi_max_y,
                           int use_roi,
                           double *xyz_out, uint8_t **mask_ptrs,
                           uint64_t *tile_counts_out, uint64_t *total_out)
{
    uint64_t t;

    /* Pass 1: compute masks and counts per tile */
    #pragma omp parallel for schedule(dynamic)
    for (t = 0; t < n_tiles; t++)
    {
        double *tx = tile_x_ptrs[t];
        double *ty = tile_y_ptrs[t];
        uint8_t *mask = mask_ptrs[t];
        uint64_t n = tile_sizes[t];
        double off_x = tile_offsets[3 * t];
        double off_y = tile_offsets[3 * t + 1];
        uint64_t count = 0;
        uint64_t i;

        if (use_roi)
        {
            /* ROI in tile-local coords: subtract offset to go from scene-local to tile-local */
            double local_min_x = roi_min_x - off_x;
            double local_max_x = roi_max_x - off_x;
            double local_min_y = roi_min_y - off_y;
            double local_max_y = roi_max_y - off_y;

            for (i = 0; i < n; i++)
            {
                if (tx[i] >= local_min_x && tx[i] < local_max_x &&
                    ty[i] >= local_min_y && ty[i] < local_max_y)
                {
                    mask[i] = 1;
                    count++;
                }
                else
                {
                    mask[i] = 0;
                }
            }
        }
        else
        {
            for (i = 0; i < n; i++) mask[i] = 1;
            count = n;
        }
        tile_counts_out[t] = count;
    }

    /* Prefix sum to compute write offsets */
    uint64_t *write_offsets = (uint64_t *)malloc((n_tiles + 1) * sizeof(uint64_t));
    write_offsets[0] = 0;
    for (t = 0; t < n_tiles; t++)
        write_offsets[t + 1] = write_offsets[t] + tile_counts_out[t];
    *total_out = write_offsets[n_tiles];

    /* Pass 2: transform + scatter surviving points to interleaved output */
    #pragma omp parallel for schedule(dynamic)
    for (t = 0; t < n_tiles; t++)
    {
        double *tx = tile_x_ptrs[t];
        double *ty = tile_y_ptrs[t];
        double *tz = tile_z_ptrs[t];
        uint8_t *mask = mask_ptrs[t];
        uint64_t n = tile_sizes[t];
        double off_x = tile_offsets[3 * t];
        double off_y = tile_offsets[3 * t + 1];
        double off_z = tile_offsets[3 * t + 2];
        uint64_t write_pos = write_offsets[t];
        uint64_t i;

        for (i = 0; i < n; i++)
        {
            if (mask[i])
            {
                xyz_out[3 * write_pos]     = tx[i] + off_x;
                xyz_out[3 * write_pos + 1] = ty[i] + off_y;
                xyz_out[3 * write_pos + 2] = tz[i] + off_z;
                write_pos++;
            }
        }
    }

    free(write_offsets);
}

/************************************************
Compute overlap weights by distance to ROI center.

Points in the center zone (d <= center_ratio) get weight 1.0.
Weight decreases linearly to 0.0 at ROI edge (d = 1.0).

Params:
    points_xy     : (n * 2) contiguous XY coords
    n             : number of points
    roi_cx/cy     : ROI center
    roi_half_sx/sy: ROI half-sizes
    center_ratio  : fraction of ROI considered "center" (0-1)
    weights_out   : (n,) output weights in [0, 1]
************************************************/
void compute_overlap_weights_double(double *points_xy, uint64_t n,
                                       double roi_cx, double roi_cy,
                                       double roi_half_sx, double roi_half_sy,
                                       double center_ratio,
                                       double *weights_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    double inv_falloff = 1.0 / (1.0 - center_ratio + 1e-12);

    #pragma omp parallel for schedule(static)
    for (i = 0; i < local_n; i++)
    {
        double dx = fabs(points_xy[2 * i]     - roi_cx) / roi_half_sx;
        double dy = fabs(points_xy[2 * i + 1] - roi_cy) / roi_half_sy;
        double d = dx > dy ? dx : dy;
        double w = 1.0 - (d - center_ratio) * inv_falloff;
        if (w < 0.0) w = 0.0;
        if (w > 1.0) w = 1.0;
        weights_out[i] = w;
    }
}

/************************************************
Accumulate predictions with overlap weights.

Strategies:
  0 = "center": keep prediction with highest weight per point
  1 = "mean":   weighted sum of logits/probas
  2 = "vote":   weighted vote counting per class

For "center" (strategy=0):
    preds: (m,) int32 class predictions
    acc_preds: (n_points,) int32 accumulator (init to default_value)
    acc_weights: (n_points,) float accumulator (init to 0)

For "mean" (strategy=1):
    preds: (m * n_classes) float predictions (row-major)
    acc_sum: (n_points * n_classes) float accumulator (init to 0)
    acc_weights: (n_points,) float accumulator (init to 0)

For "vote" (strategy=2):
    preds: (m,) int32 class predictions
    acc_votes: (n_points * n_classes) float accumulator (init to 0)
    acc_has_pred: (n_points,) uint8 flag (init to 0)

Params:
    row_ids       : (m,) global point indices
    weights       : (m,) overlap weights
    m             : number of points in this ROI
    strategy      : 0=center, 1=mean, 2=vote
    n_classes     : number of classes (for mean/vote)
    preds_int     : (m,) int32 predictions (center/vote)
    preds_float   : (m * n_classes) float predictions (mean)
    acc_preds_int : (n_points,) int32 accumulator (center)
    acc_weights   : (n_points,) float accumulator (center/mean)
    acc_sum       : (n_points * n_classes) float accumulator (mean)
    acc_votes     : (n_points * n_classes) float accumulator (vote)
    acc_has_pred  : (n_points,) uint8 flag (vote)
************************************************/
void accumulate_predictions_double(uint64_t *row_ids, double *weights,
                                      uint64_t m, int strategy,
                                      uint64_t n_classes,
                                      int32_t *preds_int,
                                      double *preds_float,
                                      int32_t *acc_preds_int,
                                      double *acc_weights,
                                      double *acc_sum,
                                      double *acc_votes,
                                      uint8_t *acc_has_pred)
{
    uint64_t i, j;

    if (strategy == 0)
    {
        /* center: keep highest weight — sequential due to potential conflicts */
        for (i = 0; i < m; i++)
        {
            uint64_t idx = row_ids[i];
            if (weights[i] > acc_weights[idx])
            {
                acc_weights[idx] = weights[i];
                acc_preds_int[idx] = preds_int[i];
            }
        }
    }
    else if (strategy == 1)
    {
        /* mean: weighted sum — sequential scatter */
        for (i = 0; i < m; i++)
        {
            uint64_t idx = row_ids[i];
            double w = weights[i];
            acc_weights[idx] += w;
            for (j = 0; j < n_classes; j++)
                acc_sum[idx * n_classes + j] += preds_float[i * n_classes + j] * w;
        }
    }
    else /* strategy == 2: vote */
    {
        for (i = 0; i < m; i++)
        {
            uint64_t idx = row_ids[i];
            int32_t cls = preds_int[i];
            if (cls >= 0 && (uint64_t)cls < n_classes)
            {
                acc_votes[idx * n_classes + cls] += weights[i];
                acc_has_pred[idx] = 1;
            }
        }
    }
}

/************************************************
Finalize accumulated predictions.

For "center" (strategy=0): no-op (already done during accumulate)
For "mean" (strategy=1): divide sum by weight, normalize if probas
For "vote" (strategy=2): argmax over vote counts

Params:
    n_points      : total number of points
    strategy      : 0=center, 1=mean, 2=vote
    n_classes     : number of classes
    is_probas     : 1 if output should be re-normalized as probabilities
    default_value : fill value for points with no predictions
    acc_preds_int : (n_points,) int32 (center result, or vote output)
    acc_weights   : (n_points,) float weights (mean)
    acc_sum       : (n_points * n_classes) float sums (mean input)
    result_float  : (n_points * n_classes) float output (mean)
    acc_votes     : (n_points * n_classes) float vote counts
    acc_has_pred  : (n_points,) uint8 flags (vote)
************************************************/
void finalize_predictions_double(uint64_t n_points, int strategy,
                                    uint64_t n_classes, int is_probas,
                                    double default_value,
                                    int32_t *acc_preds_int,
                                    double *acc_weights,
                                    double *acc_sum,
                                    double *result_float,
                                    double *acc_votes,
                                    uint8_t *acc_has_pred)
{
    int64_t i;
    int64_t local_n = (int64_t)n_points;
    uint64_t j;

    if (strategy == 1) /* mean */
    {
        #pragma omp parallel for schedule(static) private(j)
        for (i = 0; i < local_n; i++)
        {
            if (acc_weights[i] > 0)
            {
                double inv_w = 1.0 / acc_weights[i];
                double row_sum = 0;
                for (j = 0; j < n_classes; j++)
                {
                    result_float[i * n_classes + j] = acc_sum[i * n_classes + j] * inv_w;
                    row_sum += result_float[i * n_classes + j];
                }
                if (is_probas && row_sum > 1e-8)
                {
                    double inv_sum = 1.0 / row_sum;
                    for (j = 0; j < n_classes; j++)
                        result_float[i * n_classes + j] *= inv_sum;
                }
            }
            else
            {
                for (j = 0; j < n_classes; j++)
                    result_float[i * n_classes + j] = default_value;
            }
        }
    }
    else if (strategy == 2) /* vote */
    {
        #pragma omp parallel for schedule(static) private(j)
        for (i = 0; i < local_n; i++)
        {
            if (acc_has_pred[i])
            {
                int32_t best_cls = 0;
                double best_val = acc_votes[i * n_classes];
                for (j = 1; j < n_classes; j++)
                {
                    if (acc_votes[i * n_classes + j] > best_val)
                    {
                        best_val = acc_votes[i * n_classes + j];
                        best_cls = (int32_t)j;
                    }
                }
                acc_preds_int[i] = best_cls;
            }
            /* else: already default_value */
        }
    }
    /* strategy == 0 (center): nothing to do */
}

/************************************************
Filter points to core bbox and scatter results to original indices.

Fuses:
  1. Core mask computation (2D bbox filter on XY)
  2. Scatter attributes by original row indices

Params:
    points_xy     : (n * 2) contiguous XY coords of all points (buffer+core)
    n             : number of points
    core_min_x/y  : core bbox lower bounds
    core_max_x/y  : core bbox upper bounds (exclusive)
    row_indices   : (n,) original row indices in the output array
    src_values    : (n * n_cols) source values to scatter (contiguous, row-major)
    n_cols        : number of columns to scatter
    dst_values    : (n_dst * n_cols) output array (pre-allocated)
    core_mask_out : (n,) output boolean mask (1 = inside core)
    n_core_out    : number of points in core (return)
************************************************/
void unbuffer_and_scatter_double(double *points_xy, uint64_t n,
                                    double core_min_x, double core_min_y,
                                    double core_max_x, double core_max_y,
                                    uint64_t *row_indices,
                                    double *src_values, uint64_t n_cols,
                                    double *dst_values,
                                    uint8_t *core_mask_out,
                                    uint64_t *n_core_out)
{
    int64_t i;
    int64_t local_n = (int64_t)n;
    uint64_t j;
    uint64_t n_core = 0;

    /* Pass 1: compute core mask (parallel) */
    #pragma omp parallel for reduction(+:n_core) schedule(static)
    for (i = 0; i < local_n; i++)
    {
        double px = points_xy[2 * i];
        double py = points_xy[2 * i + 1];
        if (px >= core_min_x && px < core_max_x &&
            py >= core_min_y && py < core_max_y)
        {
            core_mask_out[i] = 1;
            n_core++;
        }
        else
        {
            core_mask_out[i] = 0;
        }
    }

    *n_core_out = n_core;

    /* Pass 2: scatter core values to destination (parallel) */
    if (src_values && dst_values && row_indices)
    {
        #pragma omp parallel for schedule(static) private(j)
        for (i = 0; i < local_n; i++)
        {
            if (core_mask_out[i])
            {
                uint64_t dst_idx = row_indices[i];
                for (j = 0; j < n_cols; j++)
                    dst_values[dst_idx * n_cols + j] = src_values[i * n_cols + j];
            }
        }
    }
}


/* =========================================================
   Dtype-agnostic masked concatenation
   ========================================================= */

/************************************************
Concatenate multiple arrays, keeping only elements where mask == 1.
Works with any dtype by operating on raw bytes.

Params:
    data_ptrs   : array of n_arrays pointers to source data
    mask_ptrs   : array of n_arrays pointers to uint8 mask arrays
    sizes       : (n_arrays,) number of elements per array
    n_arrays    : number of arrays
    elem_size   : size of one element in bytes (e.g. 4 for float32)
    out         : pre-allocated output buffer
    total_out   : total number of elements written (return)
************************************************/
void concat_masked(void **data_ptrs, uint8_t **mask_ptrs,
                   uint64_t *sizes, uint64_t n_arrays,
                   uint64_t elem_size,
                   void *out, uint64_t *total_out)
{
    uint64_t t;

    /* Pass 1: count survivors per array for write offsets */
    uint64_t *counts = (uint64_t *)malloc(n_arrays * sizeof(uint64_t));
    #pragma omp parallel for schedule(dynamic)
    for (t = 0; t < n_arrays; t++)
    {
        uint8_t *mask = mask_ptrs[t];
        uint64_t n = sizes[t];
        uint64_t c = 0;
        uint64_t i;
        for (i = 0; i < n; i++)
            c += mask[i];
        counts[t] = c;
    }

    /* Prefix sum */
    uint64_t *write_offsets = (uint64_t *)malloc((n_arrays + 1) * sizeof(uint64_t));
    write_offsets[0] = 0;
    for (t = 0; t < n_arrays; t++)
        write_offsets[t + 1] = write_offsets[t] + counts[t];
    *total_out = write_offsets[n_arrays];

    /* Pass 2: scatter surviving elements */
    #pragma omp parallel for schedule(dynamic)
    for (t = 0; t < n_arrays; t++)
    {
        uint8_t *src = (uint8_t *)data_ptrs[t];
        uint8_t *mask = mask_ptrs[t];
        uint8_t *dst = (uint8_t *)out + write_offsets[t] * elem_size;
        uint64_t n = sizes[t];
        uint64_t write_pos = 0;
        uint64_t i;

        for (i = 0; i < n; i++)
        {
            if (mask[i])
            {
                memcpy(dst + write_pos * elem_size, src + i * elem_size, elem_size);
                write_pos++;
            }
        }
    }

    free(counts);
    free(write_offsets);
}
