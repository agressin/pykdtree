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
This kd-tree implementation is based on the scipy.spatial.cKDTree by
Anne M. Archibald and libANN by David M. Mount and Sunil Arya.
*/


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NUM_DESCRIPTORS 20

#define PA(i,d)			(pa[no_dims * pidx[i] + d])
#define PASWAP_int32_t(a,b) { uint32_t tmp = pidx[a]; pidx[a] = pidx[b]; pidx[b] = tmp; }
#define PASWAP_int64_t(a,b) { uint64_t tmp = pidx[a]; pidx[a] = pidx[b]; pidx[b] = tmp; }

#define IDX_MAX_int32_t UINT32_MAX
#define IDX_MAX_int64_t UINT64_MAX
#define DIST_MAX_float FLT_MAX
#define DIST_MAX_double DBL_MAX

#ifdef _MSC_VER
#define restrict __restrict
#endif


typedef struct
{
    float cut_val;
    int8_t cut_dim;
    uint32_t start_idx;
    uint32_t n;
    float cut_bounds_lv;
    float cut_bounds_hv;
    struct Node_float_int32_t *left_child;
    struct Node_float_int32_t *right_child;
} Node_float_int32_t;

typedef struct
{
    float *bbox;
    int8_t no_dims;
    uint32_t *pidx;
    struct Node_float_int32_t *root;
} Tree_float_int32_t;


typedef struct
{
    float cut_val;
    int8_t cut_dim;
    uint64_t start_idx;
    uint64_t n;
    float cut_bounds_lv;
    float cut_bounds_hv;
    struct Node_float_int64_t *left_child;
    struct Node_float_int64_t *right_child;
} Node_float_int64_t;

typedef struct
{
    float *bbox;
    int8_t no_dims;
    uint64_t *pidx;
    struct Node_float_int64_t *root;
} Tree_float_int64_t;


typedef struct
{
    double cut_val;
    int8_t cut_dim;
    uint32_t start_idx;
    uint32_t n;
    double cut_bounds_lv;
    double cut_bounds_hv;
    struct Node_double_int32_t *left_child;
    struct Node_double_int32_t *right_child;
} Node_double_int32_t;

typedef struct
{
    double *bbox;
    int8_t no_dims;
    uint32_t *pidx;
    struct Node_double_int32_t *root;
} Tree_double_int32_t;


typedef struct
{
    double cut_val;
    int8_t cut_dim;
    uint64_t start_idx;
    uint64_t n;
    double cut_bounds_lv;
    double cut_bounds_hv;
    struct Node_double_int64_t *left_child;
    struct Node_double_int64_t *right_child;
} Node_double_int64_t;

typedef struct
{
    double *bbox;
    int8_t no_dims;
    uint64_t *pidx;
    struct Node_double_int64_t *root;
} Tree_double_int64_t;



float calc_dist_float(float *point1_coord, float *point2_coord, int8_t no_dims);
float get_cube_offset_float(int8_t dim, float *point_coord, float *bbox);
float get_min_dist_float(float *point_coord, int8_t no_dims, float *bbox);
void eigen_symmetric_3x3_float(float cov_xx, float cov_xy, float cov_xz,
                                   float cov_yy, float cov_yz, float cov_zz,
                                   float *evals, float *normal);


void insert_point_float_int32_t(uint32_t *closest_idx, float *closest_dist, uint32_t pidx, float cur_dist, uint32_t k);
void get_bounding_box_float_int32_t(float *pa, uint32_t *pidx, int8_t no_dims, uint32_t n, float *bbox);
int partition_float_int32_t(float *pa, uint32_t *pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, float *bbox, int8_t *cut_dim,
              float *cut_val, uint32_t *n_lo);
Tree_float_int32_t* construct_tree_float_int32_t(float *pa, int8_t no_dims, uint32_t n, uint32_t bsp);
Node_float_int32_t* construct_subtree_float_int32_t(float *pa, uint32_t *pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, uint32_t bsp, float *bbox);
Node_float_int32_t * create_node_float_int32_t(uint32_t start_idx, uint32_t n, int is_leaf);
void delete_subtree_float_int32_t(Node_float_int32_t *root);
void delete_tree_float_int32_t(Tree_float_int32_t *tree);
void print_tree_float_int32_t(Node_float_int32_t *root, int level);
void search_leaf_float_int32_t(float *restrict pa, uint32_t *restrict pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, float *restrict point_coord,
                 uint32_t k, uint32_t *restrict closest_idx, float *restrict closest_dist);
void search_leaf_float_int32_t_mask(float *restrict pa, uint32_t *restrict pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, float *restrict point_coord,
                 uint32_t k, uint8_t *restrict mask, uint32_t *restrict closest_idx, float *restrict closest_dist);
void search_splitnode_float_int32_t(Node_float_int32_t *root, float *pa, uint32_t *pidx, int8_t no_dims, float *point_coord,
                      float min_dist, uint32_t k, float distance_upper_bound, float eps_fac, uint8_t *mask, uint32_t *  closest_idx, float *closest_dist);
void search_tree_float_int32_t(Tree_float_int32_t *tree, float *pa, float *point_coords,
                 uint32_t num_points, uint32_t k,  float distance_upper_bound,
                 float eps, uint8_t *mask, uint32_t *closest_idxs, float *closest_dists);
void sor_mean_dists_float_int32_t(Tree_float_int32_t *tree, float *pa,
                 uint32_t num_points, uint32_t k, float *mean_dists_out);
void compute_descriptors_multiscale_float_int32_t(Tree_float_int32_t *tree, float *pa, float *point_coords,
                 uint32_t num_points, uint32_t k_max,
                 int32_t *k_scales, int32_t num_scales,
                 float distance_upper_bound, float eps, uint8_t *mask,
                 float *descriptors_out);
void radius_filter_float_int32_t(Tree_float_int32_t *tree, float *pa,
                 uint32_t num_points, uint32_t k_min, float radius,
                 uint8_t *inlier_mask_out, uint32_t *count_out);
void estimate_normals_float_int32_t(Tree_float_int32_t *tree, float *pa,
                 float *point_coords, uint32_t num_points, uint32_t k,
                 float distance_upper_bound, float eps, uint8_t *mask,
                 float *normals_out, float *curvatures_out);


void insert_point_float_int64_t(uint64_t *closest_idx, float *closest_dist, uint64_t pidx, float cur_dist, uint64_t k);
void get_bounding_box_float_int64_t(float *pa, uint64_t *pidx, int8_t no_dims, uint64_t n, float *bbox);
int partition_float_int64_t(float *pa, uint64_t *pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, float *bbox, int8_t *cut_dim,
              float *cut_val, uint64_t *n_lo);
Tree_float_int64_t* construct_tree_float_int64_t(float *pa, int8_t no_dims, uint64_t n, uint64_t bsp);
Node_float_int64_t* construct_subtree_float_int64_t(float *pa, uint64_t *pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, uint64_t bsp, float *bbox);
Node_float_int64_t * create_node_float_int64_t(uint64_t start_idx, uint64_t n, int is_leaf);
void delete_subtree_float_int64_t(Node_float_int64_t *root);
void delete_tree_float_int64_t(Tree_float_int64_t *tree);
void print_tree_float_int64_t(Node_float_int64_t *root, int level);
void search_leaf_float_int64_t(float *restrict pa, uint64_t *restrict pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, float *restrict point_coord,
                 uint64_t k, uint64_t *restrict closest_idx, float *restrict closest_dist);
void search_leaf_float_int64_t_mask(float *restrict pa, uint64_t *restrict pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, float *restrict point_coord,
                 uint64_t k, uint8_t *restrict mask, uint64_t *restrict closest_idx, float *restrict closest_dist);
void search_splitnode_float_int64_t(Node_float_int64_t *root, float *pa, uint64_t *pidx, int8_t no_dims, float *point_coord,
                      float min_dist, uint64_t k, float distance_upper_bound, float eps_fac, uint8_t *mask, uint64_t *  closest_idx, float *closest_dist);
void search_tree_float_int64_t(Tree_float_int64_t *tree, float *pa, float *point_coords,
                 uint64_t num_points, uint64_t k,  float distance_upper_bound,
                 float eps, uint8_t *mask, uint64_t *closest_idxs, float *closest_dists);
void sor_mean_dists_float_int64_t(Tree_float_int64_t *tree, float *pa,
                 uint64_t num_points, uint64_t k, float *mean_dists_out);
void compute_descriptors_multiscale_float_int64_t(Tree_float_int64_t *tree, float *pa, float *point_coords,
                 uint64_t num_points, uint64_t k_max,
                 int32_t *k_scales, int32_t num_scales,
                 float distance_upper_bound, float eps, uint8_t *mask,
                 float *descriptors_out);
void radius_filter_float_int64_t(Tree_float_int64_t *tree, float *pa,
                 uint64_t num_points, uint64_t k_min, float radius,
                 uint8_t *inlier_mask_out, uint64_t *count_out);
void estimate_normals_float_int64_t(Tree_float_int64_t *tree, float *pa,
                 float *point_coords, uint64_t num_points, uint64_t k,
                 float distance_upper_bound, float eps, uint8_t *mask,
                 float *normals_out, float *curvatures_out);


double calc_dist_double(double *point1_coord, double *point2_coord, int8_t no_dims);
double get_cube_offset_double(int8_t dim, double *point_coord, double *bbox);
double get_min_dist_double(double *point_coord, int8_t no_dims, double *bbox);
void eigen_symmetric_3x3_double(double cov_xx, double cov_xy, double cov_xz,
                                   double cov_yy, double cov_yz, double cov_zz,
                                   double *evals, double *normal);


void insert_point_double_int32_t(uint32_t *closest_idx, double *closest_dist, uint32_t pidx, double cur_dist, uint32_t k);
void get_bounding_box_double_int32_t(double *pa, uint32_t *pidx, int8_t no_dims, uint32_t n, double *bbox);
int partition_double_int32_t(double *pa, uint32_t *pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, double *bbox, int8_t *cut_dim,
              double *cut_val, uint32_t *n_lo);
Tree_double_int32_t* construct_tree_double_int32_t(double *pa, int8_t no_dims, uint32_t n, uint32_t bsp);
Node_double_int32_t* construct_subtree_double_int32_t(double *pa, uint32_t *pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, uint32_t bsp, double *bbox);
Node_double_int32_t * create_node_double_int32_t(uint32_t start_idx, uint32_t n, int is_leaf);
void delete_subtree_double_int32_t(Node_double_int32_t *root);
void delete_tree_double_int32_t(Tree_double_int32_t *tree);
void print_tree_double_int32_t(Node_double_int32_t *root, int level);
void search_leaf_double_int32_t(double *restrict pa, uint32_t *restrict pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, double *restrict point_coord,
                 uint32_t k, uint32_t *restrict closest_idx, double *restrict closest_dist);
void search_leaf_double_int32_t_mask(double *restrict pa, uint32_t *restrict pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, double *restrict point_coord,
                 uint32_t k, uint8_t *restrict mask, uint32_t *restrict closest_idx, double *restrict closest_dist);
void search_splitnode_double_int32_t(Node_double_int32_t *root, double *pa, uint32_t *pidx, int8_t no_dims, double *point_coord,
                      double min_dist, uint32_t k, double distance_upper_bound, double eps_fac, uint8_t *mask, uint32_t *  closest_idx, double *closest_dist);
void search_tree_double_int32_t(Tree_double_int32_t *tree, double *pa, double *point_coords,
                 uint32_t num_points, uint32_t k,  double distance_upper_bound,
                 double eps, uint8_t *mask, uint32_t *closest_idxs, double *closest_dists);
void sor_mean_dists_double_int32_t(Tree_double_int32_t *tree, double *pa,
                 uint32_t num_points, uint32_t k, double *mean_dists_out);
void compute_descriptors_multiscale_double_int32_t(Tree_double_int32_t *tree, double *pa, double *point_coords,
                 uint32_t num_points, uint32_t k_max,
                 int32_t *k_scales, int32_t num_scales,
                 double distance_upper_bound, double eps, uint8_t *mask,
                 double *descriptors_out);
void radius_filter_double_int32_t(Tree_double_int32_t *tree, double *pa,
                 uint32_t num_points, uint32_t k_min, double radius,
                 uint8_t *inlier_mask_out, uint32_t *count_out);
void estimate_normals_double_int32_t(Tree_double_int32_t *tree, double *pa,
                 double *point_coords, uint32_t num_points, uint32_t k,
                 double distance_upper_bound, double eps, uint8_t *mask,
                 double *normals_out, double *curvatures_out);


void insert_point_double_int64_t(uint64_t *closest_idx, double *closest_dist, uint64_t pidx, double cur_dist, uint64_t k);
void get_bounding_box_double_int64_t(double *pa, uint64_t *pidx, int8_t no_dims, uint64_t n, double *bbox);
int partition_double_int64_t(double *pa, uint64_t *pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, double *bbox, int8_t *cut_dim,
              double *cut_val, uint64_t *n_lo);
Tree_double_int64_t* construct_tree_double_int64_t(double *pa, int8_t no_dims, uint64_t n, uint64_t bsp);
Node_double_int64_t* construct_subtree_double_int64_t(double *pa, uint64_t *pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, uint64_t bsp, double *bbox);
Node_double_int64_t * create_node_double_int64_t(uint64_t start_idx, uint64_t n, int is_leaf);
void delete_subtree_double_int64_t(Node_double_int64_t *root);
void delete_tree_double_int64_t(Tree_double_int64_t *tree);
void print_tree_double_int64_t(Node_double_int64_t *root, int level);
void search_leaf_double_int64_t(double *restrict pa, uint64_t *restrict pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, double *restrict point_coord,
                 uint64_t k, uint64_t *restrict closest_idx, double *restrict closest_dist);
void search_leaf_double_int64_t_mask(double *restrict pa, uint64_t *restrict pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, double *restrict point_coord,
                 uint64_t k, uint8_t *restrict mask, uint64_t *restrict closest_idx, double *restrict closest_dist);
void search_splitnode_double_int64_t(Node_double_int64_t *root, double *pa, uint64_t *pidx, int8_t no_dims, double *point_coord,
                      double min_dist, uint64_t k, double distance_upper_bound, double eps_fac, uint8_t *mask, uint64_t *  closest_idx, double *closest_dist);
void search_tree_double_int64_t(Tree_double_int64_t *tree, double *pa, double *point_coords,
                 uint64_t num_points, uint64_t k,  double distance_upper_bound,
                 double eps, uint8_t *mask, uint64_t *closest_idxs, double *closest_dists);
void sor_mean_dists_double_int64_t(Tree_double_int64_t *tree, double *pa,
                 uint64_t num_points, uint64_t k, double *mean_dists_out);
void compute_descriptors_multiscale_double_int64_t(Tree_double_int64_t *tree, double *pa, double *point_coords,
                 uint64_t num_points, uint64_t k_max,
                 int32_t *k_scales, int32_t num_scales,
                 double distance_upper_bound, double eps, uint8_t *mask,
                 double *descriptors_out);
void radius_filter_double_int64_t(Tree_double_int64_t *tree, double *pa,
                 uint64_t num_points, uint64_t k_min, double radius,
                 uint8_t *inlier_mask_out, uint64_t *count_out);
void estimate_normals_double_int64_t(Tree_double_int64_t *tree, double *pa,
                 double *point_coords, uint64_t num_points, uint64_t k,
                 double distance_upper_bound, double eps, uint8_t *mask,
                 double *normals_out, double *curvatures_out);



/************************************************
Calculate squared cartesian distance between points
Params:
    point1_coord : point 1
    point2_coord : point 2
************************************************/
float calc_dist_float(float *point1_coord, float *point2_coord, int8_t no_dims)
{
    /* Calculate squared distance */
    float dist = 0, dim_dist;
    int8_t i;
    for (i = 0; i < no_dims; i++)
    {
        dim_dist = point2_coord[i] - point1_coord[i];
        dist += dim_dist * dim_dist;
    }
    return dist;
}

/************************************************
Get squared distance from point to cube in specified dimension
Params:
    dim : dimension
    point_coord : cartesian coordinates of point
    bbox : cube
************************************************/
float get_cube_offset_float(int8_t dim, float *point_coord, float *bbox)
{
    float dim_coord = point_coord[dim];

    if (dim_coord < bbox[2 * dim])
    {
        /* Left of cube in dimension */
        return dim_coord - bbox[2 * dim];
    }
    else if (dim_coord > bbox[2 * dim + 1])
    {
        /* Right of cube in dimension */
        return dim_coord - bbox[2 * dim + 1];
    }
    else
    {
        /* Inside cube in dimension */
        return 0.;
    }
}

/************************************************
Get minimum squared distance between point and cube.
Params:
    point_coord : cartesian coordinates of point
    no_dims : number of dimensions
    bbox : cube
************************************************/
float get_min_dist_float(float *point_coord, int8_t no_dims, float *bbox)
{
    float cube_offset = 0, cube_offset_dim;
    int8_t i;

    for (i = 0; i < no_dims; i++)
    {
        cube_offset_dim = get_cube_offset_float(i, point_coord, bbox);
        cube_offset += cube_offset_dim * cube_offset_dim;
    }

    return cube_offset;
}

/************************************************
Eigendecomposition of 3x3 symmetric matrix using Cardano's formula.
Eigenvalues returned sorted: evals[0] >= evals[1] >= evals[2].
Normal is the eigenvector of the smallest eigenvalue.
Params:
    cov_xx..cov_zz : upper triangle of symmetric matrix
    evals : eigenvalues output (3 values)
    normal : eigenvector of smallest eigenvalue (3 values)
************************************************/
void eigen_symmetric_3x3_float(float cov_xx, float cov_xy, float cov_xz,
                                   float cov_yy, float cov_yz, float cov_zz,
                                   float *evals, float *normal)
{
    float e1, e2, e3;
    float p1 = cov_xy * cov_xy + cov_xz * cov_xz + cov_yz * cov_yz;
    float q = (cov_xx + cov_yy + cov_zz) / 3;
    float p2 = (cov_xx - q) * (cov_xx - q) + (cov_yy - q) * (cov_yy - q) +
                  (cov_zz - q) * (cov_zz - q) + 2 * p1;
    float p = sqrt(p2 / 6);

    if (p < (float)1e-30)
    {
        /* All eigenvalues are equal */
        e1 = e2 = e3 = q;
    }
    else
    {
        float inv_p = 1 / p;
        /* B = (1/p)(M - q*I) */
        float b00 = inv_p * (cov_xx - q);
        float b01 = inv_p * cov_xy;
        float b02 = inv_p * cov_xz;
        float b11 = inv_p * (cov_yy - q);
        float b12 = inv_p * cov_yz;
        float b22 = inv_p * (cov_zz - q);

        /* det(B) / 2 */
        float r = (b00 * (b11 * b22 - b12 * b12)
                     - b01 * (b01 * b22 - b12 * b02)
                     + b02 * (b01 * b12 - b11 * b02)) / 2;

        /* Clamp for numerical stability */
        if (r <= -1) r = -1;
        else if (r >= 1) r = 1;

        float phi = acos(r) / 3;

        e1 = q + 2 * p * cos(phi);
        e3 = q + 2 * p * cos(phi + 2 * M_PI / 3);
        e2 = 3 * q - e1 - e3;
    }

    /* Clamp negative eigenvalues to zero */
    if (e1 < 0) e1 = 0;
    if (e2 < 0) e2 = 0;
    if (e3 < 0) e3 = 0;

    evals[0] = e1;
    evals[1] = e2;
    evals[2] = e3;

    /* Compute normal: eigenvector of smallest eigenvalue e3 */
    /* Form rows of (M - e3*I) */
    float r0x = cov_xx - e3, r0y = cov_xy,       r0z = cov_xz;
    float r1x = cov_xy,       r1y = cov_yy - e3, r1z = cov_yz;
    float r2x = cov_xz,       r2y = cov_yz,       r2z = cov_zz - e3;

    /* Cross products of all row pairs, pick largest */
    float nx, ny, nz, len_sq, best_len_sq;
    float cx, cy, cz;

    /* r0 x r1 */
    nx = r0y * r1z - r0z * r1y;
    ny = r0z * r1x - r0x * r1z;
    nz = r0x * r1y - r0y * r1x;
    best_len_sq = nx * nx + ny * ny + nz * nz;

    /* r0 x r2 */
    cx = r0y * r2z - r0z * r2y;
    cy = r0z * r2x - r0x * r2z;
    cz = r0x * r2y - r0y * r2x;
    len_sq = cx * cx + cy * cy + cz * cz;
    if (len_sq > best_len_sq) { nx = cx; ny = cy; nz = cz; best_len_sq = len_sq; }

    /* r1 x r2 */
    cx = r1y * r2z - r1z * r2y;
    cy = r1z * r2x - r1x * r2z;
    cz = r1x * r2y - r1y * r2x;
    len_sq = cx * cx + cy * cy + cz * cz;
    if (len_sq > best_len_sq) { nx = cx; ny = cy; nz = cz; best_len_sq = len_sq; }

    /* Normalize */
    if (best_len_sq > (float)1e-30)
    {
        float inv_len = 1 / sqrt(best_len_sq);
        normal[0] = nx * inv_len;
        normal[1] = ny * inv_len;
        normal[2] = nz * inv_len;
    }
    else
    {
        /* Degenerate case */
        normal[0] = 0;
        normal[1] = 0;
        normal[2] = 1;
    }
}


/************************************************
Insert point into priority queue
Params:
    closest_idx : index queue
    closest_dist : distance queue
    pidx : permutation index of data points
    cur_dist : distance to point inserted
    k : number of neighbours
************************************************/
void insert_point_float_int32_t(uint32_t *closest_idx, float *closest_dist, uint32_t pidx, float cur_dist, uint32_t k)
{
    int i;
    for (i = k - 1; i > 0; i--)
    {
        if (closest_dist[i - 1] > cur_dist)
        {
            closest_dist[i] = closest_dist[i - 1];
            closest_idx[i] = closest_idx[i - 1];
        }
        else
        {
            break;
        }
    }
    closest_idx[i] = pidx;
    closest_dist[i] = cur_dist;
}

/************************************************
Get the bounding box of a set of points
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims: number of dimensions
    n : number of points
    bbox : bounding box (return)
************************************************/
void get_bounding_box_float_int32_t(float *pa, uint32_t *pidx, int8_t no_dims, uint32_t n, float *bbox)
{
    float cur;
    int8_t i, j;
    uint32_t bbox_idx, i2;

    /* Use first data point to initialize */
    for (i = 0; i < no_dims; i++)
    {
        bbox[2 * i] = bbox[2 * i + 1] = PA(0, i);
    }

    /* Update using rest of data points */
    for (i2 = 1; i2 < n; i2++)
    {
        for (j = 0; j < no_dims; j++)
        {
            bbox_idx = 2 * j;
            cur = PA(i2, j);
            if (cur < bbox[bbox_idx])
            {
                bbox[bbox_idx] = cur;
            }
            else if (cur > bbox[bbox_idx + 1])
            {
                bbox[bbox_idx + 1] = cur;
            }
        }
    }
}

/************************************************
Partition a range of data points by manipulation the permutation index.
The sliding midpoint rule is used for the partitioning.
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims: number of dimensions
    start_idx : index of first data point to use
    n :  number of data points
    bbox : bounding box of data points
    cut_dim : dimension used for partition (return)
    cut_val : value of cutting point (return)
    n_lo : number of point below cutting plane (return)
************************************************/
int partition_float_int32_t(float *pa, uint32_t *pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, float *bbox, int8_t *cut_dim, float *cut_val, uint32_t *n_lo)
{
    int8_t dim = 0, i;
    uint32_t p, q, i2;
    float size = 0, min_val, max_val, split, side_len, cur_val;
    uint32_t end_idx = start_idx + n - 1;

    /* Find largest bounding box side */
    for (i = 0; i < no_dims; i++)
    {
        side_len = bbox[2 * i + 1] - bbox[2 * i];
        if (side_len > size)
        {
            dim = i;
            size = side_len;
        }
    }

    min_val = bbox[2 * dim];
    max_val = bbox[2 * dim + 1];

    /* Check for zero length or inconsistent */
    if (min_val >= max_val)
        return 1;

    /* Use middle for splitting */
    split = (min_val + max_val) / 2;

    /* Partition all data points around middle */
    p = start_idx;
    q = end_idx;
    while (p <= q)
    {
        if (PA(p, dim) < split)
        {
            p++;
        }
        else if (PA(q, dim) >= split)
        {
            /* Guard for underflow */
            if (q > 0)
            {
                q--;
            }
            else
            {
                break;
            }
        }
        else
        {
            PASWAP_int32_t(p, q);
            p++;
            q--;
        }
    }

    /* Check for empty splits */
    if (p == start_idx)
    {
        /* No points less than split.
           Split at lowest point instead.
           Minimum 1 point will be in lower box.
        */

        uint32_t j = start_idx;
        split = PA(j, dim);
        for (i2 = start_idx + 1; i2 <= end_idx; i2++)
        {
            /* Find lowest point */
            cur_val = PA(i2, dim);
            if (cur_val < split)
            {
                j = i2;
                split = cur_val;
            }
        }
        PASWAP_int32_t(j, start_idx);
        p = start_idx + 1;
    }
    else if (p == end_idx + 1)
    {
        /* No points greater than split.
           Split at highest point instead.
           Minimum 1 point will be in higher box.
        */

        uint32_t j = end_idx;
        split = PA(j, dim);
        for (i2 = start_idx; i2 < end_idx; i2++)
        {
            /* Find highest point */
            cur_val = PA(i2, dim);
            if (cur_val > split)
            {
                j = i2;
                split = cur_val;
            }
        }
        PASWAP_int32_t(j, end_idx);
        p = end_idx;
    }

    /* Set return values */
    *cut_dim = dim;
    *cut_val = split;
    *n_lo = p - start_idx;
    return 0;
}

/************************************************
Construct a sub tree over a range of data points.
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims: number of dimensions
    start_idx : index of first data point to use
    n :  number of data points
    bsp : number of points per leaf
    bbox : bounding box of set of data points
************************************************/
Node_float_int32_t* construct_subtree_float_int32_t(float *pa, uint32_t *pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, uint32_t bsp, float *bbox)
{
    /* Create new node */
    int is_leaf = (n <= bsp);
    Node_float_int32_t *root = create_node_float_int32_t(start_idx, n, is_leaf);
    int rval;
    int8_t cut_dim;
    uint32_t n_lo;
    float cut_val, lv, hv;
    if (is_leaf)
    {
        /* Make leaf node */
        root->cut_dim = -1;
    }
    else
    {
        /* Make split node */
        /* Partition data set and set node info */
        rval = partition_float_int32_t(pa, pidx, no_dims, start_idx, n, bbox, &cut_dim, &cut_val, &n_lo);
        if (rval == 1)
        {
            root->cut_dim = -1;
            return root;
        }
        root->cut_val = cut_val;
        root->cut_dim = cut_dim;

        /* Recurse on both subsets */
        lv = bbox[2 * cut_dim];
        hv = bbox[2 * cut_dim + 1];

        /* Set bounds for cut dimension */
        root->cut_bounds_lv = lv;
        root->cut_bounds_hv = hv;

        /* Update bounding box before call to lower subset and restore after */
        bbox[2 * cut_dim + 1] = cut_val;
        root->left_child = (struct Node_float_int32_t *)construct_subtree_float_int32_t(pa, pidx, no_dims, start_idx, n_lo, bsp, bbox);
        bbox[2 * cut_dim + 1] = hv;

        /* Update bounding box before call to higher subset and restore after */
        bbox[2 * cut_dim] = cut_val;
        root->right_child = (struct Node_float_int32_t *)construct_subtree_float_int32_t(pa, pidx, no_dims, start_idx + n_lo, n - n_lo, bsp, bbox);
        bbox[2 * cut_dim] = lv;
    }
    return root;
}

/************************************************
Construct a tree over data points.
Params:
    pa : data points
    no_dims: number of dimensions
    n :  number of data points
    bsp : number of points per leaf
************************************************/
Tree_float_int32_t* construct_tree_float_int32_t(float *pa, int8_t no_dims, uint32_t n, uint32_t bsp)
{
    Tree_float_int32_t *tree = (Tree_float_int32_t *)malloc(sizeof(Tree_float_int32_t));
    uint32_t i;
    uint32_t *pidx;
    float *bbox;

    tree->no_dims = no_dims;

    /* Initialize permutation array */
    pidx = (uint32_t *)malloc(sizeof(uint32_t) * n);
    for (i = 0; i < n; i++)
    {
        pidx[i] = i;
    }

    bbox = (float *)malloc(2 * sizeof(float) * no_dims);
    get_bounding_box_float_int32_t(pa, pidx, no_dims, n, bbox);
    tree->bbox = bbox;

    /* Construct subtree on full dataset */
    tree->root = (struct Node_float_int32_t *)construct_subtree_float_int32_t(pa, pidx, no_dims, 0, n, bsp, bbox);

    tree->pidx = pidx;
    return tree;
}

/************************************************
Create a tree node.
Params:
    start_idx : index of first data point to use
    n :  number of data points
************************************************/
Node_float_int32_t* create_node_float_int32_t(uint32_t start_idx, uint32_t n, int is_leaf)
{
    Node_float_int32_t *new_node;
    if (is_leaf)
    {
        /*
            Allocate only the part of the struct that will be used in a leaf node.
            This relies on the C99 specification of struct layout conservation and padding and
            that dereferencing is never attempted for the node pointers in a leaf.
        */
        new_node = (Node_float_int32_t *)malloc(sizeof(Node_float_int32_t) - 2 * sizeof(Node_float_int32_t *));
    }
    else
    {
        new_node = (Node_float_int32_t *)malloc(sizeof(Node_float_int32_t));
    }
    new_node->n = n;
    new_node->start_idx = start_idx;
    return new_node;
}

/************************************************
Delete subtree
Params:
    root : root node of subtree to delete
************************************************/
void delete_subtree_float_int32_t(Node_float_int32_t *root)
{
    if (root->cut_dim != -1)
    {
        delete_subtree_float_int32_t((Node_float_int32_t *)root->left_child);
        delete_subtree_float_int32_t((Node_float_int32_t *)root->right_child);
    }
    free(root);
}

/************************************************
Delete tree
Params:
    tree : Tree struct of kd tree
************************************************/
void delete_tree_float_int32_t(Tree_float_int32_t *tree)
{
    delete_subtree_float_int32_t((Node_float_int32_t *)tree->root);
    free(tree->bbox);
    free(tree->pidx);
    free(tree);
}

/************************************************
Print
************************************************/
void print_tree_float_int32_t(Node_float_int32_t *root, int level)
{
    int i;
    for (i = 0; i < level; i++)
    {
        printf(" ");
    }
    printf("(cut_val: %f, cut_dim: %i)\n", root->cut_val, root->cut_dim);
    if (root->cut_dim != -1)
        print_tree_float_int32_t((Node_float_int32_t *)root->left_child, level + 1);
    if (root->cut_dim != -1)
        print_tree_float_int32_t((Node_float_int32_t *)root->right_child, level + 1);
}

/************************************************
Search a leaf node for closest point
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims : number of dimensions
    start_idx : index of first data point to use
    size :  number of data points
    point_coord : query point
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_leaf_float_int32_t(float *restrict pa, uint32_t *restrict pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, float *restrict point_coord,
                 uint32_t k, uint32_t *restrict closest_idx, float *restrict closest_dist)
{
    float cur_dist;
    uint32_t i;
    /* Loop through all points in leaf */
    for (i = 0; i < n; i++)
    {
        /* Get distance to query point */
        cur_dist = calc_dist_float(&PA(start_idx + i, 0), point_coord, no_dims);
        /* Update closest info if new point is closest so far*/
        if (cur_dist < closest_dist[k - 1])
        {
            insert_point_float_int32_t(closest_idx, closest_dist, pidx[start_idx + i], cur_dist, k);
        }
    }
}


/************************************************
Search a leaf node for closest point with data point mask
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims : number of dimensions
    start_idx : index of first data point to use
    size :  number of data points
    point_coord : query point
    mask : boolean array of invalid (True) and valid (False) data points
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_leaf_float_int32_t_mask(float *restrict pa, uint32_t *restrict pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, float *restrict point_coord,
                               uint32_t k, uint8_t *mask, uint32_t *restrict closest_idx, float *restrict closest_dist)
{
    float cur_dist;
    uint32_t i;
    /* Loop through all points in leaf */
    for (i = 0; i < n; i++)
    {
        /* Is this point masked out? */
        if (mask[pidx[start_idx + i]])
        {
            continue;
        }
        /* Get distance to query point */
        cur_dist = calc_dist_float(&PA(start_idx + i, 0), point_coord, no_dims);
        /* Update closest info if new point is closest so far*/
        if (cur_dist < closest_dist[k - 1])
        {
            insert_point_float_int32_t(closest_idx, closest_dist, pidx[start_idx + i], cur_dist, k);
        }
    }
}

/************************************************
Search subtree for nearest to query point
Params:
    root : root node of subtree
    pa : data points
    pidx : permutation index of data points
    no_dims : number of dimensions
    point_coord : query point
    min_dist : minumum distance to nearest neighbour
    mask : boolean array of invalid (True) and valid (False) data points
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_splitnode_float_int32_t(Node_float_int32_t *root, float *pa, uint32_t *pidx, int8_t no_dims, float *point_coord, 
                      float min_dist, uint32_t k, float distance_upper_bound, float eps_fac, uint8_t *mask,
                      uint32_t *closest_idx, float *closest_dist)
{
    int8_t dim;
    float dist_left, dist_right;
    float new_offset;
    float box_diff;

    /* Skip if distance bound exeeded */
    if (min_dist > distance_upper_bound)
    {
        return;
    }

    dim = root->cut_dim;

    /* Handle leaf node */
    if (dim == -1)
    {
        if (mask)
        {
            search_leaf_float_int32_t_mask(pa, pidx, no_dims, root->start_idx, root->n, point_coord, k, mask, closest_idx, closest_dist);
        }
        else
        {
            search_leaf_float_int32_t(pa, pidx, no_dims, root->start_idx, root->n, point_coord, k, closest_idx, closest_dist);
        }
        return;
    }

    /* Get distance to cutting plane */
    new_offset = point_coord[dim] - root->cut_val;

    if (new_offset < 0)
    {
        /* Left of cutting plane */
        dist_left = min_dist;
        if (dist_left < closest_dist[k - 1] * eps_fac)
        {
            /* Search left subtree if minimum distance is below limit */
            search_splitnode_float_int32_t((Node_float_int32_t *)root->left_child, pa, pidx, no_dims, point_coord, dist_left, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }

        /* Right of cutting plane. Update minimum distance.
           See Algorithms for Fast Vector Quantization
           Sunil Arya and David M. Mount. */
        box_diff = root->cut_bounds_lv - point_coord[dim];
        if (box_diff < 0)
        {
		box_diff = 0;
        }
        dist_right = min_dist - box_diff * box_diff + new_offset * new_offset;
        if (dist_right < closest_dist[k - 1] * eps_fac)
        {
            /* Search right subtree if minimum distance is below limit*/
            search_splitnode_float_int32_t((Node_float_int32_t *)root->right_child, pa, pidx, no_dims, point_coord, dist_right, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }
    }
    else
    {
        /* Right of cutting plane */
        dist_right = min_dist;
        if (dist_right < closest_dist[k - 1] * eps_fac)
        {
            /* Search right subtree if minimum distance is below limit*/
            search_splitnode_float_int32_t((Node_float_int32_t *)root->right_child, pa, pidx, no_dims, point_coord, dist_right, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }

        /* Left of cutting plane. Update minimum distance.
           See Algorithms for Fast Vector Quantization
           Sunil Arya and David M. Mount. */
        box_diff = point_coord[dim] - root->cut_bounds_hv;
        if (box_diff < 0)
        {
        	box_diff = 0;
        }
        dist_left = min_dist - box_diff * box_diff + new_offset * new_offset;
	  if (dist_left < closest_dist[k - 1] * eps_fac)
        {
            /* Search left subtree if minimum distance is below limit*/
            search_splitnode_float_int32_t((Node_float_int32_t *)root->left_child, pa, pidx, no_dims, point_coord, dist_left, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }
    }
}

/************************************************
Search for nearest neighbour for a set of query points
Params:
    tree : Tree struct of kd tree
    pa : data points
    pidx : permutation index of data points
    point_coords : query points
    num_points : number of query points
    mask : boolean array of invalid (True) and valid (False) data points
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_tree_float_int32_t(Tree_float_int32_t *tree, float *pa, float *point_coords,
                 uint32_t num_points, uint32_t k, float distance_upper_bound,
                 float eps, uint8_t *mask, uint32_t *closest_idxs, float *closest_dists)
{
    float min_dist;
    float eps_fac = 1 / ((1 + eps) * (1 + eps));
    int8_t no_dims = tree->no_dims;
    float *bbox = tree->bbox;
    uint32_t *pidx = tree->pidx;
    /* use 64-bit ints for indexing to avoid overflow, use signed ints to support all Openmp implementations */
    int64_t i = 0;
    int64_t j = 0;
    int64_t local_num_points = (int64_t) num_points;
    Node_float_int32_t *root = (Node_float_int32_t *)tree->root;

    /* Queries are OpenMP enabled */
    #pragma omp parallel
    {
        /* The low chunk size is important to avoid L2 cache trashing
           for spatial coherent query datasets
        */
        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            for (j = 0; j < k; j++)
            {
                closest_idxs[i * k + j] = IDX_MAX_int32_t;
                closest_dists[i * k + j] = DIST_MAX_float;
            }
            min_dist = get_min_dist_float(point_coords + no_dims * i, no_dims, bbox);
            search_splitnode_float_int32_t(root, pa, pidx, no_dims, point_coords + no_dims * i, min_dist,
                             k, distance_upper_bound, eps_fac, mask, &closest_idxs[i * k], &closest_dists[i * k]);
        }
    }
}

/************************************************
Compute mean k-NN Euclidean distances for Statistical Outlier Removal.
Self-queries data points, skips closest neighbor (self), computes
mean Euclidean distance to k remaining neighbors.
Params:
    tree : Tree struct of kd tree
    pa : data points (also used as query points)
    num_points : number of data points
    k : number of neighbors (excluding self)
    mean_dists_out : mean distances output, shape (num_points,)
************************************************/
void sor_mean_dists_float_int32_t(Tree_float_int32_t *tree, float *pa,
                 uint32_t num_points, uint32_t k, float *mean_dists_out)
{
    int8_t no_dims = tree->no_dims;
    float *bbox = tree->bbox;
    uint32_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    uint32_t k_total = k + 1;  /* +1 to include self */
    int64_t local_k_total = (int64_t) k_total;
    Node_float_int32_t *root = (Node_float_int32_t *)tree->root;

    #pragma omp parallel
    {
        uint32_t *local_idx = (uint32_t *)malloc(k_total * sizeof(uint32_t));
        float *local_dist = (float *)malloc(k_total * sizeof(float));

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            float min_dist, sum;
            int64_t count;

            /* Initialize k-NN arrays */
            for (j = 0; j < local_k_total; j++)
            {
                local_idx[j] = IDX_MAX_int32_t;
                local_dist[j] = DIST_MAX_float;
            }

            /* Query k+1 neighbors (includes self) */
            min_dist = get_min_dist_float(pa + no_dims * i, no_dims, bbox);
            search_splitnode_float_int32_t(root, pa, pidx, no_dims,
                             pa + no_dims * i, min_dist,
                             k_total, DIST_MAX_float, (float)1.0, NULL,
                             local_idx, local_dist);

            /* Mean Euclidean distance, skipping closest (self, index 0) */
            sum = 0;
            count = 0;
            for (j = 1; j < local_k_total; j++)
            {
                if (local_dist[j] < DIST_MAX_float)
                {
                    sum += sqrt(local_dist[j]);
                    count++;
                }
            }
            mean_dists_out[i] = (count > 0) ? sum / (float)count : 0;
        }

        free(local_idx);
        free(local_dist);
    }
}

/************************************************
Compute comprehensive point descriptors in a single k-NN pass.
Outputs NUM_DESCRIPTORS (20) features per point per scale:
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
  14:  z_range (z_max - z_min of neighbors)
  15:  z_above (max_z_neighbor - query_z)
  16:  z_below (query_z - min_z_neighbor)
  17:  z_std (height standard deviation of neighbors)
  18:  density (k / bounding_box_volume)
  19:  roughness (|point-to-plane distance|)
k_scales must be sorted ascending.
************************************************/
void compute_descriptors_multiscale_float_int32_t(Tree_float_int32_t *tree, float *pa, float *point_coords,
                 uint32_t num_points, uint32_t k_max,
                 int32_t *k_scales, int32_t num_scales,
                 float distance_upper_bound, float eps, uint8_t *mask,
                 float *descriptors_out)
{
    float eps_fac = 1 / ((1 + eps) * (1 + eps));
    int8_t no_dims = tree->no_dims;
    float *bbox = tree->bbox;
    uint32_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    int64_t local_k_max = (int64_t) k_max;
    Node_float_int32_t *root = (Node_float_int32_t *)tree->root;

    #pragma omp parallel
    {
        uint32_t *local_idx = (uint32_t *)malloc(k_max * sizeof(uint32_t));
        float *local_dist = (float *)malloc(k_max * sizeof(float));

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            float min_dist;
            int32_t s, scale_idx;
            float sum_x, sum_y, sum_z;
            float sum_xx, sum_xy, sum_xz, sum_yy, sum_yz, sum_zz;
            float bb_min_x, bb_min_y, bb_min_z, bb_max_x, bb_max_y, bb_max_z;
            float *pt;
            float px, py, pz;
            float qx = point_coords[no_dims * i];
            float qy = point_coords[no_dims * i + 1];
            float qz = point_coords[no_dims * i + 2];
            int64_t out_base = (int64_t)i * num_scales * NUM_DESCRIPTORS;

            /* Initialize k-NN arrays */
            for (j = 0; j < local_k_max; j++)
            {
                local_idx[j] = IDX_MAX_int32_t;
                local_dist[j] = DIST_MAX_float;
            }

            /* Query k_max neighbors */
            min_dist = get_min_dist_float(point_coords + no_dims * i, no_dims, bbox);
            search_splitnode_float_int32_t(root, pa, pidx, no_dims,
                             point_coords + no_dims * i, min_dist,
                             k_max, distance_upper_bound, eps_fac, mask,
                             local_idx, local_dist);

            /* Initialize accumulators */
            sum_x = sum_y = sum_z = 0;
            sum_xx = sum_xy = sum_xz = sum_yy = sum_yz = sum_zz = 0;
            bb_min_x = bb_min_y = bb_min_z = DIST_MAX_float;
            bb_max_x = bb_max_y = bb_max_z = -DIST_MAX_float;
            scale_idx = 0;

            for (j = 0; j < local_k_max && scale_idx < num_scales; j++)
            {
                if (local_idx[j] == IDX_MAX_int32_t || local_dist[j] >= DIST_MAX_float)
                    break;

                pt = pa + no_dims * local_idx[j];
                px = pt[0]; py = pt[1]; pz = pt[2];

                /* Accumulate sums for covariance */
                sum_x += px; sum_y += py; sum_z += pz;
                sum_xx += px * px; sum_xy += px * py; sum_xz += px * pz;
                sum_yy += py * py; sum_yz += py * pz; sum_zz += pz * pz;

                /* Update bounding box */
                if (px < bb_min_x) bb_min_x = px;
                if (px > bb_max_x) bb_max_x = px;
                if (py < bb_min_y) bb_min_y = py;
                if (py > bb_max_y) bb_max_y = py;
                if (pz < bb_min_z) bb_min_z = pz;
                if (pz > bb_max_z) bb_max_z = pz;

                /* At each scale boundary, compute all descriptors */
                if ((int32_t)(j + 1) == k_scales[scale_idx])
                {
                    int64_t out_off = out_base + (int64_t)scale_idx * NUM_DESCRIPTORS;
                    float evals[3], normal[3];

                    if (j + 1 < 2)
                    {
                        for (s = 0; s < NUM_DESCRIPTORS; s++)
                            descriptors_out[out_off + s] = 0;
                        descriptors_out[out_off + 5] = 1;  /* nz = 1 */
                    }
                    else
                    {
                        float inv_k = 1 / (float)(j + 1);
                        float mx = sum_x * inv_k;
                        float my = sum_y * inv_k;
                        float mz = sum_z * inv_k;

                        float cov_xx = sum_xx * inv_k - mx * mx;
                        float cov_xy = sum_xy * inv_k - mx * my;
                        float cov_xz = sum_xz * inv_k - mx * mz;
                        float cov_yy = sum_yy * inv_k - my * my;
                        float cov_yz = sum_yz * inv_k - my * mz;
                        float cov_zz = sum_zz * inv_k - mz * mz;

                        eigen_symmetric_3x3_float(cov_xx, cov_xy, cov_xz,
                                                     cov_yy, cov_yz, cov_zz,
                                                     evals, normal);

                        /* 0-2: eigenvalues */
                        descriptors_out[out_off + 0] = evals[0];
                        descriptors_out[out_off + 1] = evals[1];
                        descriptors_out[out_off + 2] = evals[2];

                        /* 3-5: normal */
                        descriptors_out[out_off + 3] = normal[0];
                        descriptors_out[out_off + 4] = normal[1];
                        descriptors_out[out_off + 5] = normal[2];

                        /* 6: verticality = 1 - |nz| */
                        descriptors_out[out_off + 6] = 1 - (normal[2] >= 0 ? normal[2] : -normal[2]);

                        /* 7-13: derived eigenvalue features */
                        {
                            float sum_eig = evals[0] + evals[1] + evals[2];
                            float inv_l1 = evals[0] > (float)1e-30 ? 1 / evals[0] : 0;
                            float inv_sum = sum_eig > (float)1e-30 ? 1 / sum_eig : 0;

                            /* 7: linearity = (l1 - l2) / l1 */
                            descriptors_out[out_off + 7] = (evals[0] - evals[1]) * inv_l1;

                            /* 8: planarity = (l2 - l3) / l1 */
                            descriptors_out[out_off + 8] = (evals[1] - evals[2]) * inv_l1;

                            /* 9: sphericity = l3 / l1 */
                            descriptors_out[out_off + 9] = evals[2] * inv_l1;

                            /* 10: omnivariance = (l1 * l2 * l3)^(1/3) */
                            {
                                float prod = evals[0] * evals[1] * evals[2];
                                descriptors_out[out_off + 10] = prod > 0 ? cbrt(prod) : 0;
                            }

                            /* 11: anisotropy = (l1 - l3) / l1 */
                            descriptors_out[out_off + 11] = (evals[0] - evals[2]) * inv_l1;

                            /* 12: eigenentropy = -sum(li/S * ln(li/S)) */
                            {
                                float entropy = 0;
                                if (sum_eig > (float)1e-30)
                                {
                                    int32_t ei;
                                    for (ei = 0; ei < 3; ei++)
                                    {
                                        float p = evals[ei] * inv_sum;
                                        if (p > (float)1e-30)
                                            entropy -= p * log(p);
                                    }
                                }
                                descriptors_out[out_off + 12] = entropy;
                            }

                            /* 13: surface_variation = l3 / (l1 + l2 + l3) */
                            descriptors_out[out_off + 13] = evals[2] * inv_sum;
                        }

                        /* 14: z_range */
                        descriptors_out[out_off + 14] = bb_max_z - bb_min_z;

                        /* 15: z_above = max_z - query_z */
                        descriptors_out[out_off + 15] = bb_max_z - qz;

                        /* 16: z_below = query_z - min_z */
                        descriptors_out[out_off + 16] = qz - bb_min_z;

                        /* 17: z_std = sqrt(cov_zz) */
                        descriptors_out[out_off + 17] = cov_zz > 0 ? sqrt(cov_zz) : 0;

                        /* 18: density = k / bbox_volume */
                        {
                            float vol = (bb_max_x - bb_min_x) * (bb_max_y - bb_min_y) * (bb_max_z - bb_min_z);
                            descriptors_out[out_off + 18] = vol > (float)1e-30 ? (float)(j + 1) / vol : 0;
                        }

                        /* 19: roughness = |dot(query - centroid, normal)| */
                        {
                            float dot = (qx - mx) * normal[0] + (qy - my) * normal[1] + (qz - mz) * normal[2];
                            descriptors_out[out_off + 19] = dot >= 0 ? dot : -dot;
                        }
                    }
                    scale_idx++;
                }
            }

            /* Fill remaining scales with zeros */
            for (s = scale_idx; s < num_scales; s++)
            {
                int64_t out_off = out_base + (int64_t)s * NUM_DESCRIPTORS;
                int32_t f;
                for (f = 0; f < NUM_DESCRIPTORS; f++)
                    descriptors_out[out_off + f] = 0;
                descriptors_out[out_off + 5] = 1;  /* nz = 1 */
            }
        }

        free(local_idx);
        free(local_dist);
    }
}

/************************************************
Radius-based outlier filter.
Queries k_min neighbors for each point. If the k_min-th
neighbor is beyond radius, the point is marked as outlier.
Params:
    tree : Tree struct
    pa : data points (also query points)
    num_points : number of points
    k_min : minimum number of neighbors required within radius
    radius : search radius (Euclidean)
    inlier_mask_out : (num_points,) output, 1 = inlier
    count_out : number of inliers (return, may be NULL)
************************************************/
void radius_filter_float_int32_t(Tree_float_int32_t *tree, float *pa,
                 uint32_t num_points, uint32_t k_min, float radius,
                 uint8_t *inlier_mask_out, uint32_t *count_out)
{
    int8_t no_dims = tree->no_dims;
    float *bbox = tree->bbox;
    uint32_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    /* Query k_min+1 to include self */
    uint32_t k_total = k_min + 1;
    int64_t local_k_total = (int64_t) k_total;
    float radius_sq = radius * radius;
    Node_float_int32_t *root = (Node_float_int32_t *)tree->root;
    uint32_t total_inliers = 0;

    #pragma omp parallel
    {
        uint32_t *local_idx = (uint32_t *)malloc(k_total * sizeof(uint32_t));
        float *local_dist = (float *)malloc(k_total * sizeof(float));
        uint32_t thread_inliers = 0;

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            float min_dist;
            uint32_t count;

            for (j = 0; j < local_k_total; j++)
            {
                local_idx[j] = IDX_MAX_int32_t;
                local_dist[j] = DIST_MAX_float;
            }

            min_dist = get_min_dist_float(pa + no_dims * i, no_dims, bbox);
            search_splitnode_float_int32_t(root, pa, pidx, no_dims,
                             pa + no_dims * i, min_dist,
                             k_total, radius_sq, (float)1.0, NULL,
                             local_idx, local_dist);

            /* Count valid neighbors (excluding self at index 0) */
            count = 0;
            for (j = 1; j < local_k_total; j++)
            {
                if (local_dist[j] < radius_sq && local_idx[j] != IDX_MAX_int32_t)
                    count++;
            }

            if (count >= k_min)
            {
                inlier_mask_out[i] = 1;
                thread_inliers++;
            }
            else
            {
                inlier_mask_out[i] = 0;
            }
        }

        #pragma omp atomic
        total_inliers += thread_inliers;

        free(local_idx);
        free(local_dist);
    }

    if (count_out) *count_out = total_inliers;
}

/************************************************
Estimate normals and curvatures from k-NN neighborhoods.
Lightweight alternative to compute_descriptors when only
normals and curvature are needed.
Params:
    tree : Tree struct
    pa : data points
    point_coords : query points (n * 3)
    num_points : number of query points
    k : number of neighbors
    distance_upper_bound : max distance
    eps : approximation factor
    mask : point validity mask (may be NULL)
    normals_out : (num_points * 3) output normals
    curvatures_out : (num_points,) output curvatures (surface variation)
************************************************/
void estimate_normals_float_int32_t(Tree_float_int32_t *tree, float *pa,
                 float *point_coords, uint32_t num_points, uint32_t k,
                 float distance_upper_bound, float eps, uint8_t *mask,
                 float *normals_out, float *curvatures_out)
{
    float eps_fac = 1 / ((1 + eps) * (1 + eps));
    int8_t no_dims = tree->no_dims;
    float *bbox = tree->bbox;
    uint32_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    int64_t local_k = (int64_t) k;
    Node_float_int32_t *root = (Node_float_int32_t *)tree->root;

    #pragma omp parallel
    {
        uint32_t *local_idx = (uint32_t *)malloc(k * sizeof(uint32_t));
        float *local_dist = (float *)malloc(k * sizeof(float));

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            float min_dist;
            float sum_x, sum_y, sum_z;
            float sum_xx, sum_xy, sum_xz, sum_yy, sum_yz, sum_zz;
            float *pt;
            float px, py, pz;
            int64_t valid_count;

            /* Initialize */
            for (j = 0; j < local_k; j++)
            {
                local_idx[j] = IDX_MAX_int32_t;
                local_dist[j] = DIST_MAX_float;
            }

            /* Query k neighbors */
            min_dist = get_min_dist_float(point_coords + no_dims * i, no_dims, bbox);
            search_splitnode_float_int32_t(root, pa, pidx, no_dims,
                             point_coords + no_dims * i, min_dist,
                             k, distance_upper_bound, eps_fac, mask,
                             local_idx, local_dist);

            /* Accumulate covariance sums */
            sum_x = sum_y = sum_z = 0;
            sum_xx = sum_xy = sum_xz = sum_yy = sum_yz = sum_zz = 0;
            valid_count = 0;

            for (j = 0; j < local_k; j++)
            {
                if (local_idx[j] == IDX_MAX_int32_t || local_dist[j] >= DIST_MAX_float)
                    break;
                pt = pa + no_dims * local_idx[j];
                px = pt[0]; py = pt[1]; pz = pt[2];
                sum_x += px; sum_y += py; sum_z += pz;
                sum_xx += px * px; sum_xy += px * py; sum_xz += px * pz;
                sum_yy += py * py; sum_yz += py * pz; sum_zz += pz * pz;
                valid_count++;
            }

            if (valid_count < 2)
            {
                normals_out[3 * i]     = 0;
                normals_out[3 * i + 1] = 0;
                normals_out[3 * i + 2] = 1;
                curvatures_out[i] = 0;
            }
            else
            {
                float inv_k = 1 / (float)valid_count;
                float mx = sum_x * inv_k, my = sum_y * inv_k, mz = sum_z * inv_k;
                float cov_xx = sum_xx * inv_k - mx * mx;
                float cov_xy = sum_xy * inv_k - mx * my;
                float cov_xz = sum_xz * inv_k - mx * mz;
                float cov_yy = sum_yy * inv_k - my * my;
                float cov_yz = sum_yz * inv_k - my * mz;
                float cov_zz = sum_zz * inv_k - mz * mz;

                float evals[3], normal[3];
                eigen_symmetric_3x3_float(cov_xx, cov_xy, cov_xz,
                                             cov_yy, cov_yz, cov_zz,
                                             evals, normal);

                /* Orient normal upward */
                if (normal[2] < 0) { normal[0] = -normal[0]; normal[1] = -normal[1]; normal[2] = -normal[2]; }

                normals_out[3 * i]     = normal[0];
                normals_out[3 * i + 1] = normal[1];
                normals_out[3 * i + 2] = normal[2];

                /* Curvature = surface variation = l3 / (l1 + l2 + l3) */
                {
                    float sum_eig = evals[0] + evals[1] + evals[2];
                    curvatures_out[i] = sum_eig > (float)1e-30 ? evals[2] / sum_eig : 0;
                }
            }
        }

        free(local_idx);
        free(local_dist);
    }
}

/************************************************
Insert point into priority queue
Params:
    closest_idx : index queue
    closest_dist : distance queue
    pidx : permutation index of data points
    cur_dist : distance to point inserted
    k : number of neighbours
************************************************/
void insert_point_float_int64_t(uint64_t *closest_idx, float *closest_dist, uint64_t pidx, float cur_dist, uint64_t k)
{
    int i;
    for (i = k - 1; i > 0; i--)
    {
        if (closest_dist[i - 1] > cur_dist)
        {
            closest_dist[i] = closest_dist[i - 1];
            closest_idx[i] = closest_idx[i - 1];
        }
        else
        {
            break;
        }
    }
    closest_idx[i] = pidx;
    closest_dist[i] = cur_dist;
}

/************************************************
Get the bounding box of a set of points
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims: number of dimensions
    n : number of points
    bbox : bounding box (return)
************************************************/
void get_bounding_box_float_int64_t(float *pa, uint64_t *pidx, int8_t no_dims, uint64_t n, float *bbox)
{
    float cur;
    int8_t i, j;
    uint64_t bbox_idx, i2;

    /* Use first data point to initialize */
    for (i = 0; i < no_dims; i++)
    {
        bbox[2 * i] = bbox[2 * i + 1] = PA(0, i);
    }

    /* Update using rest of data points */
    for (i2 = 1; i2 < n; i2++)
    {
        for (j = 0; j < no_dims; j++)
        {
            bbox_idx = 2 * j;
            cur = PA(i2, j);
            if (cur < bbox[bbox_idx])
            {
                bbox[bbox_idx] = cur;
            }
            else if (cur > bbox[bbox_idx + 1])
            {
                bbox[bbox_idx + 1] = cur;
            }
        }
    }
}

/************************************************
Partition a range of data points by manipulation the permutation index.
The sliding midpoint rule is used for the partitioning.
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims: number of dimensions
    start_idx : index of first data point to use
    n :  number of data points
    bbox : bounding box of data points
    cut_dim : dimension used for partition (return)
    cut_val : value of cutting point (return)
    n_lo : number of point below cutting plane (return)
************************************************/
int partition_float_int64_t(float *pa, uint64_t *pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, float *bbox, int8_t *cut_dim, float *cut_val, uint64_t *n_lo)
{
    int8_t dim = 0, i;
    uint64_t p, q, i2;
    float size = 0, min_val, max_val, split, side_len, cur_val;
    uint64_t end_idx = start_idx + n - 1;

    /* Find largest bounding box side */
    for (i = 0; i < no_dims; i++)
    {
        side_len = bbox[2 * i + 1] - bbox[2 * i];
        if (side_len > size)
        {
            dim = i;
            size = side_len;
        }
    }

    min_val = bbox[2 * dim];
    max_val = bbox[2 * dim + 1];

    /* Check for zero length or inconsistent */
    if (min_val >= max_val)
        return 1;

    /* Use middle for splitting */
    split = (min_val + max_val) / 2;

    /* Partition all data points around middle */
    p = start_idx;
    q = end_idx;
    while (p <= q)
    {
        if (PA(p, dim) < split)
        {
            p++;
        }
        else if (PA(q, dim) >= split)
        {
            /* Guard for underflow */
            if (q > 0)
            {
                q--;
            }
            else
            {
                break;
            }
        }
        else
        {
            PASWAP_int64_t(p, q);
            p++;
            q--;
        }
    }

    /* Check for empty splits */
    if (p == start_idx)
    {
        /* No points less than split.
           Split at lowest point instead.
           Minimum 1 point will be in lower box.
        */

        uint64_t j = start_idx;
        split = PA(j, dim);
        for (i2 = start_idx + 1; i2 <= end_idx; i2++)
        {
            /* Find lowest point */
            cur_val = PA(i2, dim);
            if (cur_val < split)
            {
                j = i2;
                split = cur_val;
            }
        }
        PASWAP_int64_t(j, start_idx);
        p = start_idx + 1;
    }
    else if (p == end_idx + 1)
    {
        /* No points greater than split.
           Split at highest point instead.
           Minimum 1 point will be in higher box.
        */

        uint64_t j = end_idx;
        split = PA(j, dim);
        for (i2 = start_idx; i2 < end_idx; i2++)
        {
            /* Find highest point */
            cur_val = PA(i2, dim);
            if (cur_val > split)
            {
                j = i2;
                split = cur_val;
            }
        }
        PASWAP_int64_t(j, end_idx);
        p = end_idx;
    }

    /* Set return values */
    *cut_dim = dim;
    *cut_val = split;
    *n_lo = p - start_idx;
    return 0;
}

/************************************************
Construct a sub tree over a range of data points.
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims: number of dimensions
    start_idx : index of first data point to use
    n :  number of data points
    bsp : number of points per leaf
    bbox : bounding box of set of data points
************************************************/
Node_float_int64_t* construct_subtree_float_int64_t(float *pa, uint64_t *pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, uint64_t bsp, float *bbox)
{
    /* Create new node */
    int is_leaf = (n <= bsp);
    Node_float_int64_t *root = create_node_float_int64_t(start_idx, n, is_leaf);
    int rval;
    int8_t cut_dim;
    uint64_t n_lo;
    float cut_val, lv, hv;
    if (is_leaf)
    {
        /* Make leaf node */
        root->cut_dim = -1;
    }
    else
    {
        /* Make split node */
        /* Partition data set and set node info */
        rval = partition_float_int64_t(pa, pidx, no_dims, start_idx, n, bbox, &cut_dim, &cut_val, &n_lo);
        if (rval == 1)
        {
            root->cut_dim = -1;
            return root;
        }
        root->cut_val = cut_val;
        root->cut_dim = cut_dim;

        /* Recurse on both subsets */
        lv = bbox[2 * cut_dim];
        hv = bbox[2 * cut_dim + 1];

        /* Set bounds for cut dimension */
        root->cut_bounds_lv = lv;
        root->cut_bounds_hv = hv;

        /* Update bounding box before call to lower subset and restore after */
        bbox[2 * cut_dim + 1] = cut_val;
        root->left_child = (struct Node_float_int64_t *)construct_subtree_float_int64_t(pa, pidx, no_dims, start_idx, n_lo, bsp, bbox);
        bbox[2 * cut_dim + 1] = hv;

        /* Update bounding box before call to higher subset and restore after */
        bbox[2 * cut_dim] = cut_val;
        root->right_child = (struct Node_float_int64_t *)construct_subtree_float_int64_t(pa, pidx, no_dims, start_idx + n_lo, n - n_lo, bsp, bbox);
        bbox[2 * cut_dim] = lv;
    }
    return root;
}

/************************************************
Construct a tree over data points.
Params:
    pa : data points
    no_dims: number of dimensions
    n :  number of data points
    bsp : number of points per leaf
************************************************/
Tree_float_int64_t* construct_tree_float_int64_t(float *pa, int8_t no_dims, uint64_t n, uint64_t bsp)
{
    Tree_float_int64_t *tree = (Tree_float_int64_t *)malloc(sizeof(Tree_float_int64_t));
    uint64_t i;
    uint64_t *pidx;
    float *bbox;

    tree->no_dims = no_dims;

    /* Initialize permutation array */
    pidx = (uint64_t *)malloc(sizeof(uint64_t) * n);
    for (i = 0; i < n; i++)
    {
        pidx[i] = i;
    }

    bbox = (float *)malloc(2 * sizeof(float) * no_dims);
    get_bounding_box_float_int64_t(pa, pidx, no_dims, n, bbox);
    tree->bbox = bbox;

    /* Construct subtree on full dataset */
    tree->root = (struct Node_float_int64_t *)construct_subtree_float_int64_t(pa, pidx, no_dims, 0, n, bsp, bbox);

    tree->pidx = pidx;
    return tree;
}

/************************************************
Create a tree node.
Params:
    start_idx : index of first data point to use
    n :  number of data points
************************************************/
Node_float_int64_t* create_node_float_int64_t(uint64_t start_idx, uint64_t n, int is_leaf)
{
    Node_float_int64_t *new_node;
    if (is_leaf)
    {
        /*
            Allocate only the part of the struct that will be used in a leaf node.
            This relies on the C99 specification of struct layout conservation and padding and
            that dereferencing is never attempted for the node pointers in a leaf.
        */
        new_node = (Node_float_int64_t *)malloc(sizeof(Node_float_int64_t) - 2 * sizeof(Node_float_int64_t *));
    }
    else
    {
        new_node = (Node_float_int64_t *)malloc(sizeof(Node_float_int64_t));
    }
    new_node->n = n;
    new_node->start_idx = start_idx;
    return new_node;
}

/************************************************
Delete subtree
Params:
    root : root node of subtree to delete
************************************************/
void delete_subtree_float_int64_t(Node_float_int64_t *root)
{
    if (root->cut_dim != -1)
    {
        delete_subtree_float_int64_t((Node_float_int64_t *)root->left_child);
        delete_subtree_float_int64_t((Node_float_int64_t *)root->right_child);
    }
    free(root);
}

/************************************************
Delete tree
Params:
    tree : Tree struct of kd tree
************************************************/
void delete_tree_float_int64_t(Tree_float_int64_t *tree)
{
    delete_subtree_float_int64_t((Node_float_int64_t *)tree->root);
    free(tree->bbox);
    free(tree->pidx);
    free(tree);
}

/************************************************
Print
************************************************/
void print_tree_float_int64_t(Node_float_int64_t *root, int level)
{
    int i;
    for (i = 0; i < level; i++)
    {
        printf(" ");
    }
    printf("(cut_val: %f, cut_dim: %i)\n", root->cut_val, root->cut_dim);
    if (root->cut_dim != -1)
        print_tree_float_int64_t((Node_float_int64_t *)root->left_child, level + 1);
    if (root->cut_dim != -1)
        print_tree_float_int64_t((Node_float_int64_t *)root->right_child, level + 1);
}

/************************************************
Search a leaf node for closest point
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims : number of dimensions
    start_idx : index of first data point to use
    size :  number of data points
    point_coord : query point
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_leaf_float_int64_t(float *restrict pa, uint64_t *restrict pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, float *restrict point_coord,
                 uint64_t k, uint64_t *restrict closest_idx, float *restrict closest_dist)
{
    float cur_dist;
    uint64_t i;
    /* Loop through all points in leaf */
    for (i = 0; i < n; i++)
    {
        /* Get distance to query point */
        cur_dist = calc_dist_float(&PA(start_idx + i, 0), point_coord, no_dims);
        /* Update closest info if new point is closest so far*/
        if (cur_dist < closest_dist[k - 1])
        {
            insert_point_float_int64_t(closest_idx, closest_dist, pidx[start_idx + i], cur_dist, k);
        }
    }
}


/************************************************
Search a leaf node for closest point with data point mask
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims : number of dimensions
    start_idx : index of first data point to use
    size :  number of data points
    point_coord : query point
    mask : boolean array of invalid (True) and valid (False) data points
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_leaf_float_int64_t_mask(float *restrict pa, uint64_t *restrict pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, float *restrict point_coord,
                               uint64_t k, uint8_t *mask, uint64_t *restrict closest_idx, float *restrict closest_dist)
{
    float cur_dist;
    uint64_t i;
    /* Loop through all points in leaf */
    for (i = 0; i < n; i++)
    {
        /* Is this point masked out? */
        if (mask[pidx[start_idx + i]])
        {
            continue;
        }
        /* Get distance to query point */
        cur_dist = calc_dist_float(&PA(start_idx + i, 0), point_coord, no_dims);
        /* Update closest info if new point is closest so far*/
        if (cur_dist < closest_dist[k - 1])
        {
            insert_point_float_int64_t(closest_idx, closest_dist, pidx[start_idx + i], cur_dist, k);
        }
    }
}

/************************************************
Search subtree for nearest to query point
Params:
    root : root node of subtree
    pa : data points
    pidx : permutation index of data points
    no_dims : number of dimensions
    point_coord : query point
    min_dist : minumum distance to nearest neighbour
    mask : boolean array of invalid (True) and valid (False) data points
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_splitnode_float_int64_t(Node_float_int64_t *root, float *pa, uint64_t *pidx, int8_t no_dims, float *point_coord, 
                      float min_dist, uint64_t k, float distance_upper_bound, float eps_fac, uint8_t *mask,
                      uint64_t *closest_idx, float *closest_dist)
{
    int8_t dim;
    float dist_left, dist_right;
    float new_offset;
    float box_diff;

    /* Skip if distance bound exeeded */
    if (min_dist > distance_upper_bound)
    {
        return;
    }

    dim = root->cut_dim;

    /* Handle leaf node */
    if (dim == -1)
    {
        if (mask)
        {
            search_leaf_float_int64_t_mask(pa, pidx, no_dims, root->start_idx, root->n, point_coord, k, mask, closest_idx, closest_dist);
        }
        else
        {
            search_leaf_float_int64_t(pa, pidx, no_dims, root->start_idx, root->n, point_coord, k, closest_idx, closest_dist);
        }
        return;
    }

    /* Get distance to cutting plane */
    new_offset = point_coord[dim] - root->cut_val;

    if (new_offset < 0)
    {
        /* Left of cutting plane */
        dist_left = min_dist;
        if (dist_left < closest_dist[k - 1] * eps_fac)
        {
            /* Search left subtree if minimum distance is below limit */
            search_splitnode_float_int64_t((Node_float_int64_t *)root->left_child, pa, pidx, no_dims, point_coord, dist_left, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }

        /* Right of cutting plane. Update minimum distance.
           See Algorithms for Fast Vector Quantization
           Sunil Arya and David M. Mount. */
        box_diff = root->cut_bounds_lv - point_coord[dim];
        if (box_diff < 0)
        {
		box_diff = 0;
        }
        dist_right = min_dist - box_diff * box_diff + new_offset * new_offset;
        if (dist_right < closest_dist[k - 1] * eps_fac)
        {
            /* Search right subtree if minimum distance is below limit*/
            search_splitnode_float_int64_t((Node_float_int64_t *)root->right_child, pa, pidx, no_dims, point_coord, dist_right, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }
    }
    else
    {
        /* Right of cutting plane */
        dist_right = min_dist;
        if (dist_right < closest_dist[k - 1] * eps_fac)
        {
            /* Search right subtree if minimum distance is below limit*/
            search_splitnode_float_int64_t((Node_float_int64_t *)root->right_child, pa, pidx, no_dims, point_coord, dist_right, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }

        /* Left of cutting plane. Update minimum distance.
           See Algorithms for Fast Vector Quantization
           Sunil Arya and David M. Mount. */
        box_diff = point_coord[dim] - root->cut_bounds_hv;
        if (box_diff < 0)
        {
        	box_diff = 0;
        }
        dist_left = min_dist - box_diff * box_diff + new_offset * new_offset;
	  if (dist_left < closest_dist[k - 1] * eps_fac)
        {
            /* Search left subtree if minimum distance is below limit*/
            search_splitnode_float_int64_t((Node_float_int64_t *)root->left_child, pa, pidx, no_dims, point_coord, dist_left, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }
    }
}

/************************************************
Search for nearest neighbour for a set of query points
Params:
    tree : Tree struct of kd tree
    pa : data points
    pidx : permutation index of data points
    point_coords : query points
    num_points : number of query points
    mask : boolean array of invalid (True) and valid (False) data points
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_tree_float_int64_t(Tree_float_int64_t *tree, float *pa, float *point_coords,
                 uint64_t num_points, uint64_t k, float distance_upper_bound,
                 float eps, uint8_t *mask, uint64_t *closest_idxs, float *closest_dists)
{
    float min_dist;
    float eps_fac = 1 / ((1 + eps) * (1 + eps));
    int8_t no_dims = tree->no_dims;
    float *bbox = tree->bbox;
    uint64_t *pidx = tree->pidx;
    /* use 64-bit ints for indexing to avoid overflow, use signed ints to support all Openmp implementations */
    int64_t i = 0;
    int64_t j = 0;
    int64_t local_num_points = (int64_t) num_points;
    Node_float_int64_t *root = (Node_float_int64_t *)tree->root;

    /* Queries are OpenMP enabled */
    #pragma omp parallel
    {
        /* The low chunk size is important to avoid L2 cache trashing
           for spatial coherent query datasets
        */
        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            for (j = 0; j < k; j++)
            {
                closest_idxs[i * k + j] = IDX_MAX_int64_t;
                closest_dists[i * k + j] = DIST_MAX_float;
            }
            min_dist = get_min_dist_float(point_coords + no_dims * i, no_dims, bbox);
            search_splitnode_float_int64_t(root, pa, pidx, no_dims, point_coords + no_dims * i, min_dist,
                             k, distance_upper_bound, eps_fac, mask, &closest_idxs[i * k], &closest_dists[i * k]);
        }
    }
}

/************************************************
Compute mean k-NN Euclidean distances for Statistical Outlier Removal.
Self-queries data points, skips closest neighbor (self), computes
mean Euclidean distance to k remaining neighbors.
Params:
    tree : Tree struct of kd tree
    pa : data points (also used as query points)
    num_points : number of data points
    k : number of neighbors (excluding self)
    mean_dists_out : mean distances output, shape (num_points,)
************************************************/
void sor_mean_dists_float_int64_t(Tree_float_int64_t *tree, float *pa,
                 uint64_t num_points, uint64_t k, float *mean_dists_out)
{
    int8_t no_dims = tree->no_dims;
    float *bbox = tree->bbox;
    uint64_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    uint64_t k_total = k + 1;  /* +1 to include self */
    int64_t local_k_total = (int64_t) k_total;
    Node_float_int64_t *root = (Node_float_int64_t *)tree->root;

    #pragma omp parallel
    {
        uint64_t *local_idx = (uint64_t *)malloc(k_total * sizeof(uint64_t));
        float *local_dist = (float *)malloc(k_total * sizeof(float));

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            float min_dist, sum;
            int64_t count;

            /* Initialize k-NN arrays */
            for (j = 0; j < local_k_total; j++)
            {
                local_idx[j] = IDX_MAX_int64_t;
                local_dist[j] = DIST_MAX_float;
            }

            /* Query k+1 neighbors (includes self) */
            min_dist = get_min_dist_float(pa + no_dims * i, no_dims, bbox);
            search_splitnode_float_int64_t(root, pa, pidx, no_dims,
                             pa + no_dims * i, min_dist,
                             k_total, DIST_MAX_float, (float)1.0, NULL,
                             local_idx, local_dist);

            /* Mean Euclidean distance, skipping closest (self, index 0) */
            sum = 0;
            count = 0;
            for (j = 1; j < local_k_total; j++)
            {
                if (local_dist[j] < DIST_MAX_float)
                {
                    sum += sqrt(local_dist[j]);
                    count++;
                }
            }
            mean_dists_out[i] = (count > 0) ? sum / (float)count : 0;
        }

        free(local_idx);
        free(local_dist);
    }
}

/************************************************
Compute comprehensive point descriptors in a single k-NN pass.
Outputs NUM_DESCRIPTORS (20) features per point per scale:
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
  14:  z_range (z_max - z_min of neighbors)
  15:  z_above (max_z_neighbor - query_z)
  16:  z_below (query_z - min_z_neighbor)
  17:  z_std (height standard deviation of neighbors)
  18:  density (k / bounding_box_volume)
  19:  roughness (|point-to-plane distance|)
k_scales must be sorted ascending.
************************************************/
void compute_descriptors_multiscale_float_int64_t(Tree_float_int64_t *tree, float *pa, float *point_coords,
                 uint64_t num_points, uint64_t k_max,
                 int32_t *k_scales, int32_t num_scales,
                 float distance_upper_bound, float eps, uint8_t *mask,
                 float *descriptors_out)
{
    float eps_fac = 1 / ((1 + eps) * (1 + eps));
    int8_t no_dims = tree->no_dims;
    float *bbox = tree->bbox;
    uint64_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    int64_t local_k_max = (int64_t) k_max;
    Node_float_int64_t *root = (Node_float_int64_t *)tree->root;

    #pragma omp parallel
    {
        uint64_t *local_idx = (uint64_t *)malloc(k_max * sizeof(uint64_t));
        float *local_dist = (float *)malloc(k_max * sizeof(float));

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            float min_dist;
            int32_t s, scale_idx;
            float sum_x, sum_y, sum_z;
            float sum_xx, sum_xy, sum_xz, sum_yy, sum_yz, sum_zz;
            float bb_min_x, bb_min_y, bb_min_z, bb_max_x, bb_max_y, bb_max_z;
            float *pt;
            float px, py, pz;
            float qx = point_coords[no_dims * i];
            float qy = point_coords[no_dims * i + 1];
            float qz = point_coords[no_dims * i + 2];
            int64_t out_base = (int64_t)i * num_scales * NUM_DESCRIPTORS;

            /* Initialize k-NN arrays */
            for (j = 0; j < local_k_max; j++)
            {
                local_idx[j] = IDX_MAX_int64_t;
                local_dist[j] = DIST_MAX_float;
            }

            /* Query k_max neighbors */
            min_dist = get_min_dist_float(point_coords + no_dims * i, no_dims, bbox);
            search_splitnode_float_int64_t(root, pa, pidx, no_dims,
                             point_coords + no_dims * i, min_dist,
                             k_max, distance_upper_bound, eps_fac, mask,
                             local_idx, local_dist);

            /* Initialize accumulators */
            sum_x = sum_y = sum_z = 0;
            sum_xx = sum_xy = sum_xz = sum_yy = sum_yz = sum_zz = 0;
            bb_min_x = bb_min_y = bb_min_z = DIST_MAX_float;
            bb_max_x = bb_max_y = bb_max_z = -DIST_MAX_float;
            scale_idx = 0;

            for (j = 0; j < local_k_max && scale_idx < num_scales; j++)
            {
                if (local_idx[j] == IDX_MAX_int64_t || local_dist[j] >= DIST_MAX_float)
                    break;

                pt = pa + no_dims * local_idx[j];
                px = pt[0]; py = pt[1]; pz = pt[2];

                /* Accumulate sums for covariance */
                sum_x += px; sum_y += py; sum_z += pz;
                sum_xx += px * px; sum_xy += px * py; sum_xz += px * pz;
                sum_yy += py * py; sum_yz += py * pz; sum_zz += pz * pz;

                /* Update bounding box */
                if (px < bb_min_x) bb_min_x = px;
                if (px > bb_max_x) bb_max_x = px;
                if (py < bb_min_y) bb_min_y = py;
                if (py > bb_max_y) bb_max_y = py;
                if (pz < bb_min_z) bb_min_z = pz;
                if (pz > bb_max_z) bb_max_z = pz;

                /* At each scale boundary, compute all descriptors */
                if ((int32_t)(j + 1) == k_scales[scale_idx])
                {
                    int64_t out_off = out_base + (int64_t)scale_idx * NUM_DESCRIPTORS;
                    float evals[3], normal[3];

                    if (j + 1 < 2)
                    {
                        for (s = 0; s < NUM_DESCRIPTORS; s++)
                            descriptors_out[out_off + s] = 0;
                        descriptors_out[out_off + 5] = 1;  /* nz = 1 */
                    }
                    else
                    {
                        float inv_k = 1 / (float)(j + 1);
                        float mx = sum_x * inv_k;
                        float my = sum_y * inv_k;
                        float mz = sum_z * inv_k;

                        float cov_xx = sum_xx * inv_k - mx * mx;
                        float cov_xy = sum_xy * inv_k - mx * my;
                        float cov_xz = sum_xz * inv_k - mx * mz;
                        float cov_yy = sum_yy * inv_k - my * my;
                        float cov_yz = sum_yz * inv_k - my * mz;
                        float cov_zz = sum_zz * inv_k - mz * mz;

                        eigen_symmetric_3x3_float(cov_xx, cov_xy, cov_xz,
                                                     cov_yy, cov_yz, cov_zz,
                                                     evals, normal);

                        /* 0-2: eigenvalues */
                        descriptors_out[out_off + 0] = evals[0];
                        descriptors_out[out_off + 1] = evals[1];
                        descriptors_out[out_off + 2] = evals[2];

                        /* 3-5: normal */
                        descriptors_out[out_off + 3] = normal[0];
                        descriptors_out[out_off + 4] = normal[1];
                        descriptors_out[out_off + 5] = normal[2];

                        /* 6: verticality = 1 - |nz| */
                        descriptors_out[out_off + 6] = 1 - (normal[2] >= 0 ? normal[2] : -normal[2]);

                        /* 7-13: derived eigenvalue features */
                        {
                            float sum_eig = evals[0] + evals[1] + evals[2];
                            float inv_l1 = evals[0] > (float)1e-30 ? 1 / evals[0] : 0;
                            float inv_sum = sum_eig > (float)1e-30 ? 1 / sum_eig : 0;

                            /* 7: linearity = (l1 - l2) / l1 */
                            descriptors_out[out_off + 7] = (evals[0] - evals[1]) * inv_l1;

                            /* 8: planarity = (l2 - l3) / l1 */
                            descriptors_out[out_off + 8] = (evals[1] - evals[2]) * inv_l1;

                            /* 9: sphericity = l3 / l1 */
                            descriptors_out[out_off + 9] = evals[2] * inv_l1;

                            /* 10: omnivariance = (l1 * l2 * l3)^(1/3) */
                            {
                                float prod = evals[0] * evals[1] * evals[2];
                                descriptors_out[out_off + 10] = prod > 0 ? cbrt(prod) : 0;
                            }

                            /* 11: anisotropy = (l1 - l3) / l1 */
                            descriptors_out[out_off + 11] = (evals[0] - evals[2]) * inv_l1;

                            /* 12: eigenentropy = -sum(li/S * ln(li/S)) */
                            {
                                float entropy = 0;
                                if (sum_eig > (float)1e-30)
                                {
                                    int32_t ei;
                                    for (ei = 0; ei < 3; ei++)
                                    {
                                        float p = evals[ei] * inv_sum;
                                        if (p > (float)1e-30)
                                            entropy -= p * log(p);
                                    }
                                }
                                descriptors_out[out_off + 12] = entropy;
                            }

                            /* 13: surface_variation = l3 / (l1 + l2 + l3) */
                            descriptors_out[out_off + 13] = evals[2] * inv_sum;
                        }

                        /* 14: z_range */
                        descriptors_out[out_off + 14] = bb_max_z - bb_min_z;

                        /* 15: z_above = max_z - query_z */
                        descriptors_out[out_off + 15] = bb_max_z - qz;

                        /* 16: z_below = query_z - min_z */
                        descriptors_out[out_off + 16] = qz - bb_min_z;

                        /* 17: z_std = sqrt(cov_zz) */
                        descriptors_out[out_off + 17] = cov_zz > 0 ? sqrt(cov_zz) : 0;

                        /* 18: density = k / bbox_volume */
                        {
                            float vol = (bb_max_x - bb_min_x) * (bb_max_y - bb_min_y) * (bb_max_z - bb_min_z);
                            descriptors_out[out_off + 18] = vol > (float)1e-30 ? (float)(j + 1) / vol : 0;
                        }

                        /* 19: roughness = |dot(query - centroid, normal)| */
                        {
                            float dot = (qx - mx) * normal[0] + (qy - my) * normal[1] + (qz - mz) * normal[2];
                            descriptors_out[out_off + 19] = dot >= 0 ? dot : -dot;
                        }
                    }
                    scale_idx++;
                }
            }

            /* Fill remaining scales with zeros */
            for (s = scale_idx; s < num_scales; s++)
            {
                int64_t out_off = out_base + (int64_t)s * NUM_DESCRIPTORS;
                int32_t f;
                for (f = 0; f < NUM_DESCRIPTORS; f++)
                    descriptors_out[out_off + f] = 0;
                descriptors_out[out_off + 5] = 1;  /* nz = 1 */
            }
        }

        free(local_idx);
        free(local_dist);
    }
}

/************************************************
Radius-based outlier filter.
Queries k_min neighbors for each point. If the k_min-th
neighbor is beyond radius, the point is marked as outlier.
Params:
    tree : Tree struct
    pa : data points (also query points)
    num_points : number of points
    k_min : minimum number of neighbors required within radius
    radius : search radius (Euclidean)
    inlier_mask_out : (num_points,) output, 1 = inlier
    count_out : number of inliers (return, may be NULL)
************************************************/
void radius_filter_float_int64_t(Tree_float_int64_t *tree, float *pa,
                 uint64_t num_points, uint64_t k_min, float radius,
                 uint8_t *inlier_mask_out, uint64_t *count_out)
{
    int8_t no_dims = tree->no_dims;
    float *bbox = tree->bbox;
    uint64_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    /* Query k_min+1 to include self */
    uint64_t k_total = k_min + 1;
    int64_t local_k_total = (int64_t) k_total;
    float radius_sq = radius * radius;
    Node_float_int64_t *root = (Node_float_int64_t *)tree->root;
    uint64_t total_inliers = 0;

    #pragma omp parallel
    {
        uint64_t *local_idx = (uint64_t *)malloc(k_total * sizeof(uint64_t));
        float *local_dist = (float *)malloc(k_total * sizeof(float));
        uint64_t thread_inliers = 0;

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            float min_dist;
            uint64_t count;

            for (j = 0; j < local_k_total; j++)
            {
                local_idx[j] = IDX_MAX_int64_t;
                local_dist[j] = DIST_MAX_float;
            }

            min_dist = get_min_dist_float(pa + no_dims * i, no_dims, bbox);
            search_splitnode_float_int64_t(root, pa, pidx, no_dims,
                             pa + no_dims * i, min_dist,
                             k_total, radius_sq, (float)1.0, NULL,
                             local_idx, local_dist);

            /* Count valid neighbors (excluding self at index 0) */
            count = 0;
            for (j = 1; j < local_k_total; j++)
            {
                if (local_dist[j] < radius_sq && local_idx[j] != IDX_MAX_int64_t)
                    count++;
            }

            if (count >= k_min)
            {
                inlier_mask_out[i] = 1;
                thread_inliers++;
            }
            else
            {
                inlier_mask_out[i] = 0;
            }
        }

        #pragma omp atomic
        total_inliers += thread_inliers;

        free(local_idx);
        free(local_dist);
    }

    if (count_out) *count_out = total_inliers;
}

/************************************************
Estimate normals and curvatures from k-NN neighborhoods.
Lightweight alternative to compute_descriptors when only
normals and curvature are needed.
Params:
    tree : Tree struct
    pa : data points
    point_coords : query points (n * 3)
    num_points : number of query points
    k : number of neighbors
    distance_upper_bound : max distance
    eps : approximation factor
    mask : point validity mask (may be NULL)
    normals_out : (num_points * 3) output normals
    curvatures_out : (num_points,) output curvatures (surface variation)
************************************************/
void estimate_normals_float_int64_t(Tree_float_int64_t *tree, float *pa,
                 float *point_coords, uint64_t num_points, uint64_t k,
                 float distance_upper_bound, float eps, uint8_t *mask,
                 float *normals_out, float *curvatures_out)
{
    float eps_fac = 1 / ((1 + eps) * (1 + eps));
    int8_t no_dims = tree->no_dims;
    float *bbox = tree->bbox;
    uint64_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    int64_t local_k = (int64_t) k;
    Node_float_int64_t *root = (Node_float_int64_t *)tree->root;

    #pragma omp parallel
    {
        uint64_t *local_idx = (uint64_t *)malloc(k * sizeof(uint64_t));
        float *local_dist = (float *)malloc(k * sizeof(float));

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            float min_dist;
            float sum_x, sum_y, sum_z;
            float sum_xx, sum_xy, sum_xz, sum_yy, sum_yz, sum_zz;
            float *pt;
            float px, py, pz;
            int64_t valid_count;

            /* Initialize */
            for (j = 0; j < local_k; j++)
            {
                local_idx[j] = IDX_MAX_int64_t;
                local_dist[j] = DIST_MAX_float;
            }

            /* Query k neighbors */
            min_dist = get_min_dist_float(point_coords + no_dims * i, no_dims, bbox);
            search_splitnode_float_int64_t(root, pa, pidx, no_dims,
                             point_coords + no_dims * i, min_dist,
                             k, distance_upper_bound, eps_fac, mask,
                             local_idx, local_dist);

            /* Accumulate covariance sums */
            sum_x = sum_y = sum_z = 0;
            sum_xx = sum_xy = sum_xz = sum_yy = sum_yz = sum_zz = 0;
            valid_count = 0;

            for (j = 0; j < local_k; j++)
            {
                if (local_idx[j] == IDX_MAX_int64_t || local_dist[j] >= DIST_MAX_float)
                    break;
                pt = pa + no_dims * local_idx[j];
                px = pt[0]; py = pt[1]; pz = pt[2];
                sum_x += px; sum_y += py; sum_z += pz;
                sum_xx += px * px; sum_xy += px * py; sum_xz += px * pz;
                sum_yy += py * py; sum_yz += py * pz; sum_zz += pz * pz;
                valid_count++;
            }

            if (valid_count < 2)
            {
                normals_out[3 * i]     = 0;
                normals_out[3 * i + 1] = 0;
                normals_out[3 * i + 2] = 1;
                curvatures_out[i] = 0;
            }
            else
            {
                float inv_k = 1 / (float)valid_count;
                float mx = sum_x * inv_k, my = sum_y * inv_k, mz = sum_z * inv_k;
                float cov_xx = sum_xx * inv_k - mx * mx;
                float cov_xy = sum_xy * inv_k - mx * my;
                float cov_xz = sum_xz * inv_k - mx * mz;
                float cov_yy = sum_yy * inv_k - my * my;
                float cov_yz = sum_yz * inv_k - my * mz;
                float cov_zz = sum_zz * inv_k - mz * mz;

                float evals[3], normal[3];
                eigen_symmetric_3x3_float(cov_xx, cov_xy, cov_xz,
                                             cov_yy, cov_yz, cov_zz,
                                             evals, normal);

                /* Orient normal upward */
                if (normal[2] < 0) { normal[0] = -normal[0]; normal[1] = -normal[1]; normal[2] = -normal[2]; }

                normals_out[3 * i]     = normal[0];
                normals_out[3 * i + 1] = normal[1];
                normals_out[3 * i + 2] = normal[2];

                /* Curvature = surface variation = l3 / (l1 + l2 + l3) */
                {
                    float sum_eig = evals[0] + evals[1] + evals[2];
                    curvatures_out[i] = sum_eig > (float)1e-30 ? evals[2] / sum_eig : 0;
                }
            }
        }

        free(local_idx);
        free(local_dist);
    }
}

/************************************************
Calculate squared cartesian distance between points
Params:
    point1_coord : point 1
    point2_coord : point 2
************************************************/
double calc_dist_double(double *point1_coord, double *point2_coord, int8_t no_dims)
{
    /* Calculate squared distance */
    double dist = 0, dim_dist;
    int8_t i;
    for (i = 0; i < no_dims; i++)
    {
        dim_dist = point2_coord[i] - point1_coord[i];
        dist += dim_dist * dim_dist;
    }
    return dist;
}

/************************************************
Get squared distance from point to cube in specified dimension
Params:
    dim : dimension
    point_coord : cartesian coordinates of point
    bbox : cube
************************************************/
double get_cube_offset_double(int8_t dim, double *point_coord, double *bbox)
{
    double dim_coord = point_coord[dim];

    if (dim_coord < bbox[2 * dim])
    {
        /* Left of cube in dimension */
        return dim_coord - bbox[2 * dim];
    }
    else if (dim_coord > bbox[2 * dim + 1])
    {
        /* Right of cube in dimension */
        return dim_coord - bbox[2 * dim + 1];
    }
    else
    {
        /* Inside cube in dimension */
        return 0.;
    }
}

/************************************************
Get minimum squared distance between point and cube.
Params:
    point_coord : cartesian coordinates of point
    no_dims : number of dimensions
    bbox : cube
************************************************/
double get_min_dist_double(double *point_coord, int8_t no_dims, double *bbox)
{
    double cube_offset = 0, cube_offset_dim;
    int8_t i;

    for (i = 0; i < no_dims; i++)
    {
        cube_offset_dim = get_cube_offset_double(i, point_coord, bbox);
        cube_offset += cube_offset_dim * cube_offset_dim;
    }

    return cube_offset;
}

/************************************************
Eigendecomposition of 3x3 symmetric matrix using Cardano's formula.
Eigenvalues returned sorted: evals[0] >= evals[1] >= evals[2].
Normal is the eigenvector of the smallest eigenvalue.
Params:
    cov_xx..cov_zz : upper triangle of symmetric matrix
    evals : eigenvalues output (3 values)
    normal : eigenvector of smallest eigenvalue (3 values)
************************************************/
void eigen_symmetric_3x3_double(double cov_xx, double cov_xy, double cov_xz,
                                   double cov_yy, double cov_yz, double cov_zz,
                                   double *evals, double *normal)
{
    double e1, e2, e3;
    double p1 = cov_xy * cov_xy + cov_xz * cov_xz + cov_yz * cov_yz;
    double q = (cov_xx + cov_yy + cov_zz) / 3;
    double p2 = (cov_xx - q) * (cov_xx - q) + (cov_yy - q) * (cov_yy - q) +
                  (cov_zz - q) * (cov_zz - q) + 2 * p1;
    double p = sqrt(p2 / 6);

    if (p < (double)1e-30)
    {
        /* All eigenvalues are equal */
        e1 = e2 = e3 = q;
    }
    else
    {
        double inv_p = 1 / p;
        /* B = (1/p)(M - q*I) */
        double b00 = inv_p * (cov_xx - q);
        double b01 = inv_p * cov_xy;
        double b02 = inv_p * cov_xz;
        double b11 = inv_p * (cov_yy - q);
        double b12 = inv_p * cov_yz;
        double b22 = inv_p * (cov_zz - q);

        /* det(B) / 2 */
        double r = (b00 * (b11 * b22 - b12 * b12)
                     - b01 * (b01 * b22 - b12 * b02)
                     + b02 * (b01 * b12 - b11 * b02)) / 2;

        /* Clamp for numerical stability */
        if (r <= -1) r = -1;
        else if (r >= 1) r = 1;

        double phi = acos(r) / 3;

        e1 = q + 2 * p * cos(phi);
        e3 = q + 2 * p * cos(phi + 2 * M_PI / 3);
        e2 = 3 * q - e1 - e3;
    }

    /* Clamp negative eigenvalues to zero */
    if (e1 < 0) e1 = 0;
    if (e2 < 0) e2 = 0;
    if (e3 < 0) e3 = 0;

    evals[0] = e1;
    evals[1] = e2;
    evals[2] = e3;

    /* Compute normal: eigenvector of smallest eigenvalue e3 */
    /* Form rows of (M - e3*I) */
    double r0x = cov_xx - e3, r0y = cov_xy,       r0z = cov_xz;
    double r1x = cov_xy,       r1y = cov_yy - e3, r1z = cov_yz;
    double r2x = cov_xz,       r2y = cov_yz,       r2z = cov_zz - e3;

    /* Cross products of all row pairs, pick largest */
    double nx, ny, nz, len_sq, best_len_sq;
    double cx, cy, cz;

    /* r0 x r1 */
    nx = r0y * r1z - r0z * r1y;
    ny = r0z * r1x - r0x * r1z;
    nz = r0x * r1y - r0y * r1x;
    best_len_sq = nx * nx + ny * ny + nz * nz;

    /* r0 x r2 */
    cx = r0y * r2z - r0z * r2y;
    cy = r0z * r2x - r0x * r2z;
    cz = r0x * r2y - r0y * r2x;
    len_sq = cx * cx + cy * cy + cz * cz;
    if (len_sq > best_len_sq) { nx = cx; ny = cy; nz = cz; best_len_sq = len_sq; }

    /* r1 x r2 */
    cx = r1y * r2z - r1z * r2y;
    cy = r1z * r2x - r1x * r2z;
    cz = r1x * r2y - r1y * r2x;
    len_sq = cx * cx + cy * cy + cz * cz;
    if (len_sq > best_len_sq) { nx = cx; ny = cy; nz = cz; best_len_sq = len_sq; }

    /* Normalize */
    if (best_len_sq > (double)1e-30)
    {
        double inv_len = 1 / sqrt(best_len_sq);
        normal[0] = nx * inv_len;
        normal[1] = ny * inv_len;
        normal[2] = nz * inv_len;
    }
    else
    {
        /* Degenerate case */
        normal[0] = 0;
        normal[1] = 0;
        normal[2] = 1;
    }
}


/************************************************
Insert point into priority queue
Params:
    closest_idx : index queue
    closest_dist : distance queue
    pidx : permutation index of data points
    cur_dist : distance to point inserted
    k : number of neighbours
************************************************/
void insert_point_double_int32_t(uint32_t *closest_idx, double *closest_dist, uint32_t pidx, double cur_dist, uint32_t k)
{
    int i;
    for (i = k - 1; i > 0; i--)
    {
        if (closest_dist[i - 1] > cur_dist)
        {
            closest_dist[i] = closest_dist[i - 1];
            closest_idx[i] = closest_idx[i - 1];
        }
        else
        {
            break;
        }
    }
    closest_idx[i] = pidx;
    closest_dist[i] = cur_dist;
}

/************************************************
Get the bounding box of a set of points
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims: number of dimensions
    n : number of points
    bbox : bounding box (return)
************************************************/
void get_bounding_box_double_int32_t(double *pa, uint32_t *pidx, int8_t no_dims, uint32_t n, double *bbox)
{
    double cur;
    int8_t i, j;
    uint32_t bbox_idx, i2;

    /* Use first data point to initialize */
    for (i = 0; i < no_dims; i++)
    {
        bbox[2 * i] = bbox[2 * i + 1] = PA(0, i);
    }

    /* Update using rest of data points */
    for (i2 = 1; i2 < n; i2++)
    {
        for (j = 0; j < no_dims; j++)
        {
            bbox_idx = 2 * j;
            cur = PA(i2, j);
            if (cur < bbox[bbox_idx])
            {
                bbox[bbox_idx] = cur;
            }
            else if (cur > bbox[bbox_idx + 1])
            {
                bbox[bbox_idx + 1] = cur;
            }
        }
    }
}

/************************************************
Partition a range of data points by manipulation the permutation index.
The sliding midpoint rule is used for the partitioning.
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims: number of dimensions
    start_idx : index of first data point to use
    n :  number of data points
    bbox : bounding box of data points
    cut_dim : dimension used for partition (return)
    cut_val : value of cutting point (return)
    n_lo : number of point below cutting plane (return)
************************************************/
int partition_double_int32_t(double *pa, uint32_t *pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, double *bbox, int8_t *cut_dim, double *cut_val, uint32_t *n_lo)
{
    int8_t dim = 0, i;
    uint32_t p, q, i2;
    double size = 0, min_val, max_val, split, side_len, cur_val;
    uint32_t end_idx = start_idx + n - 1;

    /* Find largest bounding box side */
    for (i = 0; i < no_dims; i++)
    {
        side_len = bbox[2 * i + 1] - bbox[2 * i];
        if (side_len > size)
        {
            dim = i;
            size = side_len;
        }
    }

    min_val = bbox[2 * dim];
    max_val = bbox[2 * dim + 1];

    /* Check for zero length or inconsistent */
    if (min_val >= max_val)
        return 1;

    /* Use middle for splitting */
    split = (min_val + max_val) / 2;

    /* Partition all data points around middle */
    p = start_idx;
    q = end_idx;
    while (p <= q)
    {
        if (PA(p, dim) < split)
        {
            p++;
        }
        else if (PA(q, dim) >= split)
        {
            /* Guard for underflow */
            if (q > 0)
            {
                q--;
            }
            else
            {
                break;
            }
        }
        else
        {
            PASWAP_int32_t(p, q);
            p++;
            q--;
        }
    }

    /* Check for empty splits */
    if (p == start_idx)
    {
        /* No points less than split.
           Split at lowest point instead.
           Minimum 1 point will be in lower box.
        */

        uint32_t j = start_idx;
        split = PA(j, dim);
        for (i2 = start_idx + 1; i2 <= end_idx; i2++)
        {
            /* Find lowest point */
            cur_val = PA(i2, dim);
            if (cur_val < split)
            {
                j = i2;
                split = cur_val;
            }
        }
        PASWAP_int32_t(j, start_idx);
        p = start_idx + 1;
    }
    else if (p == end_idx + 1)
    {
        /* No points greater than split.
           Split at highest point instead.
           Minimum 1 point will be in higher box.
        */

        uint32_t j = end_idx;
        split = PA(j, dim);
        for (i2 = start_idx; i2 < end_idx; i2++)
        {
            /* Find highest point */
            cur_val = PA(i2, dim);
            if (cur_val > split)
            {
                j = i2;
                split = cur_val;
            }
        }
        PASWAP_int32_t(j, end_idx);
        p = end_idx;
    }

    /* Set return values */
    *cut_dim = dim;
    *cut_val = split;
    *n_lo = p - start_idx;
    return 0;
}

/************************************************
Construct a sub tree over a range of data points.
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims: number of dimensions
    start_idx : index of first data point to use
    n :  number of data points
    bsp : number of points per leaf
    bbox : bounding box of set of data points
************************************************/
Node_double_int32_t* construct_subtree_double_int32_t(double *pa, uint32_t *pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, uint32_t bsp, double *bbox)
{
    /* Create new node */
    int is_leaf = (n <= bsp);
    Node_double_int32_t *root = create_node_double_int32_t(start_idx, n, is_leaf);
    int rval;
    int8_t cut_dim;
    uint32_t n_lo;
    double cut_val, lv, hv;
    if (is_leaf)
    {
        /* Make leaf node */
        root->cut_dim = -1;
    }
    else
    {
        /* Make split node */
        /* Partition data set and set node info */
        rval = partition_double_int32_t(pa, pidx, no_dims, start_idx, n, bbox, &cut_dim, &cut_val, &n_lo);
        if (rval == 1)
        {
            root->cut_dim = -1;
            return root;
        }
        root->cut_val = cut_val;
        root->cut_dim = cut_dim;

        /* Recurse on both subsets */
        lv = bbox[2 * cut_dim];
        hv = bbox[2 * cut_dim + 1];

        /* Set bounds for cut dimension */
        root->cut_bounds_lv = lv;
        root->cut_bounds_hv = hv;

        /* Update bounding box before call to lower subset and restore after */
        bbox[2 * cut_dim + 1] = cut_val;
        root->left_child = (struct Node_double_int32_t *)construct_subtree_double_int32_t(pa, pidx, no_dims, start_idx, n_lo, bsp, bbox);
        bbox[2 * cut_dim + 1] = hv;

        /* Update bounding box before call to higher subset and restore after */
        bbox[2 * cut_dim] = cut_val;
        root->right_child = (struct Node_double_int32_t *)construct_subtree_double_int32_t(pa, pidx, no_dims, start_idx + n_lo, n - n_lo, bsp, bbox);
        bbox[2 * cut_dim] = lv;
    }
    return root;
}

/************************************************
Construct a tree over data points.
Params:
    pa : data points
    no_dims: number of dimensions
    n :  number of data points
    bsp : number of points per leaf
************************************************/
Tree_double_int32_t* construct_tree_double_int32_t(double *pa, int8_t no_dims, uint32_t n, uint32_t bsp)
{
    Tree_double_int32_t *tree = (Tree_double_int32_t *)malloc(sizeof(Tree_double_int32_t));
    uint32_t i;
    uint32_t *pidx;
    double *bbox;

    tree->no_dims = no_dims;

    /* Initialize permutation array */
    pidx = (uint32_t *)malloc(sizeof(uint32_t) * n);
    for (i = 0; i < n; i++)
    {
        pidx[i] = i;
    }

    bbox = (double *)malloc(2 * sizeof(double) * no_dims);
    get_bounding_box_double_int32_t(pa, pidx, no_dims, n, bbox);
    tree->bbox = bbox;

    /* Construct subtree on full dataset */
    tree->root = (struct Node_double_int32_t *)construct_subtree_double_int32_t(pa, pidx, no_dims, 0, n, bsp, bbox);

    tree->pidx = pidx;
    return tree;
}

/************************************************
Create a tree node.
Params:
    start_idx : index of first data point to use
    n :  number of data points
************************************************/
Node_double_int32_t* create_node_double_int32_t(uint32_t start_idx, uint32_t n, int is_leaf)
{
    Node_double_int32_t *new_node;
    if (is_leaf)
    {
        /*
            Allocate only the part of the struct that will be used in a leaf node.
            This relies on the C99 specification of struct layout conservation and padding and
            that dereferencing is never attempted for the node pointers in a leaf.
        */
        new_node = (Node_double_int32_t *)malloc(sizeof(Node_double_int32_t) - 2 * sizeof(Node_double_int32_t *));
    }
    else
    {
        new_node = (Node_double_int32_t *)malloc(sizeof(Node_double_int32_t));
    }
    new_node->n = n;
    new_node->start_idx = start_idx;
    return new_node;
}

/************************************************
Delete subtree
Params:
    root : root node of subtree to delete
************************************************/
void delete_subtree_double_int32_t(Node_double_int32_t *root)
{
    if (root->cut_dim != -1)
    {
        delete_subtree_double_int32_t((Node_double_int32_t *)root->left_child);
        delete_subtree_double_int32_t((Node_double_int32_t *)root->right_child);
    }
    free(root);
}

/************************************************
Delete tree
Params:
    tree : Tree struct of kd tree
************************************************/
void delete_tree_double_int32_t(Tree_double_int32_t *tree)
{
    delete_subtree_double_int32_t((Node_double_int32_t *)tree->root);
    free(tree->bbox);
    free(tree->pidx);
    free(tree);
}

/************************************************
Print
************************************************/
void print_tree_double_int32_t(Node_double_int32_t *root, int level)
{
    int i;
    for (i = 0; i < level; i++)
    {
        printf(" ");
    }
    printf("(cut_val: %f, cut_dim: %i)\n", root->cut_val, root->cut_dim);
    if (root->cut_dim != -1)
        print_tree_double_int32_t((Node_double_int32_t *)root->left_child, level + 1);
    if (root->cut_dim != -1)
        print_tree_double_int32_t((Node_double_int32_t *)root->right_child, level + 1);
}

/************************************************
Search a leaf node for closest point
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims : number of dimensions
    start_idx : index of first data point to use
    size :  number of data points
    point_coord : query point
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_leaf_double_int32_t(double *restrict pa, uint32_t *restrict pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, double *restrict point_coord,
                 uint32_t k, uint32_t *restrict closest_idx, double *restrict closest_dist)
{
    double cur_dist;
    uint32_t i;
    /* Loop through all points in leaf */
    for (i = 0; i < n; i++)
    {
        /* Get distance to query point */
        cur_dist = calc_dist_double(&PA(start_idx + i, 0), point_coord, no_dims);
        /* Update closest info if new point is closest so far*/
        if (cur_dist < closest_dist[k - 1])
        {
            insert_point_double_int32_t(closest_idx, closest_dist, pidx[start_idx + i], cur_dist, k);
        }
    }
}


/************************************************
Search a leaf node for closest point with data point mask
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims : number of dimensions
    start_idx : index of first data point to use
    size :  number of data points
    point_coord : query point
    mask : boolean array of invalid (True) and valid (False) data points
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_leaf_double_int32_t_mask(double *restrict pa, uint32_t *restrict pidx, int8_t no_dims, uint32_t start_idx, uint32_t n, double *restrict point_coord,
                               uint32_t k, uint8_t *mask, uint32_t *restrict closest_idx, double *restrict closest_dist)
{
    double cur_dist;
    uint32_t i;
    /* Loop through all points in leaf */
    for (i = 0; i < n; i++)
    {
        /* Is this point masked out? */
        if (mask[pidx[start_idx + i]])
        {
            continue;
        }
        /* Get distance to query point */
        cur_dist = calc_dist_double(&PA(start_idx + i, 0), point_coord, no_dims);
        /* Update closest info if new point is closest so far*/
        if (cur_dist < closest_dist[k - 1])
        {
            insert_point_double_int32_t(closest_idx, closest_dist, pidx[start_idx + i], cur_dist, k);
        }
    }
}

/************************************************
Search subtree for nearest to query point
Params:
    root : root node of subtree
    pa : data points
    pidx : permutation index of data points
    no_dims : number of dimensions
    point_coord : query point
    min_dist : minumum distance to nearest neighbour
    mask : boolean array of invalid (True) and valid (False) data points
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_splitnode_double_int32_t(Node_double_int32_t *root, double *pa, uint32_t *pidx, int8_t no_dims, double *point_coord, 
                      double min_dist, uint32_t k, double distance_upper_bound, double eps_fac, uint8_t *mask,
                      uint32_t *closest_idx, double *closest_dist)
{
    int8_t dim;
    double dist_left, dist_right;
    double new_offset;
    double box_diff;

    /* Skip if distance bound exeeded */
    if (min_dist > distance_upper_bound)
    {
        return;
    }

    dim = root->cut_dim;

    /* Handle leaf node */
    if (dim == -1)
    {
        if (mask)
        {
            search_leaf_double_int32_t_mask(pa, pidx, no_dims, root->start_idx, root->n, point_coord, k, mask, closest_idx, closest_dist);
        }
        else
        {
            search_leaf_double_int32_t(pa, pidx, no_dims, root->start_idx, root->n, point_coord, k, closest_idx, closest_dist);
        }
        return;
    }

    /* Get distance to cutting plane */
    new_offset = point_coord[dim] - root->cut_val;

    if (new_offset < 0)
    {
        /* Left of cutting plane */
        dist_left = min_dist;
        if (dist_left < closest_dist[k - 1] * eps_fac)
        {
            /* Search left subtree if minimum distance is below limit */
            search_splitnode_double_int32_t((Node_double_int32_t *)root->left_child, pa, pidx, no_dims, point_coord, dist_left, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }

        /* Right of cutting plane. Update minimum distance.
           See Algorithms for Fast Vector Quantization
           Sunil Arya and David M. Mount. */
        box_diff = root->cut_bounds_lv - point_coord[dim];
        if (box_diff < 0)
        {
		box_diff = 0;
        }
        dist_right = min_dist - box_diff * box_diff + new_offset * new_offset;
        if (dist_right < closest_dist[k - 1] * eps_fac)
        {
            /* Search right subtree if minimum distance is below limit*/
            search_splitnode_double_int32_t((Node_double_int32_t *)root->right_child, pa, pidx, no_dims, point_coord, dist_right, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }
    }
    else
    {
        /* Right of cutting plane */
        dist_right = min_dist;
        if (dist_right < closest_dist[k - 1] * eps_fac)
        {
            /* Search right subtree if minimum distance is below limit*/
            search_splitnode_double_int32_t((Node_double_int32_t *)root->right_child, pa, pidx, no_dims, point_coord, dist_right, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }

        /* Left of cutting plane. Update minimum distance.
           See Algorithms for Fast Vector Quantization
           Sunil Arya and David M. Mount. */
        box_diff = point_coord[dim] - root->cut_bounds_hv;
        if (box_diff < 0)
        {
        	box_diff = 0;
        }
        dist_left = min_dist - box_diff * box_diff + new_offset * new_offset;
	  if (dist_left < closest_dist[k - 1] * eps_fac)
        {
            /* Search left subtree if minimum distance is below limit*/
            search_splitnode_double_int32_t((Node_double_int32_t *)root->left_child, pa, pidx, no_dims, point_coord, dist_left, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }
    }
}

/************************************************
Search for nearest neighbour for a set of query points
Params:
    tree : Tree struct of kd tree
    pa : data points
    pidx : permutation index of data points
    point_coords : query points
    num_points : number of query points
    mask : boolean array of invalid (True) and valid (False) data points
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_tree_double_int32_t(Tree_double_int32_t *tree, double *pa, double *point_coords,
                 uint32_t num_points, uint32_t k, double distance_upper_bound,
                 double eps, uint8_t *mask, uint32_t *closest_idxs, double *closest_dists)
{
    double min_dist;
    double eps_fac = 1 / ((1 + eps) * (1 + eps));
    int8_t no_dims = tree->no_dims;
    double *bbox = tree->bbox;
    uint32_t *pidx = tree->pidx;
    /* use 64-bit ints for indexing to avoid overflow, use signed ints to support all Openmp implementations */
    int64_t i = 0;
    int64_t j = 0;
    int64_t local_num_points = (int64_t) num_points;
    Node_double_int32_t *root = (Node_double_int32_t *)tree->root;

    /* Queries are OpenMP enabled */
    #pragma omp parallel
    {
        /* The low chunk size is important to avoid L2 cache trashing
           for spatial coherent query datasets
        */
        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            for (j = 0; j < k; j++)
            {
                closest_idxs[i * k + j] = IDX_MAX_int32_t;
                closest_dists[i * k + j] = DIST_MAX_double;
            }
            min_dist = get_min_dist_double(point_coords + no_dims * i, no_dims, bbox);
            search_splitnode_double_int32_t(root, pa, pidx, no_dims, point_coords + no_dims * i, min_dist,
                             k, distance_upper_bound, eps_fac, mask, &closest_idxs[i * k], &closest_dists[i * k]);
        }
    }
}

/************************************************
Compute mean k-NN Euclidean distances for Statistical Outlier Removal.
Self-queries data points, skips closest neighbor (self), computes
mean Euclidean distance to k remaining neighbors.
Params:
    tree : Tree struct of kd tree
    pa : data points (also used as query points)
    num_points : number of data points
    k : number of neighbors (excluding self)
    mean_dists_out : mean distances output, shape (num_points,)
************************************************/
void sor_mean_dists_double_int32_t(Tree_double_int32_t *tree, double *pa,
                 uint32_t num_points, uint32_t k, double *mean_dists_out)
{
    int8_t no_dims = tree->no_dims;
    double *bbox = tree->bbox;
    uint32_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    uint32_t k_total = k + 1;  /* +1 to include self */
    int64_t local_k_total = (int64_t) k_total;
    Node_double_int32_t *root = (Node_double_int32_t *)tree->root;

    #pragma omp parallel
    {
        uint32_t *local_idx = (uint32_t *)malloc(k_total * sizeof(uint32_t));
        double *local_dist = (double *)malloc(k_total * sizeof(double));

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            double min_dist, sum;
            int64_t count;

            /* Initialize k-NN arrays */
            for (j = 0; j < local_k_total; j++)
            {
                local_idx[j] = IDX_MAX_int32_t;
                local_dist[j] = DIST_MAX_double;
            }

            /* Query k+1 neighbors (includes self) */
            min_dist = get_min_dist_double(pa + no_dims * i, no_dims, bbox);
            search_splitnode_double_int32_t(root, pa, pidx, no_dims,
                             pa + no_dims * i, min_dist,
                             k_total, DIST_MAX_double, (double)1.0, NULL,
                             local_idx, local_dist);

            /* Mean Euclidean distance, skipping closest (self, index 0) */
            sum = 0;
            count = 0;
            for (j = 1; j < local_k_total; j++)
            {
                if (local_dist[j] < DIST_MAX_double)
                {
                    sum += sqrt(local_dist[j]);
                    count++;
                }
            }
            mean_dists_out[i] = (count > 0) ? sum / (double)count : 0;
        }

        free(local_idx);
        free(local_dist);
    }
}

/************************************************
Compute comprehensive point descriptors in a single k-NN pass.
Outputs NUM_DESCRIPTORS (20) features per point per scale:
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
  14:  z_range (z_max - z_min of neighbors)
  15:  z_above (max_z_neighbor - query_z)
  16:  z_below (query_z - min_z_neighbor)
  17:  z_std (height standard deviation of neighbors)
  18:  density (k / bounding_box_volume)
  19:  roughness (|point-to-plane distance|)
k_scales must be sorted ascending.
************************************************/
void compute_descriptors_multiscale_double_int32_t(Tree_double_int32_t *tree, double *pa, double *point_coords,
                 uint32_t num_points, uint32_t k_max,
                 int32_t *k_scales, int32_t num_scales,
                 double distance_upper_bound, double eps, uint8_t *mask,
                 double *descriptors_out)
{
    double eps_fac = 1 / ((1 + eps) * (1 + eps));
    int8_t no_dims = tree->no_dims;
    double *bbox = tree->bbox;
    uint32_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    int64_t local_k_max = (int64_t) k_max;
    Node_double_int32_t *root = (Node_double_int32_t *)tree->root;

    #pragma omp parallel
    {
        uint32_t *local_idx = (uint32_t *)malloc(k_max * sizeof(uint32_t));
        double *local_dist = (double *)malloc(k_max * sizeof(double));

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            double min_dist;
            int32_t s, scale_idx;
            double sum_x, sum_y, sum_z;
            double sum_xx, sum_xy, sum_xz, sum_yy, sum_yz, sum_zz;
            double bb_min_x, bb_min_y, bb_min_z, bb_max_x, bb_max_y, bb_max_z;
            double *pt;
            double px, py, pz;
            double qx = point_coords[no_dims * i];
            double qy = point_coords[no_dims * i + 1];
            double qz = point_coords[no_dims * i + 2];
            int64_t out_base = (int64_t)i * num_scales * NUM_DESCRIPTORS;

            /* Initialize k-NN arrays */
            for (j = 0; j < local_k_max; j++)
            {
                local_idx[j] = IDX_MAX_int32_t;
                local_dist[j] = DIST_MAX_double;
            }

            /* Query k_max neighbors */
            min_dist = get_min_dist_double(point_coords + no_dims * i, no_dims, bbox);
            search_splitnode_double_int32_t(root, pa, pidx, no_dims,
                             point_coords + no_dims * i, min_dist,
                             k_max, distance_upper_bound, eps_fac, mask,
                             local_idx, local_dist);

            /* Initialize accumulators */
            sum_x = sum_y = sum_z = 0;
            sum_xx = sum_xy = sum_xz = sum_yy = sum_yz = sum_zz = 0;
            bb_min_x = bb_min_y = bb_min_z = DIST_MAX_double;
            bb_max_x = bb_max_y = bb_max_z = -DIST_MAX_double;
            scale_idx = 0;

            for (j = 0; j < local_k_max && scale_idx < num_scales; j++)
            {
                if (local_idx[j] == IDX_MAX_int32_t || local_dist[j] >= DIST_MAX_double)
                    break;

                pt = pa + no_dims * local_idx[j];
                px = pt[0]; py = pt[1]; pz = pt[2];

                /* Accumulate sums for covariance */
                sum_x += px; sum_y += py; sum_z += pz;
                sum_xx += px * px; sum_xy += px * py; sum_xz += px * pz;
                sum_yy += py * py; sum_yz += py * pz; sum_zz += pz * pz;

                /* Update bounding box */
                if (px < bb_min_x) bb_min_x = px;
                if (px > bb_max_x) bb_max_x = px;
                if (py < bb_min_y) bb_min_y = py;
                if (py > bb_max_y) bb_max_y = py;
                if (pz < bb_min_z) bb_min_z = pz;
                if (pz > bb_max_z) bb_max_z = pz;

                /* At each scale boundary, compute all descriptors */
                if ((int32_t)(j + 1) == k_scales[scale_idx])
                {
                    int64_t out_off = out_base + (int64_t)scale_idx * NUM_DESCRIPTORS;
                    double evals[3], normal[3];

                    if (j + 1 < 2)
                    {
                        for (s = 0; s < NUM_DESCRIPTORS; s++)
                            descriptors_out[out_off + s] = 0;
                        descriptors_out[out_off + 5] = 1;  /* nz = 1 */
                    }
                    else
                    {
                        double inv_k = 1 / (double)(j + 1);
                        double mx = sum_x * inv_k;
                        double my = sum_y * inv_k;
                        double mz = sum_z * inv_k;

                        double cov_xx = sum_xx * inv_k - mx * mx;
                        double cov_xy = sum_xy * inv_k - mx * my;
                        double cov_xz = sum_xz * inv_k - mx * mz;
                        double cov_yy = sum_yy * inv_k - my * my;
                        double cov_yz = sum_yz * inv_k - my * mz;
                        double cov_zz = sum_zz * inv_k - mz * mz;

                        eigen_symmetric_3x3_double(cov_xx, cov_xy, cov_xz,
                                                     cov_yy, cov_yz, cov_zz,
                                                     evals, normal);

                        /* 0-2: eigenvalues */
                        descriptors_out[out_off + 0] = evals[0];
                        descriptors_out[out_off + 1] = evals[1];
                        descriptors_out[out_off + 2] = evals[2];

                        /* 3-5: normal */
                        descriptors_out[out_off + 3] = normal[0];
                        descriptors_out[out_off + 4] = normal[1];
                        descriptors_out[out_off + 5] = normal[2];

                        /* 6: verticality = 1 - |nz| */
                        descriptors_out[out_off + 6] = 1 - (normal[2] >= 0 ? normal[2] : -normal[2]);

                        /* 7-13: derived eigenvalue features */
                        {
                            double sum_eig = evals[0] + evals[1] + evals[2];
                            double inv_l1 = evals[0] > (double)1e-30 ? 1 / evals[0] : 0;
                            double inv_sum = sum_eig > (double)1e-30 ? 1 / sum_eig : 0;

                            /* 7: linearity = (l1 - l2) / l1 */
                            descriptors_out[out_off + 7] = (evals[0] - evals[1]) * inv_l1;

                            /* 8: planarity = (l2 - l3) / l1 */
                            descriptors_out[out_off + 8] = (evals[1] - evals[2]) * inv_l1;

                            /* 9: sphericity = l3 / l1 */
                            descriptors_out[out_off + 9] = evals[2] * inv_l1;

                            /* 10: omnivariance = (l1 * l2 * l3)^(1/3) */
                            {
                                double prod = evals[0] * evals[1] * evals[2];
                                descriptors_out[out_off + 10] = prod > 0 ? cbrt(prod) : 0;
                            }

                            /* 11: anisotropy = (l1 - l3) / l1 */
                            descriptors_out[out_off + 11] = (evals[0] - evals[2]) * inv_l1;

                            /* 12: eigenentropy = -sum(li/S * ln(li/S)) */
                            {
                                double entropy = 0;
                                if (sum_eig > (double)1e-30)
                                {
                                    int32_t ei;
                                    for (ei = 0; ei < 3; ei++)
                                    {
                                        double p = evals[ei] * inv_sum;
                                        if (p > (double)1e-30)
                                            entropy -= p * log(p);
                                    }
                                }
                                descriptors_out[out_off + 12] = entropy;
                            }

                            /* 13: surface_variation = l3 / (l1 + l2 + l3) */
                            descriptors_out[out_off + 13] = evals[2] * inv_sum;
                        }

                        /* 14: z_range */
                        descriptors_out[out_off + 14] = bb_max_z - bb_min_z;

                        /* 15: z_above = max_z - query_z */
                        descriptors_out[out_off + 15] = bb_max_z - qz;

                        /* 16: z_below = query_z - min_z */
                        descriptors_out[out_off + 16] = qz - bb_min_z;

                        /* 17: z_std = sqrt(cov_zz) */
                        descriptors_out[out_off + 17] = cov_zz > 0 ? sqrt(cov_zz) : 0;

                        /* 18: density = k / bbox_volume */
                        {
                            double vol = (bb_max_x - bb_min_x) * (bb_max_y - bb_min_y) * (bb_max_z - bb_min_z);
                            descriptors_out[out_off + 18] = vol > (double)1e-30 ? (double)(j + 1) / vol : 0;
                        }

                        /* 19: roughness = |dot(query - centroid, normal)| */
                        {
                            double dot = (qx - mx) * normal[0] + (qy - my) * normal[1] + (qz - mz) * normal[2];
                            descriptors_out[out_off + 19] = dot >= 0 ? dot : -dot;
                        }
                    }
                    scale_idx++;
                }
            }

            /* Fill remaining scales with zeros */
            for (s = scale_idx; s < num_scales; s++)
            {
                int64_t out_off = out_base + (int64_t)s * NUM_DESCRIPTORS;
                int32_t f;
                for (f = 0; f < NUM_DESCRIPTORS; f++)
                    descriptors_out[out_off + f] = 0;
                descriptors_out[out_off + 5] = 1;  /* nz = 1 */
            }
        }

        free(local_idx);
        free(local_dist);
    }
}

/************************************************
Radius-based outlier filter.
Queries k_min neighbors for each point. If the k_min-th
neighbor is beyond radius, the point is marked as outlier.
Params:
    tree : Tree struct
    pa : data points (also query points)
    num_points : number of points
    k_min : minimum number of neighbors required within radius
    radius : search radius (Euclidean)
    inlier_mask_out : (num_points,) output, 1 = inlier
    count_out : number of inliers (return, may be NULL)
************************************************/
void radius_filter_double_int32_t(Tree_double_int32_t *tree, double *pa,
                 uint32_t num_points, uint32_t k_min, double radius,
                 uint8_t *inlier_mask_out, uint32_t *count_out)
{
    int8_t no_dims = tree->no_dims;
    double *bbox = tree->bbox;
    uint32_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    /* Query k_min+1 to include self */
    uint32_t k_total = k_min + 1;
    int64_t local_k_total = (int64_t) k_total;
    double radius_sq = radius * radius;
    Node_double_int32_t *root = (Node_double_int32_t *)tree->root;
    uint32_t total_inliers = 0;

    #pragma omp parallel
    {
        uint32_t *local_idx = (uint32_t *)malloc(k_total * sizeof(uint32_t));
        double *local_dist = (double *)malloc(k_total * sizeof(double));
        uint32_t thread_inliers = 0;

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            double min_dist;
            uint32_t count;

            for (j = 0; j < local_k_total; j++)
            {
                local_idx[j] = IDX_MAX_int32_t;
                local_dist[j] = DIST_MAX_double;
            }

            min_dist = get_min_dist_double(pa + no_dims * i, no_dims, bbox);
            search_splitnode_double_int32_t(root, pa, pidx, no_dims,
                             pa + no_dims * i, min_dist,
                             k_total, radius_sq, (double)1.0, NULL,
                             local_idx, local_dist);

            /* Count valid neighbors (excluding self at index 0) */
            count = 0;
            for (j = 1; j < local_k_total; j++)
            {
                if (local_dist[j] < radius_sq && local_idx[j] != IDX_MAX_int32_t)
                    count++;
            }

            if (count >= k_min)
            {
                inlier_mask_out[i] = 1;
                thread_inliers++;
            }
            else
            {
                inlier_mask_out[i] = 0;
            }
        }

        #pragma omp atomic
        total_inliers += thread_inliers;

        free(local_idx);
        free(local_dist);
    }

    if (count_out) *count_out = total_inliers;
}

/************************************************
Estimate normals and curvatures from k-NN neighborhoods.
Lightweight alternative to compute_descriptors when only
normals and curvature are needed.
Params:
    tree : Tree struct
    pa : data points
    point_coords : query points (n * 3)
    num_points : number of query points
    k : number of neighbors
    distance_upper_bound : max distance
    eps : approximation factor
    mask : point validity mask (may be NULL)
    normals_out : (num_points * 3) output normals
    curvatures_out : (num_points,) output curvatures (surface variation)
************************************************/
void estimate_normals_double_int32_t(Tree_double_int32_t *tree, double *pa,
                 double *point_coords, uint32_t num_points, uint32_t k,
                 double distance_upper_bound, double eps, uint8_t *mask,
                 double *normals_out, double *curvatures_out)
{
    double eps_fac = 1 / ((1 + eps) * (1 + eps));
    int8_t no_dims = tree->no_dims;
    double *bbox = tree->bbox;
    uint32_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    int64_t local_k = (int64_t) k;
    Node_double_int32_t *root = (Node_double_int32_t *)tree->root;

    #pragma omp parallel
    {
        uint32_t *local_idx = (uint32_t *)malloc(k * sizeof(uint32_t));
        double *local_dist = (double *)malloc(k * sizeof(double));

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            double min_dist;
            double sum_x, sum_y, sum_z;
            double sum_xx, sum_xy, sum_xz, sum_yy, sum_yz, sum_zz;
            double *pt;
            double px, py, pz;
            int64_t valid_count;

            /* Initialize */
            for (j = 0; j < local_k; j++)
            {
                local_idx[j] = IDX_MAX_int32_t;
                local_dist[j] = DIST_MAX_double;
            }

            /* Query k neighbors */
            min_dist = get_min_dist_double(point_coords + no_dims * i, no_dims, bbox);
            search_splitnode_double_int32_t(root, pa, pidx, no_dims,
                             point_coords + no_dims * i, min_dist,
                             k, distance_upper_bound, eps_fac, mask,
                             local_idx, local_dist);

            /* Accumulate covariance sums */
            sum_x = sum_y = sum_z = 0;
            sum_xx = sum_xy = sum_xz = sum_yy = sum_yz = sum_zz = 0;
            valid_count = 0;

            for (j = 0; j < local_k; j++)
            {
                if (local_idx[j] == IDX_MAX_int32_t || local_dist[j] >= DIST_MAX_double)
                    break;
                pt = pa + no_dims * local_idx[j];
                px = pt[0]; py = pt[1]; pz = pt[2];
                sum_x += px; sum_y += py; sum_z += pz;
                sum_xx += px * px; sum_xy += px * py; sum_xz += px * pz;
                sum_yy += py * py; sum_yz += py * pz; sum_zz += pz * pz;
                valid_count++;
            }

            if (valid_count < 2)
            {
                normals_out[3 * i]     = 0;
                normals_out[3 * i + 1] = 0;
                normals_out[3 * i + 2] = 1;
                curvatures_out[i] = 0;
            }
            else
            {
                double inv_k = 1 / (double)valid_count;
                double mx = sum_x * inv_k, my = sum_y * inv_k, mz = sum_z * inv_k;
                double cov_xx = sum_xx * inv_k - mx * mx;
                double cov_xy = sum_xy * inv_k - mx * my;
                double cov_xz = sum_xz * inv_k - mx * mz;
                double cov_yy = sum_yy * inv_k - my * my;
                double cov_yz = sum_yz * inv_k - my * mz;
                double cov_zz = sum_zz * inv_k - mz * mz;

                double evals[3], normal[3];
                eigen_symmetric_3x3_double(cov_xx, cov_xy, cov_xz,
                                             cov_yy, cov_yz, cov_zz,
                                             evals, normal);

                /* Orient normal upward */
                if (normal[2] < 0) { normal[0] = -normal[0]; normal[1] = -normal[1]; normal[2] = -normal[2]; }

                normals_out[3 * i]     = normal[0];
                normals_out[3 * i + 1] = normal[1];
                normals_out[3 * i + 2] = normal[2];

                /* Curvature = surface variation = l3 / (l1 + l2 + l3) */
                {
                    double sum_eig = evals[0] + evals[1] + evals[2];
                    curvatures_out[i] = sum_eig > (double)1e-30 ? evals[2] / sum_eig : 0;
                }
            }
        }

        free(local_idx);
        free(local_dist);
    }
}

/************************************************
Insert point into priority queue
Params:
    closest_idx : index queue
    closest_dist : distance queue
    pidx : permutation index of data points
    cur_dist : distance to point inserted
    k : number of neighbours
************************************************/
void insert_point_double_int64_t(uint64_t *closest_idx, double *closest_dist, uint64_t pidx, double cur_dist, uint64_t k)
{
    int i;
    for (i = k - 1; i > 0; i--)
    {
        if (closest_dist[i - 1] > cur_dist)
        {
            closest_dist[i] = closest_dist[i - 1];
            closest_idx[i] = closest_idx[i - 1];
        }
        else
        {
            break;
        }
    }
    closest_idx[i] = pidx;
    closest_dist[i] = cur_dist;
}

/************************************************
Get the bounding box of a set of points
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims: number of dimensions
    n : number of points
    bbox : bounding box (return)
************************************************/
void get_bounding_box_double_int64_t(double *pa, uint64_t *pidx, int8_t no_dims, uint64_t n, double *bbox)
{
    double cur;
    int8_t i, j;
    uint64_t bbox_idx, i2;

    /* Use first data point to initialize */
    for (i = 0; i < no_dims; i++)
    {
        bbox[2 * i] = bbox[2 * i + 1] = PA(0, i);
    }

    /* Update using rest of data points */
    for (i2 = 1; i2 < n; i2++)
    {
        for (j = 0; j < no_dims; j++)
        {
            bbox_idx = 2 * j;
            cur = PA(i2, j);
            if (cur < bbox[bbox_idx])
            {
                bbox[bbox_idx] = cur;
            }
            else if (cur > bbox[bbox_idx + 1])
            {
                bbox[bbox_idx + 1] = cur;
            }
        }
    }
}

/************************************************
Partition a range of data points by manipulation the permutation index.
The sliding midpoint rule is used for the partitioning.
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims: number of dimensions
    start_idx : index of first data point to use
    n :  number of data points
    bbox : bounding box of data points
    cut_dim : dimension used for partition (return)
    cut_val : value of cutting point (return)
    n_lo : number of point below cutting plane (return)
************************************************/
int partition_double_int64_t(double *pa, uint64_t *pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, double *bbox, int8_t *cut_dim, double *cut_val, uint64_t *n_lo)
{
    int8_t dim = 0, i;
    uint64_t p, q, i2;
    double size = 0, min_val, max_val, split, side_len, cur_val;
    uint64_t end_idx = start_idx + n - 1;

    /* Find largest bounding box side */
    for (i = 0; i < no_dims; i++)
    {
        side_len = bbox[2 * i + 1] - bbox[2 * i];
        if (side_len > size)
        {
            dim = i;
            size = side_len;
        }
    }

    min_val = bbox[2 * dim];
    max_val = bbox[2 * dim + 1];

    /* Check for zero length or inconsistent */
    if (min_val >= max_val)
        return 1;

    /* Use middle for splitting */
    split = (min_val + max_val) / 2;

    /* Partition all data points around middle */
    p = start_idx;
    q = end_idx;
    while (p <= q)
    {
        if (PA(p, dim) < split)
        {
            p++;
        }
        else if (PA(q, dim) >= split)
        {
            /* Guard for underflow */
            if (q > 0)
            {
                q--;
            }
            else
            {
                break;
            }
        }
        else
        {
            PASWAP_int64_t(p, q);
            p++;
            q--;
        }
    }

    /* Check for empty splits */
    if (p == start_idx)
    {
        /* No points less than split.
           Split at lowest point instead.
           Minimum 1 point will be in lower box.
        */

        uint64_t j = start_idx;
        split = PA(j, dim);
        for (i2 = start_idx + 1; i2 <= end_idx; i2++)
        {
            /* Find lowest point */
            cur_val = PA(i2, dim);
            if (cur_val < split)
            {
                j = i2;
                split = cur_val;
            }
        }
        PASWAP_int64_t(j, start_idx);
        p = start_idx + 1;
    }
    else if (p == end_idx + 1)
    {
        /* No points greater than split.
           Split at highest point instead.
           Minimum 1 point will be in higher box.
        */

        uint64_t j = end_idx;
        split = PA(j, dim);
        for (i2 = start_idx; i2 < end_idx; i2++)
        {
            /* Find highest point */
            cur_val = PA(i2, dim);
            if (cur_val > split)
            {
                j = i2;
                split = cur_val;
            }
        }
        PASWAP_int64_t(j, end_idx);
        p = end_idx;
    }

    /* Set return values */
    *cut_dim = dim;
    *cut_val = split;
    *n_lo = p - start_idx;
    return 0;
}

/************************************************
Construct a sub tree over a range of data points.
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims: number of dimensions
    start_idx : index of first data point to use
    n :  number of data points
    bsp : number of points per leaf
    bbox : bounding box of set of data points
************************************************/
Node_double_int64_t* construct_subtree_double_int64_t(double *pa, uint64_t *pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, uint64_t bsp, double *bbox)
{
    /* Create new node */
    int is_leaf = (n <= bsp);
    Node_double_int64_t *root = create_node_double_int64_t(start_idx, n, is_leaf);
    int rval;
    int8_t cut_dim;
    uint64_t n_lo;
    double cut_val, lv, hv;
    if (is_leaf)
    {
        /* Make leaf node */
        root->cut_dim = -1;
    }
    else
    {
        /* Make split node */
        /* Partition data set and set node info */
        rval = partition_double_int64_t(pa, pidx, no_dims, start_idx, n, bbox, &cut_dim, &cut_val, &n_lo);
        if (rval == 1)
        {
            root->cut_dim = -1;
            return root;
        }
        root->cut_val = cut_val;
        root->cut_dim = cut_dim;

        /* Recurse on both subsets */
        lv = bbox[2 * cut_dim];
        hv = bbox[2 * cut_dim + 1];

        /* Set bounds for cut dimension */
        root->cut_bounds_lv = lv;
        root->cut_bounds_hv = hv;

        /* Update bounding box before call to lower subset and restore after */
        bbox[2 * cut_dim + 1] = cut_val;
        root->left_child = (struct Node_double_int64_t *)construct_subtree_double_int64_t(pa, pidx, no_dims, start_idx, n_lo, bsp, bbox);
        bbox[2 * cut_dim + 1] = hv;

        /* Update bounding box before call to higher subset and restore after */
        bbox[2 * cut_dim] = cut_val;
        root->right_child = (struct Node_double_int64_t *)construct_subtree_double_int64_t(pa, pidx, no_dims, start_idx + n_lo, n - n_lo, bsp, bbox);
        bbox[2 * cut_dim] = lv;
    }
    return root;
}

/************************************************
Construct a tree over data points.
Params:
    pa : data points
    no_dims: number of dimensions
    n :  number of data points
    bsp : number of points per leaf
************************************************/
Tree_double_int64_t* construct_tree_double_int64_t(double *pa, int8_t no_dims, uint64_t n, uint64_t bsp)
{
    Tree_double_int64_t *tree = (Tree_double_int64_t *)malloc(sizeof(Tree_double_int64_t));
    uint64_t i;
    uint64_t *pidx;
    double *bbox;

    tree->no_dims = no_dims;

    /* Initialize permutation array */
    pidx = (uint64_t *)malloc(sizeof(uint64_t) * n);
    for (i = 0; i < n; i++)
    {
        pidx[i] = i;
    }

    bbox = (double *)malloc(2 * sizeof(double) * no_dims);
    get_bounding_box_double_int64_t(pa, pidx, no_dims, n, bbox);
    tree->bbox = bbox;

    /* Construct subtree on full dataset */
    tree->root = (struct Node_double_int64_t *)construct_subtree_double_int64_t(pa, pidx, no_dims, 0, n, bsp, bbox);

    tree->pidx = pidx;
    return tree;
}

/************************************************
Create a tree node.
Params:
    start_idx : index of first data point to use
    n :  number of data points
************************************************/
Node_double_int64_t* create_node_double_int64_t(uint64_t start_idx, uint64_t n, int is_leaf)
{
    Node_double_int64_t *new_node;
    if (is_leaf)
    {
        /*
            Allocate only the part of the struct that will be used in a leaf node.
            This relies on the C99 specification of struct layout conservation and padding and
            that dereferencing is never attempted for the node pointers in a leaf.
        */
        new_node = (Node_double_int64_t *)malloc(sizeof(Node_double_int64_t) - 2 * sizeof(Node_double_int64_t *));
    }
    else
    {
        new_node = (Node_double_int64_t *)malloc(sizeof(Node_double_int64_t));
    }
    new_node->n = n;
    new_node->start_idx = start_idx;
    return new_node;
}

/************************************************
Delete subtree
Params:
    root : root node of subtree to delete
************************************************/
void delete_subtree_double_int64_t(Node_double_int64_t *root)
{
    if (root->cut_dim != -1)
    {
        delete_subtree_double_int64_t((Node_double_int64_t *)root->left_child);
        delete_subtree_double_int64_t((Node_double_int64_t *)root->right_child);
    }
    free(root);
}

/************************************************
Delete tree
Params:
    tree : Tree struct of kd tree
************************************************/
void delete_tree_double_int64_t(Tree_double_int64_t *tree)
{
    delete_subtree_double_int64_t((Node_double_int64_t *)tree->root);
    free(tree->bbox);
    free(tree->pidx);
    free(tree);
}

/************************************************
Print
************************************************/
void print_tree_double_int64_t(Node_double_int64_t *root, int level)
{
    int i;
    for (i = 0; i < level; i++)
    {
        printf(" ");
    }
    printf("(cut_val: %f, cut_dim: %i)\n", root->cut_val, root->cut_dim);
    if (root->cut_dim != -1)
        print_tree_double_int64_t((Node_double_int64_t *)root->left_child, level + 1);
    if (root->cut_dim != -1)
        print_tree_double_int64_t((Node_double_int64_t *)root->right_child, level + 1);
}

/************************************************
Search a leaf node for closest point
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims : number of dimensions
    start_idx : index of first data point to use
    size :  number of data points
    point_coord : query point
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_leaf_double_int64_t(double *restrict pa, uint64_t *restrict pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, double *restrict point_coord,
                 uint64_t k, uint64_t *restrict closest_idx, double *restrict closest_dist)
{
    double cur_dist;
    uint64_t i;
    /* Loop through all points in leaf */
    for (i = 0; i < n; i++)
    {
        /* Get distance to query point */
        cur_dist = calc_dist_double(&PA(start_idx + i, 0), point_coord, no_dims);
        /* Update closest info if new point is closest so far*/
        if (cur_dist < closest_dist[k - 1])
        {
            insert_point_double_int64_t(closest_idx, closest_dist, pidx[start_idx + i], cur_dist, k);
        }
    }
}


/************************************************
Search a leaf node for closest point with data point mask
Params:
    pa : data points
    pidx : permutation index of data points
    no_dims : number of dimensions
    start_idx : index of first data point to use
    size :  number of data points
    point_coord : query point
    mask : boolean array of invalid (True) and valid (False) data points
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_leaf_double_int64_t_mask(double *restrict pa, uint64_t *restrict pidx, int8_t no_dims, uint64_t start_idx, uint64_t n, double *restrict point_coord,
                               uint64_t k, uint8_t *mask, uint64_t *restrict closest_idx, double *restrict closest_dist)
{
    double cur_dist;
    uint64_t i;
    /* Loop through all points in leaf */
    for (i = 0; i < n; i++)
    {
        /* Is this point masked out? */
        if (mask[pidx[start_idx + i]])
        {
            continue;
        }
        /* Get distance to query point */
        cur_dist = calc_dist_double(&PA(start_idx + i, 0), point_coord, no_dims);
        /* Update closest info if new point is closest so far*/
        if (cur_dist < closest_dist[k - 1])
        {
            insert_point_double_int64_t(closest_idx, closest_dist, pidx[start_idx + i], cur_dist, k);
        }
    }
}

/************************************************
Search subtree for nearest to query point
Params:
    root : root node of subtree
    pa : data points
    pidx : permutation index of data points
    no_dims : number of dimensions
    point_coord : query point
    min_dist : minumum distance to nearest neighbour
    mask : boolean array of invalid (True) and valid (False) data points
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_splitnode_double_int64_t(Node_double_int64_t *root, double *pa, uint64_t *pidx, int8_t no_dims, double *point_coord, 
                      double min_dist, uint64_t k, double distance_upper_bound, double eps_fac, uint8_t *mask,
                      uint64_t *closest_idx, double *closest_dist)
{
    int8_t dim;
    double dist_left, dist_right;
    double new_offset;
    double box_diff;

    /* Skip if distance bound exeeded */
    if (min_dist > distance_upper_bound)
    {
        return;
    }

    dim = root->cut_dim;

    /* Handle leaf node */
    if (dim == -1)
    {
        if (mask)
        {
            search_leaf_double_int64_t_mask(pa, pidx, no_dims, root->start_idx, root->n, point_coord, k, mask, closest_idx, closest_dist);
        }
        else
        {
            search_leaf_double_int64_t(pa, pidx, no_dims, root->start_idx, root->n, point_coord, k, closest_idx, closest_dist);
        }
        return;
    }

    /* Get distance to cutting plane */
    new_offset = point_coord[dim] - root->cut_val;

    if (new_offset < 0)
    {
        /* Left of cutting plane */
        dist_left = min_dist;
        if (dist_left < closest_dist[k - 1] * eps_fac)
        {
            /* Search left subtree if minimum distance is below limit */
            search_splitnode_double_int64_t((Node_double_int64_t *)root->left_child, pa, pidx, no_dims, point_coord, dist_left, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }

        /* Right of cutting plane. Update minimum distance.
           See Algorithms for Fast Vector Quantization
           Sunil Arya and David M. Mount. */
        box_diff = root->cut_bounds_lv - point_coord[dim];
        if (box_diff < 0)
        {
		box_diff = 0;
        }
        dist_right = min_dist - box_diff * box_diff + new_offset * new_offset;
        if (dist_right < closest_dist[k - 1] * eps_fac)
        {
            /* Search right subtree if minimum distance is below limit*/
            search_splitnode_double_int64_t((Node_double_int64_t *)root->right_child, pa, pidx, no_dims, point_coord, dist_right, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }
    }
    else
    {
        /* Right of cutting plane */
        dist_right = min_dist;
        if (dist_right < closest_dist[k - 1] * eps_fac)
        {
            /* Search right subtree if minimum distance is below limit*/
            search_splitnode_double_int64_t((Node_double_int64_t *)root->right_child, pa, pidx, no_dims, point_coord, dist_right, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }

        /* Left of cutting plane. Update minimum distance.
           See Algorithms for Fast Vector Quantization
           Sunil Arya and David M. Mount. */
        box_diff = point_coord[dim] - root->cut_bounds_hv;
        if (box_diff < 0)
        {
        	box_diff = 0;
        }
        dist_left = min_dist - box_diff * box_diff + new_offset * new_offset;
	  if (dist_left < closest_dist[k - 1] * eps_fac)
        {
            /* Search left subtree if minimum distance is below limit*/
            search_splitnode_double_int64_t((Node_double_int64_t *)root->left_child, pa, pidx, no_dims, point_coord, dist_left, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }
    }
}

/************************************************
Search for nearest neighbour for a set of query points
Params:
    tree : Tree struct of kd tree
    pa : data points
    pidx : permutation index of data points
    point_coords : query points
    num_points : number of query points
    mask : boolean array of invalid (True) and valid (False) data points
    closest_idx : index of closest data point found (return)
    closest_dist : distance to closest point (return)
************************************************/
void search_tree_double_int64_t(Tree_double_int64_t *tree, double *pa, double *point_coords,
                 uint64_t num_points, uint64_t k, double distance_upper_bound,
                 double eps, uint8_t *mask, uint64_t *closest_idxs, double *closest_dists)
{
    double min_dist;
    double eps_fac = 1 / ((1 + eps) * (1 + eps));
    int8_t no_dims = tree->no_dims;
    double *bbox = tree->bbox;
    uint64_t *pidx = tree->pidx;
    /* use 64-bit ints for indexing to avoid overflow, use signed ints to support all Openmp implementations */
    int64_t i = 0;
    int64_t j = 0;
    int64_t local_num_points = (int64_t) num_points;
    Node_double_int64_t *root = (Node_double_int64_t *)tree->root;

    /* Queries are OpenMP enabled */
    #pragma omp parallel
    {
        /* The low chunk size is important to avoid L2 cache trashing
           for spatial coherent query datasets
        */
        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            for (j = 0; j < k; j++)
            {
                closest_idxs[i * k + j] = IDX_MAX_int64_t;
                closest_dists[i * k + j] = DIST_MAX_double;
            }
            min_dist = get_min_dist_double(point_coords + no_dims * i, no_dims, bbox);
            search_splitnode_double_int64_t(root, pa, pidx, no_dims, point_coords + no_dims * i, min_dist,
                             k, distance_upper_bound, eps_fac, mask, &closest_idxs[i * k], &closest_dists[i * k]);
        }
    }
}

/************************************************
Compute mean k-NN Euclidean distances for Statistical Outlier Removal.
Self-queries data points, skips closest neighbor (self), computes
mean Euclidean distance to k remaining neighbors.
Params:
    tree : Tree struct of kd tree
    pa : data points (also used as query points)
    num_points : number of data points
    k : number of neighbors (excluding self)
    mean_dists_out : mean distances output, shape (num_points,)
************************************************/
void sor_mean_dists_double_int64_t(Tree_double_int64_t *tree, double *pa,
                 uint64_t num_points, uint64_t k, double *mean_dists_out)
{
    int8_t no_dims = tree->no_dims;
    double *bbox = tree->bbox;
    uint64_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    uint64_t k_total = k + 1;  /* +1 to include self */
    int64_t local_k_total = (int64_t) k_total;
    Node_double_int64_t *root = (Node_double_int64_t *)tree->root;

    #pragma omp parallel
    {
        uint64_t *local_idx = (uint64_t *)malloc(k_total * sizeof(uint64_t));
        double *local_dist = (double *)malloc(k_total * sizeof(double));

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            double min_dist, sum;
            int64_t count;

            /* Initialize k-NN arrays */
            for (j = 0; j < local_k_total; j++)
            {
                local_idx[j] = IDX_MAX_int64_t;
                local_dist[j] = DIST_MAX_double;
            }

            /* Query k+1 neighbors (includes self) */
            min_dist = get_min_dist_double(pa + no_dims * i, no_dims, bbox);
            search_splitnode_double_int64_t(root, pa, pidx, no_dims,
                             pa + no_dims * i, min_dist,
                             k_total, DIST_MAX_double, (double)1.0, NULL,
                             local_idx, local_dist);

            /* Mean Euclidean distance, skipping closest (self, index 0) */
            sum = 0;
            count = 0;
            for (j = 1; j < local_k_total; j++)
            {
                if (local_dist[j] < DIST_MAX_double)
                {
                    sum += sqrt(local_dist[j]);
                    count++;
                }
            }
            mean_dists_out[i] = (count > 0) ? sum / (double)count : 0;
        }

        free(local_idx);
        free(local_dist);
    }
}

/************************************************
Compute comprehensive point descriptors in a single k-NN pass.
Outputs NUM_DESCRIPTORS (20) features per point per scale:
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
  14:  z_range (z_max - z_min of neighbors)
  15:  z_above (max_z_neighbor - query_z)
  16:  z_below (query_z - min_z_neighbor)
  17:  z_std (height standard deviation of neighbors)
  18:  density (k / bounding_box_volume)
  19:  roughness (|point-to-plane distance|)
k_scales must be sorted ascending.
************************************************/
void compute_descriptors_multiscale_double_int64_t(Tree_double_int64_t *tree, double *pa, double *point_coords,
                 uint64_t num_points, uint64_t k_max,
                 int32_t *k_scales, int32_t num_scales,
                 double distance_upper_bound, double eps, uint8_t *mask,
                 double *descriptors_out)
{
    double eps_fac = 1 / ((1 + eps) * (1 + eps));
    int8_t no_dims = tree->no_dims;
    double *bbox = tree->bbox;
    uint64_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    int64_t local_k_max = (int64_t) k_max;
    Node_double_int64_t *root = (Node_double_int64_t *)tree->root;

    #pragma omp parallel
    {
        uint64_t *local_idx = (uint64_t *)malloc(k_max * sizeof(uint64_t));
        double *local_dist = (double *)malloc(k_max * sizeof(double));

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            double min_dist;
            int32_t s, scale_idx;
            double sum_x, sum_y, sum_z;
            double sum_xx, sum_xy, sum_xz, sum_yy, sum_yz, sum_zz;
            double bb_min_x, bb_min_y, bb_min_z, bb_max_x, bb_max_y, bb_max_z;
            double *pt;
            double px, py, pz;
            double qx = point_coords[no_dims * i];
            double qy = point_coords[no_dims * i + 1];
            double qz = point_coords[no_dims * i + 2];
            int64_t out_base = (int64_t)i * num_scales * NUM_DESCRIPTORS;

            /* Initialize k-NN arrays */
            for (j = 0; j < local_k_max; j++)
            {
                local_idx[j] = IDX_MAX_int64_t;
                local_dist[j] = DIST_MAX_double;
            }

            /* Query k_max neighbors */
            min_dist = get_min_dist_double(point_coords + no_dims * i, no_dims, bbox);
            search_splitnode_double_int64_t(root, pa, pidx, no_dims,
                             point_coords + no_dims * i, min_dist,
                             k_max, distance_upper_bound, eps_fac, mask,
                             local_idx, local_dist);

            /* Initialize accumulators */
            sum_x = sum_y = sum_z = 0;
            sum_xx = sum_xy = sum_xz = sum_yy = sum_yz = sum_zz = 0;
            bb_min_x = bb_min_y = bb_min_z = DIST_MAX_double;
            bb_max_x = bb_max_y = bb_max_z = -DIST_MAX_double;
            scale_idx = 0;

            for (j = 0; j < local_k_max && scale_idx < num_scales; j++)
            {
                if (local_idx[j] == IDX_MAX_int64_t || local_dist[j] >= DIST_MAX_double)
                    break;

                pt = pa + no_dims * local_idx[j];
                px = pt[0]; py = pt[1]; pz = pt[2];

                /* Accumulate sums for covariance */
                sum_x += px; sum_y += py; sum_z += pz;
                sum_xx += px * px; sum_xy += px * py; sum_xz += px * pz;
                sum_yy += py * py; sum_yz += py * pz; sum_zz += pz * pz;

                /* Update bounding box */
                if (px < bb_min_x) bb_min_x = px;
                if (px > bb_max_x) bb_max_x = px;
                if (py < bb_min_y) bb_min_y = py;
                if (py > bb_max_y) bb_max_y = py;
                if (pz < bb_min_z) bb_min_z = pz;
                if (pz > bb_max_z) bb_max_z = pz;

                /* At each scale boundary, compute all descriptors */
                if ((int32_t)(j + 1) == k_scales[scale_idx])
                {
                    int64_t out_off = out_base + (int64_t)scale_idx * NUM_DESCRIPTORS;
                    double evals[3], normal[3];

                    if (j + 1 < 2)
                    {
                        for (s = 0; s < NUM_DESCRIPTORS; s++)
                            descriptors_out[out_off + s] = 0;
                        descriptors_out[out_off + 5] = 1;  /* nz = 1 */
                    }
                    else
                    {
                        double inv_k = 1 / (double)(j + 1);
                        double mx = sum_x * inv_k;
                        double my = sum_y * inv_k;
                        double mz = sum_z * inv_k;

                        double cov_xx = sum_xx * inv_k - mx * mx;
                        double cov_xy = sum_xy * inv_k - mx * my;
                        double cov_xz = sum_xz * inv_k - mx * mz;
                        double cov_yy = sum_yy * inv_k - my * my;
                        double cov_yz = sum_yz * inv_k - my * mz;
                        double cov_zz = sum_zz * inv_k - mz * mz;

                        eigen_symmetric_3x3_double(cov_xx, cov_xy, cov_xz,
                                                     cov_yy, cov_yz, cov_zz,
                                                     evals, normal);

                        /* 0-2: eigenvalues */
                        descriptors_out[out_off + 0] = evals[0];
                        descriptors_out[out_off + 1] = evals[1];
                        descriptors_out[out_off + 2] = evals[2];

                        /* 3-5: normal */
                        descriptors_out[out_off + 3] = normal[0];
                        descriptors_out[out_off + 4] = normal[1];
                        descriptors_out[out_off + 5] = normal[2];

                        /* 6: verticality = 1 - |nz| */
                        descriptors_out[out_off + 6] = 1 - (normal[2] >= 0 ? normal[2] : -normal[2]);

                        /* 7-13: derived eigenvalue features */
                        {
                            double sum_eig = evals[0] + evals[1] + evals[2];
                            double inv_l1 = evals[0] > (double)1e-30 ? 1 / evals[0] : 0;
                            double inv_sum = sum_eig > (double)1e-30 ? 1 / sum_eig : 0;

                            /* 7: linearity = (l1 - l2) / l1 */
                            descriptors_out[out_off + 7] = (evals[0] - evals[1]) * inv_l1;

                            /* 8: planarity = (l2 - l3) / l1 */
                            descriptors_out[out_off + 8] = (evals[1] - evals[2]) * inv_l1;

                            /* 9: sphericity = l3 / l1 */
                            descriptors_out[out_off + 9] = evals[2] * inv_l1;

                            /* 10: omnivariance = (l1 * l2 * l3)^(1/3) */
                            {
                                double prod = evals[0] * evals[1] * evals[2];
                                descriptors_out[out_off + 10] = prod > 0 ? cbrt(prod) : 0;
                            }

                            /* 11: anisotropy = (l1 - l3) / l1 */
                            descriptors_out[out_off + 11] = (evals[0] - evals[2]) * inv_l1;

                            /* 12: eigenentropy = -sum(li/S * ln(li/S)) */
                            {
                                double entropy = 0;
                                if (sum_eig > (double)1e-30)
                                {
                                    int32_t ei;
                                    for (ei = 0; ei < 3; ei++)
                                    {
                                        double p = evals[ei] * inv_sum;
                                        if (p > (double)1e-30)
                                            entropy -= p * log(p);
                                    }
                                }
                                descriptors_out[out_off + 12] = entropy;
                            }

                            /* 13: surface_variation = l3 / (l1 + l2 + l3) */
                            descriptors_out[out_off + 13] = evals[2] * inv_sum;
                        }

                        /* 14: z_range */
                        descriptors_out[out_off + 14] = bb_max_z - bb_min_z;

                        /* 15: z_above = max_z - query_z */
                        descriptors_out[out_off + 15] = bb_max_z - qz;

                        /* 16: z_below = query_z - min_z */
                        descriptors_out[out_off + 16] = qz - bb_min_z;

                        /* 17: z_std = sqrt(cov_zz) */
                        descriptors_out[out_off + 17] = cov_zz > 0 ? sqrt(cov_zz) : 0;

                        /* 18: density = k / bbox_volume */
                        {
                            double vol = (bb_max_x - bb_min_x) * (bb_max_y - bb_min_y) * (bb_max_z - bb_min_z);
                            descriptors_out[out_off + 18] = vol > (double)1e-30 ? (double)(j + 1) / vol : 0;
                        }

                        /* 19: roughness = |dot(query - centroid, normal)| */
                        {
                            double dot = (qx - mx) * normal[0] + (qy - my) * normal[1] + (qz - mz) * normal[2];
                            descriptors_out[out_off + 19] = dot >= 0 ? dot : -dot;
                        }
                    }
                    scale_idx++;
                }
            }

            /* Fill remaining scales with zeros */
            for (s = scale_idx; s < num_scales; s++)
            {
                int64_t out_off = out_base + (int64_t)s * NUM_DESCRIPTORS;
                int32_t f;
                for (f = 0; f < NUM_DESCRIPTORS; f++)
                    descriptors_out[out_off + f] = 0;
                descriptors_out[out_off + 5] = 1;  /* nz = 1 */
            }
        }

        free(local_idx);
        free(local_dist);
    }
}

/************************************************
Radius-based outlier filter.
Queries k_min neighbors for each point. If the k_min-th
neighbor is beyond radius, the point is marked as outlier.
Params:
    tree : Tree struct
    pa : data points (also query points)
    num_points : number of points
    k_min : minimum number of neighbors required within radius
    radius : search radius (Euclidean)
    inlier_mask_out : (num_points,) output, 1 = inlier
    count_out : number of inliers (return, may be NULL)
************************************************/
void radius_filter_double_int64_t(Tree_double_int64_t *tree, double *pa,
                 uint64_t num_points, uint64_t k_min, double radius,
                 uint8_t *inlier_mask_out, uint64_t *count_out)
{
    int8_t no_dims = tree->no_dims;
    double *bbox = tree->bbox;
    uint64_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    /* Query k_min+1 to include self */
    uint64_t k_total = k_min + 1;
    int64_t local_k_total = (int64_t) k_total;
    double radius_sq = radius * radius;
    Node_double_int64_t *root = (Node_double_int64_t *)tree->root;
    uint64_t total_inliers = 0;

    #pragma omp parallel
    {
        uint64_t *local_idx = (uint64_t *)malloc(k_total * sizeof(uint64_t));
        double *local_dist = (double *)malloc(k_total * sizeof(double));
        uint64_t thread_inliers = 0;

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            double min_dist;
            uint64_t count;

            for (j = 0; j < local_k_total; j++)
            {
                local_idx[j] = IDX_MAX_int64_t;
                local_dist[j] = DIST_MAX_double;
            }

            min_dist = get_min_dist_double(pa + no_dims * i, no_dims, bbox);
            search_splitnode_double_int64_t(root, pa, pidx, no_dims,
                             pa + no_dims * i, min_dist,
                             k_total, radius_sq, (double)1.0, NULL,
                             local_idx, local_dist);

            /* Count valid neighbors (excluding self at index 0) */
            count = 0;
            for (j = 1; j < local_k_total; j++)
            {
                if (local_dist[j] < radius_sq && local_idx[j] != IDX_MAX_int64_t)
                    count++;
            }

            if (count >= k_min)
            {
                inlier_mask_out[i] = 1;
                thread_inliers++;
            }
            else
            {
                inlier_mask_out[i] = 0;
            }
        }

        #pragma omp atomic
        total_inliers += thread_inliers;

        free(local_idx);
        free(local_dist);
    }

    if (count_out) *count_out = total_inliers;
}

/************************************************
Estimate normals and curvatures from k-NN neighborhoods.
Lightweight alternative to compute_descriptors when only
normals and curvature are needed.
Params:
    tree : Tree struct
    pa : data points
    point_coords : query points (n * 3)
    num_points : number of query points
    k : number of neighbors
    distance_upper_bound : max distance
    eps : approximation factor
    mask : point validity mask (may be NULL)
    normals_out : (num_points * 3) output normals
    curvatures_out : (num_points,) output curvatures (surface variation)
************************************************/
void estimate_normals_double_int64_t(Tree_double_int64_t *tree, double *pa,
                 double *point_coords, uint64_t num_points, uint64_t k,
                 double distance_upper_bound, double eps, uint8_t *mask,
                 double *normals_out, double *curvatures_out)
{
    double eps_fac = 1 / ((1 + eps) * (1 + eps));
    int8_t no_dims = tree->no_dims;
    double *bbox = tree->bbox;
    uint64_t *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    int64_t local_k = (int64_t) k;
    Node_double_int64_t *root = (Node_double_int64_t *)tree->root;

    #pragma omp parallel
    {
        uint64_t *local_idx = (uint64_t *)malloc(k * sizeof(uint64_t));
        double *local_dist = (double *)malloc(k * sizeof(double));

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            double min_dist;
            double sum_x, sum_y, sum_z;
            double sum_xx, sum_xy, sum_xz, sum_yy, sum_yz, sum_zz;
            double *pt;
            double px, py, pz;
            int64_t valid_count;

            /* Initialize */
            for (j = 0; j < local_k; j++)
            {
                local_idx[j] = IDX_MAX_int64_t;
                local_dist[j] = DIST_MAX_double;
            }

            /* Query k neighbors */
            min_dist = get_min_dist_double(point_coords + no_dims * i, no_dims, bbox);
            search_splitnode_double_int64_t(root, pa, pidx, no_dims,
                             point_coords + no_dims * i, min_dist,
                             k, distance_upper_bound, eps_fac, mask,
                             local_idx, local_dist);

            /* Accumulate covariance sums */
            sum_x = sum_y = sum_z = 0;
            sum_xx = sum_xy = sum_xz = sum_yy = sum_yz = sum_zz = 0;
            valid_count = 0;

            for (j = 0; j < local_k; j++)
            {
                if (local_idx[j] == IDX_MAX_int64_t || local_dist[j] >= DIST_MAX_double)
                    break;
                pt = pa + no_dims * local_idx[j];
                px = pt[0]; py = pt[1]; pz = pt[2];
                sum_x += px; sum_y += py; sum_z += pz;
                sum_xx += px * px; sum_xy += px * py; sum_xz += px * pz;
                sum_yy += py * py; sum_yz += py * pz; sum_zz += pz * pz;
                valid_count++;
            }

            if (valid_count < 2)
            {
                normals_out[3 * i]     = 0;
                normals_out[3 * i + 1] = 0;
                normals_out[3 * i + 2] = 1;
                curvatures_out[i] = 0;
            }
            else
            {
                double inv_k = 1 / (double)valid_count;
                double mx = sum_x * inv_k, my = sum_y * inv_k, mz = sum_z * inv_k;
                double cov_xx = sum_xx * inv_k - mx * mx;
                double cov_xy = sum_xy * inv_k - mx * my;
                double cov_xz = sum_xz * inv_k - mx * mz;
                double cov_yy = sum_yy * inv_k - my * my;
                double cov_yz = sum_yz * inv_k - my * mz;
                double cov_zz = sum_zz * inv_k - mz * mz;

                double evals[3], normal[3];
                eigen_symmetric_3x3_double(cov_xx, cov_xy, cov_xz,
                                             cov_yy, cov_yz, cov_zz,
                                             evals, normal);

                /* Orient normal upward */
                if (normal[2] < 0) { normal[0] = -normal[0]; normal[1] = -normal[1]; normal[2] = -normal[2]; }

                normals_out[3 * i]     = normal[0];
                normals_out[3 * i + 1] = normal[1];
                normals_out[3 * i + 2] = normal[2];

                /* Curvature = surface variation = l3 / (l1 + l2 + l3) */
                {
                    double sum_eig = evals[0] + evals[1] + evals[2];
                    curvatures_out[i] = sum_eig > (double)1e-30 ? evals[2] / sum_eig : 0;
                }
            }
        }

        free(local_idx);
        free(local_dist);
    }
}
