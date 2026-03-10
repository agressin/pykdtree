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
% for ITYPE in ['int32_t', 'int64_t']:
#define PASWAP_${ITYPE}(a,b) { u${ITYPE} tmp = pidx[a]; pidx[a] = pidx[b]; pidx[b] = tmp; }
% endfor

#define IDX_MAX_int32_t UINT32_MAX
#define IDX_MAX_int64_t UINT64_MAX
#define DIST_MAX_float FLT_MAX
#define DIST_MAX_double DBL_MAX

#ifdef _MSC_VER
#define restrict __restrict
#endif

% for DTYPE in ['float', 'double']:
% for ITYPE in ['int32_t', 'int64_t']:

typedef struct
{
    ${DTYPE} cut_val;
    int8_t cut_dim;
    u${ITYPE} start_idx;
    u${ITYPE} n;
    ${DTYPE} cut_bounds_lv;
    ${DTYPE} cut_bounds_hv;
    struct Node_${DTYPE}_${ITYPE} *left_child;
    struct Node_${DTYPE}_${ITYPE} *right_child;
} Node_${DTYPE}_${ITYPE};

typedef struct
{
    ${DTYPE} *bbox;
    int8_t no_dims;
    u${ITYPE} *pidx;
    struct Node_${DTYPE}_${ITYPE} *root;
} Tree_${DTYPE}_${ITYPE};

% endfor
% endfor

% for DTYPE in ['float', 'double']:

${DTYPE} calc_dist_${DTYPE}(${DTYPE} *point1_coord, ${DTYPE} *point2_coord, int8_t no_dims);
${DTYPE} get_cube_offset_${DTYPE}(int8_t dim, ${DTYPE} *point_coord, ${DTYPE} *bbox);
${DTYPE} get_min_dist_${DTYPE}(${DTYPE} *point_coord, int8_t no_dims, ${DTYPE} *bbox);
void eigen_symmetric_3x3_${DTYPE}(${DTYPE} cov_xx, ${DTYPE} cov_xy, ${DTYPE} cov_xz,
                                   ${DTYPE} cov_yy, ${DTYPE} cov_yz, ${DTYPE} cov_zz,
                                   ${DTYPE} *evals, ${DTYPE} *normal);

% for ITYPE in ['int32_t', 'int64_t']:

void insert_point_${DTYPE}_${ITYPE}(u${ITYPE} *closest_idx, ${DTYPE} *closest_dist, u${ITYPE} pidx, ${DTYPE} cur_dist, u${ITYPE} k);
void get_bounding_box_${DTYPE}_${ITYPE}(${DTYPE} *pa, u${ITYPE} *pidx, int8_t no_dims, u${ITYPE} n, ${DTYPE} *bbox);
int partition_${DTYPE}_${ITYPE}(${DTYPE} *pa, u${ITYPE} *pidx, int8_t no_dims, u${ITYPE} start_idx, u${ITYPE} n, ${DTYPE} *bbox, int8_t *cut_dim,
              ${DTYPE} *cut_val, u${ITYPE} *n_lo);
Tree_${DTYPE}_${ITYPE}* construct_tree_${DTYPE}_${ITYPE}(${DTYPE} *pa, int8_t no_dims, u${ITYPE} n, u${ITYPE} bsp);
Node_${DTYPE}_${ITYPE}* construct_subtree_${DTYPE}_${ITYPE}(${DTYPE} *pa, u${ITYPE} *pidx, int8_t no_dims, u${ITYPE} start_idx, u${ITYPE} n, u${ITYPE} bsp, ${DTYPE} *bbox);
Node_${DTYPE}_${ITYPE} * create_node_${DTYPE}_${ITYPE}(u${ITYPE} start_idx, u${ITYPE} n, int is_leaf);
void delete_subtree_${DTYPE}_${ITYPE}(Node_${DTYPE}_${ITYPE} *root);
void delete_tree_${DTYPE}_${ITYPE}(Tree_${DTYPE}_${ITYPE} *tree);
void print_tree_${DTYPE}_${ITYPE}(Node_${DTYPE}_${ITYPE} *root, int level);
void search_leaf_${DTYPE}_${ITYPE}(${DTYPE} *restrict pa, u${ITYPE} *restrict pidx, int8_t no_dims, u${ITYPE} start_idx, u${ITYPE} n, ${DTYPE} *restrict point_coord,
                 u${ITYPE} k, u${ITYPE} *restrict closest_idx, ${DTYPE} *restrict closest_dist);
void search_leaf_${DTYPE}_${ITYPE}_mask(${DTYPE} *restrict pa, u${ITYPE} *restrict pidx, int8_t no_dims, u${ITYPE} start_idx, u${ITYPE} n, ${DTYPE} *restrict point_coord,
                 u${ITYPE} k, uint8_t *restrict mask, u${ITYPE} *restrict closest_idx, ${DTYPE} *restrict closest_dist);
void search_splitnode_${DTYPE}_${ITYPE}(Node_${DTYPE}_${ITYPE} *root, ${DTYPE} *pa, u${ITYPE} *pidx, int8_t no_dims, ${DTYPE} *point_coord,
                      ${DTYPE} min_dist, u${ITYPE} k, ${DTYPE} distance_upper_bound, ${DTYPE} eps_fac, uint8_t *mask, u${ITYPE} *  closest_idx, ${DTYPE} *closest_dist);
void search_tree_${DTYPE}_${ITYPE}(Tree_${DTYPE}_${ITYPE} *tree, ${DTYPE} *pa, ${DTYPE} *point_coords,
                 u${ITYPE} num_points, u${ITYPE} k,  ${DTYPE} distance_upper_bound,
                 ${DTYPE} eps, uint8_t *mask, u${ITYPE} *closest_idxs, ${DTYPE} *closest_dists);
void sor_mean_dists_${DTYPE}_${ITYPE}(Tree_${DTYPE}_${ITYPE} *tree, ${DTYPE} *pa,
                 u${ITYPE} num_points, u${ITYPE} k, ${DTYPE} *mean_dists_out);
void compute_descriptors_multiscale_${DTYPE}_${ITYPE}(Tree_${DTYPE}_${ITYPE} *tree, ${DTYPE} *pa, ${DTYPE} *point_coords,
                 u${ITYPE} num_points, u${ITYPE} k_max,
                 int32_t *k_scales, int32_t num_scales,
                 ${DTYPE} distance_upper_bound, ${DTYPE} eps, uint8_t *mask,
                 ${DTYPE} *descriptors_out);

% endfor
% endfor

% for DTYPE in ['float', 'double']:

/************************************************
Calculate squared cartesian distance between points
Params:
    point1_coord : point 1
    point2_coord : point 2
************************************************/
${DTYPE} calc_dist_${DTYPE}(${DTYPE} *point1_coord, ${DTYPE} *point2_coord, int8_t no_dims)
{
    /* Calculate squared distance */
    ${DTYPE} dist = 0, dim_dist;
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
${DTYPE} get_cube_offset_${DTYPE}(int8_t dim, ${DTYPE} *point_coord, ${DTYPE} *bbox)
{
    ${DTYPE} dim_coord = point_coord[dim];

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
${DTYPE} get_min_dist_${DTYPE}(${DTYPE} *point_coord, int8_t no_dims, ${DTYPE} *bbox)
{
    ${DTYPE} cube_offset = 0, cube_offset_dim;
    int8_t i;

    for (i = 0; i < no_dims; i++)
    {
        cube_offset_dim = get_cube_offset_${DTYPE}(i, point_coord, bbox);
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
void eigen_symmetric_3x3_${DTYPE}(${DTYPE} cov_xx, ${DTYPE} cov_xy, ${DTYPE} cov_xz,
                                   ${DTYPE} cov_yy, ${DTYPE} cov_yz, ${DTYPE} cov_zz,
                                   ${DTYPE} *evals, ${DTYPE} *normal)
{
    ${DTYPE} e1, e2, e3;
    ${DTYPE} p1 = cov_xy * cov_xy + cov_xz * cov_xz + cov_yz * cov_yz;
    ${DTYPE} q = (cov_xx + cov_yy + cov_zz) / 3;
    ${DTYPE} p2 = (cov_xx - q) * (cov_xx - q) + (cov_yy - q) * (cov_yy - q) +
                  (cov_zz - q) * (cov_zz - q) + 2 * p1;
    ${DTYPE} p = sqrt(p2 / 6);

    if (p < (${DTYPE})1e-30)
    {
        /* All eigenvalues are equal */
        e1 = e2 = e3 = q;
    }
    else
    {
        ${DTYPE} inv_p = 1 / p;
        /* B = (1/p)(M - q*I) */
        ${DTYPE} b00 = inv_p * (cov_xx - q);
        ${DTYPE} b01 = inv_p * cov_xy;
        ${DTYPE} b02 = inv_p * cov_xz;
        ${DTYPE} b11 = inv_p * (cov_yy - q);
        ${DTYPE} b12 = inv_p * cov_yz;
        ${DTYPE} b22 = inv_p * (cov_zz - q);

        /* det(B) / 2 */
        ${DTYPE} r = (b00 * (b11 * b22 - b12 * b12)
                     - b01 * (b01 * b22 - b12 * b02)
                     + b02 * (b01 * b12 - b11 * b02)) / 2;

        /* Clamp for numerical stability */
        if (r <= -1) r = -1;
        else if (r >= 1) r = 1;

        ${DTYPE} phi = acos(r) / 3;

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
    ${DTYPE} r0x = cov_xx - e3, r0y = cov_xy,       r0z = cov_xz;
    ${DTYPE} r1x = cov_xy,       r1y = cov_yy - e3, r1z = cov_yz;
    ${DTYPE} r2x = cov_xz,       r2y = cov_yz,       r2z = cov_zz - e3;

    /* Cross products of all row pairs, pick largest */
    ${DTYPE} nx, ny, nz, len_sq, best_len_sq;
    ${DTYPE} cx, cy, cz;

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
    if (best_len_sq > (${DTYPE})1e-30)
    {
        ${DTYPE} inv_len = 1 / sqrt(best_len_sq);
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

% for ITYPE in ['int32_t', 'int64_t']:

/************************************************
Insert point into priority queue
Params:
    closest_idx : index queue
    closest_dist : distance queue
    pidx : permutation index of data points
    cur_dist : distance to point inserted
    k : number of neighbours
************************************************/
void insert_point_${DTYPE}_${ITYPE}(u${ITYPE} *closest_idx, ${DTYPE} *closest_dist, u${ITYPE} pidx, ${DTYPE} cur_dist, u${ITYPE} k)
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
void get_bounding_box_${DTYPE}_${ITYPE}(${DTYPE} *pa, u${ITYPE} *pidx, int8_t no_dims, u${ITYPE} n, ${DTYPE} *bbox)
{
    ${DTYPE} cur;
    int8_t i, j;
    u${ITYPE} bbox_idx, i2;

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
int partition_${DTYPE}_${ITYPE}(${DTYPE} *pa, u${ITYPE} *pidx, int8_t no_dims, u${ITYPE} start_idx, u${ITYPE} n, ${DTYPE} *bbox, int8_t *cut_dim, ${DTYPE} *cut_val, u${ITYPE} *n_lo)
{
    int8_t dim = 0, i;
    u${ITYPE} p, q, i2;
    ${DTYPE} size = 0, min_val, max_val, split, side_len, cur_val;
    u${ITYPE} end_idx = start_idx + n - 1;

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
            PASWAP_${ITYPE}(p, q);
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

        u${ITYPE} j = start_idx;
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
        PASWAP_${ITYPE}(j, start_idx);
        p = start_idx + 1;
    }
    else if (p == end_idx + 1)
    {
        /* No points greater than split.
           Split at highest point instead.
           Minimum 1 point will be in higher box.
        */

        u${ITYPE} j = end_idx;
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
        PASWAP_${ITYPE}(j, end_idx);
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
Node_${DTYPE}_${ITYPE}* construct_subtree_${DTYPE}_${ITYPE}(${DTYPE} *pa, u${ITYPE} *pidx, int8_t no_dims, u${ITYPE} start_idx, u${ITYPE} n, u${ITYPE} bsp, ${DTYPE} *bbox)
{
    /* Create new node */
    int is_leaf = (n <= bsp);
    Node_${DTYPE}_${ITYPE} *root = create_node_${DTYPE}_${ITYPE}(start_idx, n, is_leaf);
    int rval;
    int8_t cut_dim;
    u${ITYPE} n_lo;
    ${DTYPE} cut_val, lv, hv;
    if (is_leaf)
    {
        /* Make leaf node */
        root->cut_dim = -1;
    }
    else
    {
        /* Make split node */
        /* Partition data set and set node info */
        rval = partition_${DTYPE}_${ITYPE}(pa, pidx, no_dims, start_idx, n, bbox, &cut_dim, &cut_val, &n_lo);
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
        root->left_child = (struct Node_${DTYPE}_${ITYPE} *)construct_subtree_${DTYPE}_${ITYPE}(pa, pidx, no_dims, start_idx, n_lo, bsp, bbox);
        bbox[2 * cut_dim + 1] = hv;

        /* Update bounding box before call to higher subset and restore after */
        bbox[2 * cut_dim] = cut_val;
        root->right_child = (struct Node_${DTYPE}_${ITYPE} *)construct_subtree_${DTYPE}_${ITYPE}(pa, pidx, no_dims, start_idx + n_lo, n - n_lo, bsp, bbox);
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
Tree_${DTYPE}_${ITYPE}* construct_tree_${DTYPE}_${ITYPE}(${DTYPE} *pa, int8_t no_dims, u${ITYPE} n, u${ITYPE} bsp)
{
    Tree_${DTYPE}_${ITYPE} *tree = (Tree_${DTYPE}_${ITYPE} *)malloc(sizeof(Tree_${DTYPE}_${ITYPE}));
    u${ITYPE} i;
    u${ITYPE} *pidx;
    ${DTYPE} *bbox;

    tree->no_dims = no_dims;

    /* Initialize permutation array */
    pidx = (u${ITYPE} *)malloc(sizeof(u${ITYPE}) * n);
    for (i = 0; i < n; i++)
    {
        pidx[i] = i;
    }

    bbox = (${DTYPE} *)malloc(2 * sizeof(${DTYPE}) * no_dims);
    get_bounding_box_${DTYPE}_${ITYPE}(pa, pidx, no_dims, n, bbox);
    tree->bbox = bbox;

    /* Construct subtree on full dataset */
    tree->root = (struct Node_${DTYPE}_${ITYPE} *)construct_subtree_${DTYPE}_${ITYPE}(pa, pidx, no_dims, 0, n, bsp, bbox);

    tree->pidx = pidx;
    return tree;
}

/************************************************
Create a tree node.
Params:
    start_idx : index of first data point to use
    n :  number of data points
************************************************/
Node_${DTYPE}_${ITYPE}* create_node_${DTYPE}_${ITYPE}(u${ITYPE} start_idx, u${ITYPE} n, int is_leaf)
{
    Node_${DTYPE}_${ITYPE} *new_node;
    if (is_leaf)
    {
        /*
            Allocate only the part of the struct that will be used in a leaf node.
            This relies on the C99 specification of struct layout conservation and padding and
            that dereferencing is never attempted for the node pointers in a leaf.
        */
        new_node = (Node_${DTYPE}_${ITYPE} *)malloc(sizeof(Node_${DTYPE}_${ITYPE}) - 2 * sizeof(Node_${DTYPE}_${ITYPE} *));
    }
    else
    {
        new_node = (Node_${DTYPE}_${ITYPE} *)malloc(sizeof(Node_${DTYPE}_${ITYPE}));
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
void delete_subtree_${DTYPE}_${ITYPE}(Node_${DTYPE}_${ITYPE} *root)
{
    if (root->cut_dim != -1)
    {
        delete_subtree_${DTYPE}_${ITYPE}((Node_${DTYPE}_${ITYPE} *)root->left_child);
        delete_subtree_${DTYPE}_${ITYPE}((Node_${DTYPE}_${ITYPE} *)root->right_child);
    }
    free(root);
}

/************************************************
Delete tree
Params:
    tree : Tree struct of kd tree
************************************************/
void delete_tree_${DTYPE}_${ITYPE}(Tree_${DTYPE}_${ITYPE} *tree)
{
    delete_subtree_${DTYPE}_${ITYPE}((Node_${DTYPE}_${ITYPE} *)tree->root);
    free(tree->bbox);
    free(tree->pidx);
    free(tree);
}

/************************************************
Print
************************************************/
void print_tree_${DTYPE}_${ITYPE}(Node_${DTYPE}_${ITYPE} *root, int level)
{
    int i;
    for (i = 0; i < level; i++)
    {
        printf(" ");
    }
    printf("(cut_val: %f, cut_dim: %i)\n", root->cut_val, root->cut_dim);
    if (root->cut_dim != -1)
        print_tree_${DTYPE}_${ITYPE}((Node_${DTYPE}_${ITYPE} *)root->left_child, level + 1);
    if (root->cut_dim != -1)
        print_tree_${DTYPE}_${ITYPE}((Node_${DTYPE}_${ITYPE} *)root->right_child, level + 1);
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
void search_leaf_${DTYPE}_${ITYPE}(${DTYPE} *restrict pa, u${ITYPE} *restrict pidx, int8_t no_dims, u${ITYPE} start_idx, u${ITYPE} n, ${DTYPE} *restrict point_coord,
                 u${ITYPE} k, u${ITYPE} *restrict closest_idx, ${DTYPE} *restrict closest_dist)
{
    ${DTYPE} cur_dist;
    u${ITYPE} i;
    /* Loop through all points in leaf */
    for (i = 0; i < n; i++)
    {
        /* Get distance to query point */
        cur_dist = calc_dist_${DTYPE}(&PA(start_idx + i, 0), point_coord, no_dims);
        /* Update closest info if new point is closest so far*/
        if (cur_dist < closest_dist[k - 1])
        {
            insert_point_${DTYPE}_${ITYPE}(closest_idx, closest_dist, pidx[start_idx + i], cur_dist, k);
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
void search_leaf_${DTYPE}_${ITYPE}_mask(${DTYPE} *restrict pa, u${ITYPE} *restrict pidx, int8_t no_dims, u${ITYPE} start_idx, u${ITYPE} n, ${DTYPE} *restrict point_coord,
                               u${ITYPE} k, uint8_t *mask, u${ITYPE} *restrict closest_idx, ${DTYPE} *restrict closest_dist)
{
    ${DTYPE} cur_dist;
    u${ITYPE} i;
    /* Loop through all points in leaf */
    for (i = 0; i < n; i++)
    {
        /* Is this point masked out? */
        if (mask[pidx[start_idx + i]])
        {
            continue;
        }
        /* Get distance to query point */
        cur_dist = calc_dist_${DTYPE}(&PA(start_idx + i, 0), point_coord, no_dims);
        /* Update closest info if new point is closest so far*/
        if (cur_dist < closest_dist[k - 1])
        {
            insert_point_${DTYPE}_${ITYPE}(closest_idx, closest_dist, pidx[start_idx + i], cur_dist, k);
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
void search_splitnode_${DTYPE}_${ITYPE}(Node_${DTYPE}_${ITYPE} *root, ${DTYPE} *pa, u${ITYPE} *pidx, int8_t no_dims, ${DTYPE} *point_coord, 
                      ${DTYPE} min_dist, u${ITYPE} k, ${DTYPE} distance_upper_bound, ${DTYPE} eps_fac, uint8_t *mask,
                      u${ITYPE} *closest_idx, ${DTYPE} *closest_dist)
{
    int8_t dim;
    ${DTYPE} dist_left, dist_right;
    ${DTYPE} new_offset;
    ${DTYPE} box_diff;

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
            search_leaf_${DTYPE}_${ITYPE}_mask(pa, pidx, no_dims, root->start_idx, root->n, point_coord, k, mask, closest_idx, closest_dist);
        }
        else
        {
            search_leaf_${DTYPE}_${ITYPE}(pa, pidx, no_dims, root->start_idx, root->n, point_coord, k, closest_idx, closest_dist);
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
            search_splitnode_${DTYPE}_${ITYPE}((Node_${DTYPE}_${ITYPE} *)root->left_child, pa, pidx, no_dims, point_coord, dist_left, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
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
            search_splitnode_${DTYPE}_${ITYPE}((Node_${DTYPE}_${ITYPE} *)root->right_child, pa, pidx, no_dims, point_coord, dist_right, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
        }
    }
    else
    {
        /* Right of cutting plane */
        dist_right = min_dist;
        if (dist_right < closest_dist[k - 1] * eps_fac)
        {
            /* Search right subtree if minimum distance is below limit*/
            search_splitnode_${DTYPE}_${ITYPE}((Node_${DTYPE}_${ITYPE} *)root->right_child, pa, pidx, no_dims, point_coord, dist_right, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
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
            search_splitnode_${DTYPE}_${ITYPE}((Node_${DTYPE}_${ITYPE} *)root->left_child, pa, pidx, no_dims, point_coord, dist_left, k, distance_upper_bound, eps_fac, mask, closest_idx, closest_dist);
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
void search_tree_${DTYPE}_${ITYPE}(Tree_${DTYPE}_${ITYPE} *tree, ${DTYPE} *pa, ${DTYPE} *point_coords,
                 u${ITYPE} num_points, u${ITYPE} k, ${DTYPE} distance_upper_bound,
                 ${DTYPE} eps, uint8_t *mask, u${ITYPE} *closest_idxs, ${DTYPE} *closest_dists)
{
    ${DTYPE} min_dist;
    ${DTYPE} eps_fac = 1 / ((1 + eps) * (1 + eps));
    int8_t no_dims = tree->no_dims;
    ${DTYPE} *bbox = tree->bbox;
    u${ITYPE} *pidx = tree->pidx;
    /* use 64-bit ints for indexing to avoid overflow, use signed ints to support all Openmp implementations */
    int64_t i = 0;
    int64_t j = 0;
    int64_t local_num_points = (int64_t) num_points;
    Node_${DTYPE}_${ITYPE} *root = (Node_${DTYPE}_${ITYPE} *)tree->root;

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
                closest_idxs[i * k + j] = IDX_MAX_${ITYPE};
                closest_dists[i * k + j] = DIST_MAX_${DTYPE};
            }
            min_dist = get_min_dist_${DTYPE}(point_coords + no_dims * i, no_dims, bbox);
            search_splitnode_${DTYPE}_${ITYPE}(root, pa, pidx, no_dims, point_coords + no_dims * i, min_dist,
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
void sor_mean_dists_${DTYPE}_${ITYPE}(Tree_${DTYPE}_${ITYPE} *tree, ${DTYPE} *pa,
                 u${ITYPE} num_points, u${ITYPE} k, ${DTYPE} *mean_dists_out)
{
    int8_t no_dims = tree->no_dims;
    ${DTYPE} *bbox = tree->bbox;
    u${ITYPE} *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    u${ITYPE} k_total = k + 1;  /* +1 to include self */
    int64_t local_k_total = (int64_t) k_total;
    Node_${DTYPE}_${ITYPE} *root = (Node_${DTYPE}_${ITYPE} *)tree->root;

    #pragma omp parallel
    {
        u${ITYPE} *local_idx = (u${ITYPE} *)malloc(k_total * sizeof(u${ITYPE}));
        ${DTYPE} *local_dist = (${DTYPE} *)malloc(k_total * sizeof(${DTYPE}));

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            ${DTYPE} min_dist, sum;
            int64_t count;

            /* Initialize k-NN arrays */
            for (j = 0; j < local_k_total; j++)
            {
                local_idx[j] = IDX_MAX_${ITYPE};
                local_dist[j] = DIST_MAX_${DTYPE};
            }

            /* Query k+1 neighbors (includes self) */
            min_dist = get_min_dist_${DTYPE}(pa + no_dims * i, no_dims, bbox);
            search_splitnode_${DTYPE}_${ITYPE}(root, pa, pidx, no_dims,
                             pa + no_dims * i, min_dist,
                             k_total, DIST_MAX_${DTYPE}, (${DTYPE})1.0, NULL,
                             local_idx, local_dist);

            /* Mean Euclidean distance, skipping closest (self, index 0) */
            sum = 0;
            count = 0;
            for (j = 1; j < local_k_total; j++)
            {
                if (local_dist[j] < DIST_MAX_${DTYPE})
                {
                    sum += sqrt(local_dist[j]);
                    count++;
                }
            }
            mean_dists_out[i] = (count > 0) ? sum / (${DTYPE})count : 0;
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
void compute_descriptors_multiscale_${DTYPE}_${ITYPE}(Tree_${DTYPE}_${ITYPE} *tree, ${DTYPE} *pa, ${DTYPE} *point_coords,
                 u${ITYPE} num_points, u${ITYPE} k_max,
                 int32_t *k_scales, int32_t num_scales,
                 ${DTYPE} distance_upper_bound, ${DTYPE} eps, uint8_t *mask,
                 ${DTYPE} *descriptors_out)
{
    ${DTYPE} eps_fac = 1 / ((1 + eps) * (1 + eps));
    int8_t no_dims = tree->no_dims;
    ${DTYPE} *bbox = tree->bbox;
    u${ITYPE} *pidx = tree->pidx;
    int64_t i, j;
    int64_t local_num_points = (int64_t) num_points;
    int64_t local_k_max = (int64_t) k_max;
    Node_${DTYPE}_${ITYPE} *root = (Node_${DTYPE}_${ITYPE} *)tree->root;

    #pragma omp parallel
    {
        u${ITYPE} *local_idx = (u${ITYPE} *)malloc(k_max * sizeof(u${ITYPE}));
        ${DTYPE} *local_dist = (${DTYPE} *)malloc(k_max * sizeof(${DTYPE}));

        #pragma omp for private(i, j) schedule(static, 100) nowait
        for (i = 0; i < local_num_points; i++)
        {
            ${DTYPE} min_dist;
            int32_t s, scale_idx;
            ${DTYPE} sum_x, sum_y, sum_z;
            ${DTYPE} sum_xx, sum_xy, sum_xz, sum_yy, sum_yz, sum_zz;
            ${DTYPE} bb_min_x, bb_min_y, bb_min_z, bb_max_x, bb_max_y, bb_max_z;
            ${DTYPE} *pt;
            ${DTYPE} px, py, pz;
            ${DTYPE} qx = point_coords[no_dims * i];
            ${DTYPE} qy = point_coords[no_dims * i + 1];
            ${DTYPE} qz = point_coords[no_dims * i + 2];
            int64_t out_base = (int64_t)i * num_scales * NUM_DESCRIPTORS;

            /* Initialize k-NN arrays */
            for (j = 0; j < local_k_max; j++)
            {
                local_idx[j] = IDX_MAX_${ITYPE};
                local_dist[j] = DIST_MAX_${DTYPE};
            }

            /* Query k_max neighbors */
            min_dist = get_min_dist_${DTYPE}(point_coords + no_dims * i, no_dims, bbox);
            search_splitnode_${DTYPE}_${ITYPE}(root, pa, pidx, no_dims,
                             point_coords + no_dims * i, min_dist,
                             k_max, distance_upper_bound, eps_fac, mask,
                             local_idx, local_dist);

            /* Initialize accumulators */
            sum_x = sum_y = sum_z = 0;
            sum_xx = sum_xy = sum_xz = sum_yy = sum_yz = sum_zz = 0;
            bb_min_x = bb_min_y = bb_min_z = DIST_MAX_${DTYPE};
            bb_max_x = bb_max_y = bb_max_z = -DIST_MAX_${DTYPE};
            scale_idx = 0;

            for (j = 0; j < local_k_max && scale_idx < num_scales; j++)
            {
                if (local_idx[j] == IDX_MAX_${ITYPE} || local_dist[j] >= DIST_MAX_${DTYPE})
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
                    ${DTYPE} evals[3], normal[3];

                    if (j + 1 < 2)
                    {
                        for (s = 0; s < NUM_DESCRIPTORS; s++)
                            descriptors_out[out_off + s] = 0;
                        descriptors_out[out_off + 5] = 1;  /* nz = 1 */
                    }
                    else
                    {
                        ${DTYPE} inv_k = 1 / (${DTYPE})(j + 1);
                        ${DTYPE} mx = sum_x * inv_k;
                        ${DTYPE} my = sum_y * inv_k;
                        ${DTYPE} mz = sum_z * inv_k;

                        ${DTYPE} cov_xx = sum_xx * inv_k - mx * mx;
                        ${DTYPE} cov_xy = sum_xy * inv_k - mx * my;
                        ${DTYPE} cov_xz = sum_xz * inv_k - mx * mz;
                        ${DTYPE} cov_yy = sum_yy * inv_k - my * my;
                        ${DTYPE} cov_yz = sum_yz * inv_k - my * mz;
                        ${DTYPE} cov_zz = sum_zz * inv_k - mz * mz;

                        eigen_symmetric_3x3_${DTYPE}(cov_xx, cov_xy, cov_xz,
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
                            ${DTYPE} sum_eig = evals[0] + evals[1] + evals[2];
                            ${DTYPE} inv_l1 = evals[0] > (${DTYPE})1e-30 ? 1 / evals[0] : 0;
                            ${DTYPE} inv_sum = sum_eig > (${DTYPE})1e-30 ? 1 / sum_eig : 0;

                            /* 7: linearity = (l1 - l2) / l1 */
                            descriptors_out[out_off + 7] = (evals[0] - evals[1]) * inv_l1;

                            /* 8: planarity = (l2 - l3) / l1 */
                            descriptors_out[out_off + 8] = (evals[1] - evals[2]) * inv_l1;

                            /* 9: sphericity = l3 / l1 */
                            descriptors_out[out_off + 9] = evals[2] * inv_l1;

                            /* 10: omnivariance = (l1 * l2 * l3)^(1/3) */
                            {
                                ${DTYPE} prod = evals[0] * evals[1] * evals[2];
                                descriptors_out[out_off + 10] = prod > 0 ? cbrt(prod) : 0;
                            }

                            /* 11: anisotropy = (l1 - l3) / l1 */
                            descriptors_out[out_off + 11] = (evals[0] - evals[2]) * inv_l1;

                            /* 12: eigenentropy = -sum(li/S * ln(li/S)) */
                            {
                                ${DTYPE} entropy = 0;
                                if (sum_eig > (${DTYPE})1e-30)
                                {
                                    int32_t ei;
                                    for (ei = 0; ei < 3; ei++)
                                    {
                                        ${DTYPE} p = evals[ei] * inv_sum;
                                        if (p > (${DTYPE})1e-30)
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
                            ${DTYPE} vol = (bb_max_x - bb_min_x) * (bb_max_y - bb_min_y) * (bb_max_z - bb_min_z);
                            descriptors_out[out_off + 18] = vol > (${DTYPE})1e-30 ? (${DTYPE})(j + 1) / vol : 0;
                        }

                        /* 19: roughness = |dot(query - centroid, normal)| */
                        {
                            ${DTYPE} dot = (qx - mx) * normal[0] + (qy - my) * normal[1] + (qz - mz) * normal[2];
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
% endfor
% endfor
