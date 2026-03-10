#!/usr/bin/env python3
"""Benchmark pykdtree spatial operations vs numpy/scipy equivalents."""

import time
import numpy as np

def timer(label, func, *args, **kwargs):
    """Run func, print timing, return result."""
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    dt = time.perf_counter() - t0
    print(f"  {label:40s} {dt*1000:8.1f} ms")
    return result, dt

def bench_morton(n):
    print(f"\n=== Morton encode/decode  (N={n:,}) ===")
    from pykdtree.spatial import morton_encode, morton_decode

    x = np.random.randint(0, 2**21, n, dtype=np.uint64)
    y = np.random.randint(0, 2**21, n, dtype=np.uint64)
    z = np.random.randint(0, 2**21, n, dtype=np.uint64)

    # --- pykdtree C ---
    codes_c, dt_c = timer("pykdtree.spatial.morton_encode", morton_encode, x, y, z)

    # --- numpy reference (same algorithm) ---
    def _np_spread(v):
        v = v.astype(np.uint64) & np.uint64(0x1FFFFF)
        v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        v = (v | (v << np.uint64(8)))  & np.uint64(0x100F00F00F00F00F)
        v = (v | (v << np.uint64(4)))  & np.uint64(0x10C30C30C30C30C3)
        v = (v | (v << np.uint64(2)))  & np.uint64(0x1249249249249249)
        return v

    def np_morton(x, y, z):
        return _np_spread(x) | (_np_spread(y) << np.uint64(1)) | (_np_spread(z) << np.uint64(2))

    codes_np, dt_np = timer("numpy reference", np_morton, x, y, z)

    assert np.array_equal(codes_c, codes_np), "Morton mismatch!"
    print(f"  Speedup: {dt_np/dt_c:.1f}x")

    # Decode
    (xd, yd, zd), dt_dec = timer("pykdtree.spatial.morton_decode", morton_decode, codes_c)
    assert np.array_equal(xd, x) and np.array_equal(yd, y) and np.array_equal(zd, z), "Decode mismatch!"


def bench_hilbert(n):
    print(f"\n=== Hilbert encode  (N={n:,}) ===")
    from pykdtree.spatial import hilbert_encode

    x = np.random.randint(0, 2**10, n, dtype=np.uint64)
    y = np.random.randint(0, 2**10, n, dtype=np.uint64)
    z = np.random.randint(0, 2**10, n, dtype=np.uint64)

    timer("pykdtree.spatial.hilbert_encode", hilbert_encode, x, y, z, 10)


def bench_voxel_downsample(n):
    print(f"\n=== Voxel downsample  (N={n:,}) ===")
    from pykdtree.spatial import voxel_downsample

    points = np.random.rand(n, 3).astype(np.float32) * 100
    voxel_size = 0.5

    sel_inv, dt_c = timer("pykdtree.spatial.voxel_downsample", voxel_downsample, points, voxel_size)
    sel, inv = sel_inv
    n_unique = len(sel)

    # --- numpy reference ---
    def np_voxel(pts, vs):
        voxel_coords = np.floor(pts / vs).astype(np.int64)
        _, idx, inv = np.unique(voxel_coords, axis=0, return_index=True, return_inverse=True)
        return idx, inv

    (sel_np, inv_np), dt_np = timer("numpy unique", np_voxel, points, voxel_size)

    print(f"  Unique voxels (C): {n_unique:,}  (numpy): {len(sel_np):,}")
    print(f"  Speedup: {dt_np/dt_c:.1f}x")


def bench_assign_tiles(n):
    print(f"\n=== Assign tiles  (N={n:,}) ===")
    from pykdtree.spatial import assign_tiles

    points = np.random.rand(n, 3).astype(np.float32) * 10000
    tile_size = 100.0

    (tx, ty), dt_c = timer("pykdtree.spatial.assign_tiles", assign_tiles, points, tile_size)

    def np_assign(pts, ts):
        return np.floor(pts[:, 0] / ts).astype(np.int32), np.floor(pts[:, 1] / ts).astype(np.int32)

    (tx_np, ty_np), dt_np = timer("numpy floor division", np_assign, points, tile_size)

    assert np.array_equal(tx, tx_np) and np.array_equal(ty, ty_np), "Tile assign mismatch!"
    print(f"  Speedup: {dt_np/dt_c:.1f}x")


def bench_scatter_minmax(n):
    print(f"\n=== Scatter min/max  (N={n:,}) ===")
    from pykdtree.spatial import scatter_minmax

    points_xy = np.random.rand(n, 2).astype(np.float32) * 100
    values = np.random.rand(n).astype(np.float32)
    grid_h, grid_w = 200, 200
    resolution = 0.5
    origin = (0.0, 0.0)

    (min_g, max_g, cnt_g), dt_c = timer("pykdtree.spatial.scatter_minmax",
                                         scatter_minmax, points_xy, values,
                                         (grid_h, grid_w), resolution, origin)

    # --- numpy reference ---
    def np_scatter(xy, vals, gh, gw, res, orig):
        col = np.floor((xy[:, 0] - orig[0]) / res).astype(np.int32)
        row = np.floor((xy[:, 1] - orig[1]) / res).astype(np.int32)
        valid = (row >= 0) & (row < gh) & (col >= 0) & (col < gw)
        r, c, v = row[valid], col[valid], vals[valid]
        min_grid = np.full((gh, gw), np.inf, dtype=np.float32)
        max_grid = np.full((gh, gw), -np.inf, dtype=np.float32)
        cnt = np.zeros((gh, gw), dtype=np.uint32)
        np.minimum.at(min_grid, (r, c), v)
        np.maximum.at(max_grid, (r, c), v)
        np.add.at(cnt, (r, c), 1)
        return min_grid, max_grid, cnt

    (min_np, max_np, cnt_np), dt_np = timer("numpy minimum.at/maximum.at",
                                             np_scatter, points_xy, values,
                                             grid_h, grid_w, resolution, origin)

    assert np.array_equal(cnt_g, cnt_np), "Scatter count mismatch!"
    print(f"  Speedup: {dt_np/dt_c:.1f}x")


def bench_grid_sample(n):
    print(f"\n=== Grid sample nearest  (N={n:,}) ===")
    from pykdtree.spatial import grid_sample_nearest

    grid_h, grid_w = 500, 500
    grid = np.random.rand(grid_h, grid_w).astype(np.float32)
    points_xy = np.random.rand(n, 2).astype(np.float32) * 100
    resolution = 0.2
    origin = (0.0, 0.0)

    vals_c, dt_c = timer("pykdtree.spatial.grid_sample_nearest",
                               grid_sample_nearest, grid, points_xy, resolution, origin)

    def np_sample(g, xy, res, orig):
        col = np.clip(np.floor((xy[:, 0] - orig[0]) / res).astype(np.int32), 0, g.shape[1]-1)
        row = np.clip(np.floor((xy[:, 1] - orig[1]) / res).astype(np.int32), 0, g.shape[0]-1)
        return g[row, col]

    vals_np, dt_np = timer("numpy indexing", np_sample, grid, points_xy, resolution, origin)
    print(f"  Speedup: {dt_np/dt_c:.1f}x")


def bench_filter_bbox(n):
    print(f"\n=== Filter bbox  (N={n:,}) ===")
    from pykdtree.spatial import filter_bbox

    points = np.random.rand(n, 3).astype(np.float32) * 100

    (mask_c, count_c), dt_c = timer("pykdtree.spatial.filter_bbox",
                                     filter_bbox, points, 20, 20, 20, 80, 80, 80)

    def np_filter(pts, x0, y0, z0, x1, y1, z1):
        m = ((pts[:, 0] >= x0) & (pts[:, 0] < x1) &
             (pts[:, 1] >= y0) & (pts[:, 1] < y1) &
             (pts[:, 2] >= z0) & (pts[:, 2] < z1))
        return m, m.sum()

    (mask_np, count_np), dt_np = timer("numpy boolean ops",
                                        np_filter, points, 20, 20, 20, 80, 80, 80)

    assert count_c == count_np, f"Filter count mismatch: {count_c} vs {count_np}"
    print(f"  Speedup: {dt_np/dt_c:.1f}x")


def bench_merge_tiles(n_tiles, pts_per_tile):
    n_total = n_tiles * pts_per_tile
    print(f"\n=== Merge tiles  ({n_tiles} tiles × {pts_per_tile:,} pts = {n_total:,}) ===")
    from pykdtree.spatial import merge_tiles

    # Simulate tiles as separate x, y, z columns (like Parquet output)
    tiles_cols = []   # [(x, y, z), ...]
    tiles_stacked = [] # [(n, 3), ...] for numpy reference
    for t in range(n_tiles):
        x = np.random.rand(pts_per_tile).astype(np.float32) * 1000
        y = np.random.rand(pts_per_tile).astype(np.float32) * 1000
        z = np.random.rand(pts_per_tile).astype(np.float32) * 50
        tiles_cols.append((x, y, z))
        tiles_stacked.append(np.column_stack([x, y, z]))

    # Offsets: tile_origin - scene_origin (each tile at different location)
    offsets = np.zeros((n_tiles, 3), dtype=np.float32)
    for t in range(n_tiles):
        offsets[t, 0] = t * 1000.0  # tiles side by side on X

    # ROI: select ~60% of points (center region)
    roi = (200.0, 200.0, (n_tiles - 1) * 1000 + 800.0, 800.0)

    # --- pykdtree C (separate columns) ---
    (xyz_c, masks_c), dt_c = timer("pykdtree merge_tiles (x,y,z cols)",
                                    merge_tiles, tiles_cols, offsets, roi)

    # --- pykdtree C (stacked arrays, backwards compat) ---
    (xyz_c2, _), dt_c2 = timer("pykdtree merge_tiles (n,3 arrays)",
                                merge_tiles, tiles_stacked, offsets, roi)

    # --- numpy reference (what projax does) ---
    def np_merge(tile_list, offs, roi_bounds):
        results = []
        for t in range(len(tile_list)):
            xyz = tile_list[t]
            off = offs[t]
            local_min_x = roi_bounds[0] - off[0]
            local_max_x = roi_bounds[2] - off[0]
            local_min_y = roi_bounds[1] - off[1]
            local_max_y = roi_bounds[3] - off[1]
            mask = ((xyz[:, 0] >= local_min_x) & (xyz[:, 0] < local_max_x) &
                    (xyz[:, 1] >= local_min_y) & (xyz[:, 1] < local_max_y))
            results.append(xyz[mask] + off)
        return np.concatenate(results, axis=0)

    xyz_np, dt_np = timer("numpy mask+transform+concat", np_merge, tiles_stacked, offsets, roi)

    assert xyz_c.shape == xyz_np.shape, f"Shape mismatch: {xyz_c.shape} vs {xyz_np.shape}"
    assert np.allclose(xyz_c, xyz_np, atol=1e-5), "Value mismatch!"
    print(f"  Output points: {xyz_c.shape[0]:,} / {n_total:,}")
    print(f"  Speedup vs numpy: {dt_np/dt_c:.1f}x")

    # --- Benchmark apply_masks for attributes ---
    from pykdtree.spatial import apply_masks

    # Simulate 3 attributes per tile
    attr_names = ["intensity", "classification", "gps_time"]
    attr_dtypes = [np.float32, np.uint8, np.float64]
    tile_attrs = []
    for t in range(n_tiles):
        tile_attrs.append({
            "intensity": np.random.rand(pts_per_tile).astype(np.float32),
            "classification": np.random.randint(0, 20, pts_per_tile, dtype=np.uint8),
            "gps_time": np.random.rand(pts_per_tile).astype(np.float64) * 1e9,
        })

    # Get masks from the C call above
    _, masks_for_attrs = merge_tiles(tiles_cols, offsets, roi)

    # --- pykdtree apply_masks ---
    def c_apply_all(t_attrs, m):
        result = {}
        for key in attr_names:
            result[key] = apply_masks([t_attrs[i][key] for i in range(len(m))], m)
        return result

    attrs_c, dt_c_attr = timer("pykdtree apply_masks (3 attrs)", c_apply_all, tile_attrs, masks_for_attrs)

    # --- numpy reference ---
    def np_apply_all(t_attrs, m):
        result = {}
        for key in attr_names:
            result[key] = np.concatenate([t_attrs[i][key][m[i]] for i in range(len(m))])
        return result

    attrs_np, dt_np_attr = timer("numpy mask+concat (3 attrs)", np_apply_all, tile_attrs, masks_for_attrs)

    for key in attr_names:
        assert np.array_equal(attrs_c[key], attrs_np[key]), f"Attr {key} mismatch!"
    print(f"  Attr speedup: {dt_np_attr/dt_c_attr:.1f}x")


def bench_kdtree_new_methods(n):
    print(f"\n=== KDTree: radius_filter + estimate_normals  (N={n:,}) ===")
    from pykdtree.kdtree import KDTree

    points = np.random.rand(n, 3).astype(np.float32) * 10

    tree = KDTree(points)

    timer("KDTree.radius_filter(k_min=5, r=0.5)", tree.radius_filter, 5, 0.5)
    timer("KDTree.estimate_normals(k=20)", tree.estimate_normals, points, 20)
    timer("KDTree.compute_descriptors(k=20)", tree.compute_descriptors, points, 20)


if __name__ == "__main__":
    np.random.seed(42)

    N = 2_000_000
    print(f"{'='*60}")
    print(f"  pykdtree spatial ops benchmark  —  N = {N:,}")
    print(f"{'='*60}")

    bench_morton(N)
    bench_hilbert(N)
    bench_voxel_downsample(N)
    bench_assign_tiles(N)
    bench_scatter_minmax(N)
    bench_grid_sample(N)
    bench_filter_bbox(N)
    bench_merge_tiles(20, 100_000)   # 20 tiles × 100K = 2M points
    bench_merge_tiles(100, 100_000)  # 100 tiles × 100K = 10M points
    bench_kdtree_new_methods(N)

    print(f"\n{'='*60}")
    print("  Done!")
    print(f"{'='*60}")
