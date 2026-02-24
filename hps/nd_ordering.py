import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# =========================
# Helpers
# =========================

def _is_leaf(npan_dim: np.ndarray) -> bool:
    """Stop splitting when all panel counts are <= 1."""
    return np.all(npan_dim <= 1)


# =========================
# Core split (axis logic inside; 'a' is per-axis constant)
# =========================

def separate_lr(XX_node: np.ndarray,
                box_geom: np.ndarray,
                npan_dim: np.ndarray,
                a: np.ndarray, h_loc =1e-14):
    """
    Decide split axis internally (dim_select = argmax(npan_dim)),
    use fixed per-axis panel size a[dim_select] as h_loc,
    compute part_loc = left boundary + npan_left[axis] * h_loc,
    and classify points into Left / Separator / Right.

    Parameters
    ----------
    XX_node : (M, ndim) points in *this* node
    box_geom: (2, ndim) array [[min...],[max...]] of this node
    npan_dim: (ndim,) int panel counts for this node
    a       : (ndim,) constant characteristic mesh size per axis

    Returns
    -------
    Isep : (K,) local indices for separator points
    left_info  : (Ileft_local,  box_left,  npan_left)
    right_info : (Iright_local, box_right, npan_right)
    info : dict with keys {'dim_select','part_loc','h_loc'}
    """
    npan_dim = np.asarray(npan_dim, dtype=int)
    ndim = npan_dim.shape[0]
    if XX_node.size == 0:
        return (np.array([], dtype=int),
                (np.array([], dtype=int), box_geom.copy(), npan_dim.copy()),
                (np.array([], dtype=int), box_geom.copy(), npan_dim.copy()),
                {"dim_select": None, "part_loc": None, "h_loc": 0.0})

    # choose axis by largest panel count
    dim_select = int(np.argmax(npan_dim))

    # cannot split on this axis
    if npan_dim[dim_select] < 2:
        return (np.array([], dtype=int),
                (np.array([], dtype=int), box_geom.copy(), npan_dim.copy()),
                (np.array([], dtype=int), box_geom.copy(), npan_dim.copy()),
                {"dim_select": dim_select, "part_loc": None, "h_loc": 0.0})

    # panels on chosen axis: split ~ in half (left gets floor)
    pan_split = int(npan_dim[dim_select] // 2)
    if pan_split == 0:
        return (np.array([], dtype=int),
                (np.array([], dtype=int), box_geom.copy(), npan_dim.copy()),
                (np.array([], dtype=int), box_geom.copy(), npan_dim.copy()),
                {"dim_select": dim_select, "part_loc": None, "h_loc": 0.0})

    npan_left  = npan_dim.copy()
    npan_right = npan_dim.copy()
    npan_left[dim_select]  = pan_split
    npan_right[dim_select] = npan_dim[dim_select] - pan_split

    # split location per your rule
    x_min = box_geom[0, dim_select]
    x_max = box_geom[1, dim_select]
    part_loc = x_min + npan_left[dim_select] * float(a[dim_select])
    # clamp just in case (floating accumulations)
    part_loc = max(min(part_loc, x_max), x_min)

    # classify points with separator slab |x - part_loc| <= h_loc
    x = XX_node[:, dim_select]
    Isep   = np.where(np.abs(x - part_loc) <= h_loc)[0]
    Ileft  = np.where(x <  part_loc - h_loc)[0]
    Iright = np.where(x >  part_loc + h_loc)[0]

    # child boxes
    box_left  = box_geom.copy()
    box_right = box_geom.copy()
    box_left[1,  dim_select] = part_loc
    box_right[0, dim_select] = part_loc

    info = {"dim_select": dim_select, "part_loc": part_loc, "h_loc": h_loc}
    return Isep, (Ileft, box_left, npan_left), (Iright, box_right, npan_right), info


# =========================
# BFS (level-by-level) builder with always-on separator visualization
# =========================

def build_bfs_groups_and_splits(XX, box_geom, npan_dim, a, visualize: bool = False):
    """
    Level-order traversal (BFS) using a while-loop (no recursion).

    Returns
    -------
    groups : list[np.ndarray]
        groups[0] = ALL points (root).
        Then, for each level, appends LEFT then RIGHT arrays for every node of that level.
        (Separators are NOT added to groups and are NOT split further.)
    splits : list[dict]
        For plotting/reporting only. Each dict has:
          'dim_select', 'part_loc', 'h_loc', 'Ileft', 'Isep', 'Iright', 'box_geom'
    """
    XX = np.asarray(XX, dtype=float)
    box_geom = np.asarray(box_geom, dtype=float)
    npan_dim = np.asarray(npan_dim, dtype=int)
    a = np.asarray(a, dtype=float)
    N, ndim = XX.shape
    assert box_geom.shape == (2, ndim)
    assert npan_dim.shape == (ndim,)
    assert a.shape == (ndim,), "`a` must be shape (ndim,) — constant per-axis panel size."

    tmp = np.abs(XX[0,0] - XX[:,0])
    hmin= np.min( tmp [ np.where(tmp > 0)[0] ]) * 0.5

    # groups[0] = all points
    groups = [np.arange(N, dtype=int)]
    splits = []

    # queue holds node dicts: indices, box_geom, npan_dim
    q = deque()
    q.append({"indices": groups[0], "box_geom": box_geom.copy(), "npan_dim": npan_dim.copy()})

    while q:
        level_size = len(q)
        next_level_nodes = []

        for _ in range(level_size):
            node = q.popleft()
            indices = node["indices"]
            box     = node["box_geom"]
            npan    = node["npan_dim"]

            if indices.size == 0 or _is_leaf(npan):
                continue  # leaf/empty: nothing to split

            XX_node = XX[indices]

            # split; axis/part_loc/h_loc decided inside
            Isep_l, left_info, right_info, info = separate_lr(
                XX_node, box, npan, a, h_loc = hmin
            )
            dim_select = info["dim_select"]
            part_loc   = info["part_loc"]
            h_loc      = info["h_loc"]

            Ileft_l,  box_left,  npan_left  = left_info
            Iright_l, box_right, npan_right = right_info

            # map local -> global
            Ileft_g  = indices[Ileft_l]
            Iright_g = indices[Iright_l]
            Isep_g   = indices[Isep_l]

            # record split info (for plotting)
            splits.append({
                "dim_select": dim_select,
                "part_loc": part_loc,
                "h_loc": h_loc,
                "Ileft": Ileft_g,
                "Isep": Isep_g,   # always recorded & always visualized
                "Iright": Iright_g,
                "box_geom": box.copy(),
            })

            # append left/right groups only (separator is not added or split)
            groups.append(Ileft_g)
            groups.append(Iright_g)

            # enqueue children (left/right only)
            if Ileft_g.size:
                next_level_nodes.append({"indices": Ileft_g, "box_geom": box_left, "npan_dim": npan_left})
            if Iright_g.size:
                next_level_nodes.append({"indices": Iright_g, "box_geom": box_right, "npan_dim": npan_right})

            # Visualization (BFS order): ALL points faint; highlight left, sep, right
            if visualize and ndim == 2:
                other = (dim_select + 1) % ndim
                x_all = XX[:, dim_select]
                y_all = XX[:, other]
                x = XX[:, dim_select]
                y = XX[:, other]

                plt.figure()
                # background: ALL points
                plt.scatter(x_all, y_all, s=12, c="0.85", alpha=0.35, label="_all")

                # node box outline on (dim_select, other)
                bx0, bx1 = box[0, dim_select], box[1, dim_select]
                by0, by1 = box[0, other],      box[1, other]
                plt.plot([bx0, bx1, bx1, bx0, bx0], [by0, by0, by1, by1, by0], c="k", lw=1)

                # separator: center & band
                if part_loc is not None:
                    plt.axvline(part_loc, linestyle="--", c="k", lw=1)
                    if h_loc > 0:
                        plt.axvline(part_loc - h_loc, linestyle=":", c="k", lw=1)
                        plt.axvline(part_loc + h_loc, linestyle=":", c="k", lw=1)

                # highlight groups (separator ALWAYS plotted; not split further)
                if Ileft_g.size:
                    plt.scatter(x[Ileft_g],  y[Ileft_g],  s=22, marker="o", label="left")
                if Isep_g.size:
                    plt.scatter(x[Isep_g],   y[Isep_g],   s=28, marker="x", label="sep")
                if Iright_g.size:
                    plt.scatter(x[Iright_g], y[Iright_g], s=22, marker="^", label="right")

                plt.xlabel(f"axis {dim_select}")
                plt.ylabel(f"axis {other}")
                t = f"BFS split (axis={dim_select}, part={part_loc:.6g}, h={h_loc:.6g})"
                plt.title(t)
                plt.legend()
                plt.axis('equal')
                plt.show()  # close to proceed

        # push next level’s nodes
        for nd in next_level_nodes:
            q.append(nd)

    return groups, splits


def get_nd_permutation(XX, box_geom, npan_dim, a, visualize=False):
    groups, splits = build_bfs_groups_and_splits(XX, box_geom, npan_dim, a, visualize)

    perm = np.zeros(XX.shape[0], dtype=int)
    offset = 0

    for split in splits:
        tmp = split["Isep"]              
        perm[offset: offset + tmp.shape[0]] = tmp
        offset += tmp.shape[0]

    assert offset == XX.shape[0]
    assert np.unique(perm).shape[0] == XX.shape[0]

    return perm[::-1]
