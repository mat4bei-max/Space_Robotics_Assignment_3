"""ROS-independent, conservative clearance analysis of a planar occupancy grid."""

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class GeometryResult:
    clearance: np.ndarray
    reachable: np.ndarray
    widest_cell: object
    provisional: bool


def world_to_cell(x, y, origin_x, origin_y, yaw, resolution):
    """Return (row, column), accounting for a rotated map origin."""
    dx, dy = x - origin_x, y - origin_y
    c, s = math.cos(yaw), math.sin(yaw)
    return (math.floor((-s * dx + c * dy) / resolution),
            math.floor((c * dx + s * dy) / resolution))


def cell_to_world(row, col, origin_x, origin_y, yaw, resolution):
    x, y = (col + 0.5) * resolution, (row + 0.5) * resolution
    c, s = math.cos(yaw), math.sin(yaw)
    return origin_x + c * x - s * y, origin_y + s * x + c * y


def analyse_grid(grid, resolution, robot_cell, robot_radius=0.5,
                 safety_margin=0.1, free_threshold=25):
    """Find the largest clear disc reachable by a circular robot.

    Unknown/intermediate cells and the map exterior are unavailable. EDT cell
    centre distances are reduced by half a cell diagonal to conservatively
    account for the extent of blocked cells. Four-connected flood fill avoids
    diagonal corner cutting. The result is a grid approximation of reachability.
    """
    grid = np.asarray(grid)
    if grid.ndim != 2 or not grid.size:
        raise ValueError('grid must be a nonempty 2D array')
    if not np.isfinite(grid).all() or np.any((grid < -1) | (grid > 100)):
        raise ValueError('occupancy values must be between -1 and 100')
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError('resolution must be positive and finite')
    if any(not math.isfinite(v) or v < 0 for v in (robot_radius, safety_margin)):
        raise ValueError('radius and margin must be nonnegative and finite')
    if not 0 <= free_threshold < 100:
        raise ValueError('free_threshold must be in [0, 100)')
    free = (grid >= 0) & (grid <= free_threshold)
    padded = np.pad(free, 1, constant_values=False)
    distances = ndimage.distance_transform_edt(padded)[1:-1, 1:-1] * resolution
    clearance = np.maximum(0.0, distances - resolution / math.sqrt(2))
    traversable = free & (clearance >= robot_radius + safety_margin)
    reachable = np.zeros(grid.shape, dtype=bool)
    row, col = robot_cell
    if (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]
            and traversable[row, col]):
        seed = np.zeros(grid.shape, dtype=bool)
        seed[row, col] = True
        reachable = ndimage.binary_propagation(seed, mask=traversable)
    if not reachable.any():
        return GeometryResult(clearance, reachable, None, False)
    widest = tuple(int(v) for v in np.unravel_index(
        np.argmax(np.where(reachable, clearance, -1)), grid.shape))
    # Unknown cells and exterior are not measured walls. Mark ties as provisional.
    observed = np.pad(grid >= 0, 1, constant_values=False)
    unknown_distance = ndimage.distance_transform_edt(observed)[1:-1, 1:-1] * resolution
    provisional = bool(unknown_distance[widest] <= distances[widest] + 1e-9)
    return GeometryResult(clearance, reachable, widest, provisional)


@dataclass
class Passage:
    cell: tuple
    width_m: float
    kind: str
    provisional: bool
    accessible: bool
    length_m: float


def thin_free_space(free):
    """Zhang–Suen thinning; preserve connectivity without an extra dependency."""
    image = np.pad(np.asarray(free, dtype=bool), 1)
    while True:
        changed = False
        for phase in (0, 1):
            p = [image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:],
                 image[2:, 2:], image[2:, 1:-1], image[2:, :-2],
                 image[1:-1, :-2], image[:-2, :-2]]
            neighbours = sum(a.astype(np.uint8) for a in p)
            transitions = sum((~p[i] & p[(i + 1) % 8]).astype(np.uint8)
                              for i in range(8))
            if phase == 0:
                keep = ~(p[0] & p[2] & p[4]) & ~(p[2] & p[4] & p[6])
            else:
                keep = ~(p[0] & p[2] & p[6]) & ~(p[0] & p[4] & p[6])
            remove = (image[1:-1, 1:-1] & (neighbours >= 2) & (neighbours <= 6)
                      & (transitions == 1) & keep)
            if remove.any():
                image[1:-1, 1:-1][remove] = False
                changed = True
        if not changed:
            return image[1:-1, 1:-1]


def _centreline_paths(skeleton):
    cells = set(map(tuple, np.argwhere(skeleton)))
    graph = {}
    for r, c in sorted(cells):
        adjacent = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            other = (r + dr, c + dc)
            if other not in cells:
                continue
            # Prefer existing orthogonal edges over triangular shortcuts.
            if dr and dc and ((r + dr, c) in cells or (r, c + dc) in cells):
                continue
            adjacent.append(other)
        graph[(r, c)] = adjacent
    visited = set()
    paths = []
    starts = [p for p in graph if len(graph[p]) != 2]
    # Remaining degree-two components are cycles; traverse them too.
    for start in starts + list(graph):
        for neighbour in graph[start]:
            edge = frozenset((start, neighbour))
            if edge in visited:
                continue
            visited.add(edge)
            path = [start, neighbour]
            while len(graph[path[-1]]) == 2 and path[-1] != start:
                following = next(p for p in graph[path[-1]] if p != path[-2])
                edge = frozenset((path[-1], following))
                if edge in visited:
                    break
                visited.add(edge)
                path.append(following)
            paths.append(path)
    return paths


def detect_passages(grid, result, resolution, free_threshold=25,
                    narrow_width=3.0, min_length=1.2, context_distance=6.0,
                    bottleneck_ratio=0.75, min_contrast=0.6,
                    merge_distance=2.0, unknown_margin=0.6):
    """Detect sustained narrow sections and relative necks on thinned free space.

    Width is twice conservative clearance, not an exact wall cross-section.
    Endpoint/junction neighbourhoods are excluded. Relative bottlenecks need
    wider context on BOTH sides. Candidates limited by unknown space are flagged.
    Returns the centreline sections supporting selected passages, plus detections.
    The full free-space skeleton is internal, not a navigation route overlay.
    """
    for name, value in [('resolution', resolution), ('narrow_width', narrow_width),
                        ('min_length', min_length), ('context_distance', context_distance),
                        ('min_contrast', min_contrast), ('merge_distance', merge_distance)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive and finite')
    if not math.isfinite(bottleneck_ratio) or not 0 < bottleneck_ratio < 1:
        raise ValueError('bottleneck_ratio must lie between 0 and 1')
    if not math.isfinite(unknown_margin) or unknown_margin < 0:
        raise ValueError('unknown_margin must be nonnegative and finite')
    free = (grid >= 0) & (grid <= free_threshold)
    labels, _ = ndimage.label(free)
    connected = np.unique(labels[result.reachable])
    connected = connected[connected != 0]
    skeleton = thin_free_space(free & np.isin(labels, connected))
    unknown_distance = ndimage.distance_transform_edt(
        np.pad(grid >= 0, 1, constant_values=False))[1:-1, 1:-1] * resolution
    candidates = []
    sections = {}
    for path in _centreline_paths(skeleton):
        points = np.asarray(path)
        if len(points) < 3:
            continue
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))] * resolution
        widths = 2 * result.clearance[points[:, 0], points[:, 1]]
        # Median smoothing rejects single-cell clearance dips.
        smooth = ndimage.median_filter(widths, size=3, mode='nearest')
        interior = (arc >= min_length) & (arc[-1] - arc >= min_length)
        relative = np.zeros(len(path), dtype=bool)
        for i in np.flatnonzero(interior):
            left = smooth[np.searchsorted(arc, arc[i] - context_distance):
                          np.searchsorted(arc, arc[i] - min_length / 2)]
            right = smooth[np.searchsorted(arc, arc[i] + min_length / 2, side='right'):
                           np.searchsorted(arc, arc[i] + context_distance, side='right')]
            if left.size and right.size:
                shoulders = min(float(left.max()), float(right.max()))
                relative[i] = (smooth[i] <= shoulders * bottleneck_ratio
                               and shoulders - smooth[i] >= min_contrast)
        eligible = interior & ((smooth <= narrow_width) | relative) & (smooth > 0)
        runs, count = ndimage.label(eligible)
        for ident in range(1, count + 1):
            indices = np.flatnonzero(runs == ident)
            length = float(arc[indices[-1]] - arc[indices[0]])
            if length < min_length:
                continue
            # Select the middle of the minimum-width plateau, deterministically.
            low = indices[smooth[indices] <= smooth[indices].min() + resolution * 0.1]
            index = int(low[len(low) // 2])
            cell = tuple(int(v) for v in points[index])
            extent = result.clearance[cell] + resolution / math.sqrt(2)
            provisional = bool(unknown_distance[cell] <= extent + unknown_margin)
            candidates.append(Passage(cell, float(widths[index]),
                                      'bottleneck' if relative[index] else 'narrow_passage',
                                      provisional, bool(result.reachable[cell]), length))
            sections[cell] = points[indices]
    selected = []
    for candidate in sorted(candidates, key=lambda p: (p.provisional, p.width_m, p.cell)):
        if all(math.dist(candidate.cell, p.cell) * resolution >= merge_distance for p in selected):
            selected.append(candidate)
    passage_lines = np.zeros(grid.shape, dtype=bool)
    for candidate in selected:
        section = sections[candidate.cell]
        passage_lines[section[:, 0], section[:, 1]] = True
    return passage_lines, selected
