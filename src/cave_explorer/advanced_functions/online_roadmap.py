"""Incremental PRM, extending Assignment 2 Tasks 7–10; independent of ROS."""

import heapq
import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from advanced_functions.cave_geometry import analyse_grid, cell_to_world, world_to_cell


@dataclass
class Settings:
    mode: str = 'distance_guided'
    robot_radius: float = 0.5
    safety_margin: float = 0.1
    free_threshold: int = 25
    node_spacing: float = 1.2
    connection_radius: float = 4.0
    neighbourhood_radius: float = 1.5
    acceptance_fraction: float = 0.7
    uniform_fraction: float = 0.2
    samples_per_update: int = 20
    attempts_per_update: int = 300
    max_nodes: int = 600
    seed: int = 7
    coverage_radius: float = 2.0
    coverage_samples: int = 128
    repair_attempts: int = 30
    isolation_patience: int = 5
    reserve_fraction: float = 0.1
    visit_radius: float = 0.75

    def __post_init__(self):
        if self.mode not in ('baseline', 'uniform', 'distance_guided'):
            raise ValueError('mode must be baseline, uniform or distance_guided')
        for key in ('node_spacing', 'connection_radius', 'neighbourhood_radius', 'coverage_radius', 'visit_radius'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f'{key} must be positive and finite')
        for key in ('robot_radius', 'safety_margin'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(f'{key} must be nonnegative and finite')
        for key in ('acceptance_fraction', 'uniform_fraction', 'reserve_fraction'):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f'{key} must be in [0, 1]')
        for key in ('samples_per_update', 'attempts_per_update', 'max_nodes', 'coverage_samples', 'repair_attempts', 'isolation_patience'):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if not 0 <= self.free_threshold < 100:
            raise ValueError('free_threshold must be in [0, 100)')


class OnlineRoadmap:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.rng = np.random.default_rng(self.settings.seed)
        self.nodes = {}  # Stable ID -> world (x, y).
        self.sources = {}
        self.edges = {}  # Sorted ID pair -> length in metres.
        self.next_id = 0
        self.grid = None
        self.groups = {}
        self.robot = None
        self.trajectory = []
        self.seen_cells = set()
        self.isolation_age = {}
        self.cooldown = {}
        self.cycle = 0
        self.budget_rejections = 0
        self.replacements = 0
        self.retired_isolates = 0

    def lower_clearance(self, point):
        """Conservative clearance at a continuous world point, not just a cell centre."""
        r, c = world_to_cell(*point, *self.origin)
        if not (0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]):
            return -math.inf
        centre = cell_to_world(r, c, *self.origin)
        return float(self.geometry.clearance[r, c]) - math.dist(point, centre)

    def edge_valid(self, a, b):
        # Distance to blocked space is 1-Lipschitz. Subtract half the sample
        # interval so validity covers the entire segment, including grid corners.
        distance = math.dist(a, b)
        count = max(1, math.ceil(distance / (self.origin[3] * 0.5)))
        allowance = distance / count / 2
        required = self.settings.robot_radius + self.settings.safety_margin
        return all(self.lower_clearance(p) >= required + allowance
                   for p in np.linspace(a, b, count + 1))

    def remove(self, node):
        self.nodes.pop(node, None)
        self.sources.pop(node, None)
        self.isolation_age.pop(node, None)
        self.edges = {pair: cost for pair, cost in self.edges.items() if node not in pair}

    def world_key(self, point):
        # Fixed world lattice: map resizing does not make all cells "new".
        return tuple(round(float(v) / self.origin[3]) for v in point)

    def make_room(self):
        """Remove a redundant sampled node only if its neighbours remain linked.

        Requiring a neighbour clique is intentionally stricter than a global
        articulation test; removal cannot split any component or its neighbours.
        Travelled nodes and nodes near the robot are protected.
        """
        neighbours = self.adjacency()
        candidates = sorted(self.nodes, key=lambda i: (-len(neighbours[i]), i))
        for node in candidates:
            near = [i for i, _ in neighbours[node]]
            if (self.sources[node] == 'travelled' or len(near) < 2 or
                    math.dist(self.nodes[node], self.robot) < self.settings.node_spacing):
                continue
            if min(math.dist(self.nodes[node], self.nodes[i]) for i in near) > self.settings.coverage_radius:
                continue
            if all(tuple(sorted((a, b))) in self.edges for j, a in enumerate(near) for b in near[j+1:]):
                self.remove(node)
                self.replacements += 1
                return True
        return False

    def add(self, point, source, priority=False, bridge=False):
        s = self.settings
        # Same half-step guard as edge validation, before accepting a node.
        if self.lower_clearance(point) < s.robot_radius + s.safety_margin + self.origin[3]/4:
            return False
        if source != 'travelled' and self.cooldown.get(self.world_key(point), 0) > self.cycle:
            return False
        links = {i: math.dist(point, other) for i, other in self.nodes.items()
                 if math.dist(point, other) <= s.connection_radius and self.edge_valid(point, other)}
        joins_groups = len({self.groups.get(i, i) for i in links}) > 1
        # A close node across an obstruction cannot represent this candidate.
        if any(distance < s.node_spacing for distance in links.values()) and not (bridge and joins_groups):
            return False
        limit = s.max_nodes if source == 'travelled' else max(1, int(s.max_nodes*(1-s.reserve_fraction)))
        if len(self.nodes) >= limit:
            if not priority or self.replacements >= s.samples_per_update or not self.make_room():
                self.budget_rejections += 1
                return False
        ident = self.next_id
        self.nodes[ident] = tuple(float(x) for x in point)
        self.sources[ident] = source
        for other, distance in links.items():
            if other in self.nodes:
                self.edges[tuple(sorted((ident, other)))] = distance
        self.next_id += 1
        self.groups = self.components()
        return True

    def reconnect(self):
        self.edges = {}
        ids = sorted(self.nodes)
        if len(ids) > 1:
            points = np.array([self.nodes[i] for i in ids])
            for a, b in sorted(cKDTree(points).query_pairs(self.settings.connection_radius)):
                if self.edge_valid(points[a], points[b]):
                    self.edges[(ids[a], ids[b])] = math.dist(points[a], points[b])
        self.groups = self.components()

    def repair(self):
        """Bounded sampling around small components and between nearby groups."""
        if len(set(self.groups.values())) < 2:
            return
        sizes = {g: list(self.groups.values()).count(g) for g in set(self.groups.values())}
        targets = sorted(self.nodes, key=lambda i: (sizes[self.groups[i]], i))
        for attempt in range(self.settings.repair_attempts):
            node = targets[attempt % len(targets)]
            if node not in self.nodes:
                continue
            others = [i for i in self.nodes if self.groups[i] != self.groups[node]]
            if not others:
                break
            other = min(others, key=lambda i: math.dist(self.nodes[node], self.nodes[i]))
            centre = (np.array(self.nodes[node]) + self.nodes[other])/2
            if math.dist(self.nodes[node], self.nodes[other]) > 2*self.settings.connection_radius:
                centre = np.array(self.nodes[node])
            candidate = centre + self.rng.uniform(-1, 1, 2)*self.settings.node_spacing
            r, c = world_to_cell(*candidate, *self.origin)
            if (0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]
                    and self.geometry.reachable[r, c]):
                self.add(cell_to_world(r, c, *self.origin), 'repair', priority=True, bridge=True)

    def update(self, grid, origin, robot):
        """Retain world nodes; revalidate all connections against this map snapshot."""
        if len(origin) != 4 or not all(math.isfinite(v) for v in origin) or origin[3] <= 0:
            raise ValueError('invalid map origin/resolution')
        if len(robot) != 2 or not all(math.isfinite(v) for v in robot):
            raise ValueError('invalid robot position')
        self.origin = origin
        self.grid = np.asarray(grid)
        self.robot = tuple(robot)
        s = self.settings
        self.geometry = analyse_grid(self.grid, origin[3], world_to_cell(*robot, *origin),
                                     s.robot_radius, s.safety_margin, s.free_threshold)
        self.cycle += 1
        self.budget_rejections = self.replacements = self.retired_isolates = 0
        self.cooldown = {k: expiry for k, expiry in self.cooldown.items() if expiry > self.cycle}
        if not self.trajectory or math.dist(robot, self.trajectory[-1]) >= origin[3]/2:
            self.trajectory.append(tuple(robot))
        invalid = [i for i, p in self.nodes.items()
                   if self.lower_clearance(p) < s.robot_radius + s.safety_margin + origin[3]/4]
        for i in invalid:
            self.remove(i)
        self.reconnect()
        before_id = self.next_id
        self.add(robot, 'travelled', priority=True)
        if s.mode != 'baseline':
            cells = np.argwhere(self.geometry.reachable)
            points = [cell_to_world(r, c, *origin) for r, c in cells]
            new_indices = [i for i, point in enumerate(points) if self.world_key(point) not in self.seen_cells]
            self.seen_cells.update(self.world_key(point) for point in points)
            radius = max(1, math.ceil(s.neighbourhood_radius / origin[3]))
            accepted = 0
            for attempt in range(s.attempts_per_update):
                if not len(cells) or accepted >= s.samples_per_update:
                    break
                # Half the attempts favour newly observed space for both PRMs.
                fresh = bool(new_indices) and attempt % 2 == 0
                index = new_indices[self.rng.integers(len(new_indices))] if fresh else self.rng.integers(len(cells))
                r, c = cells[index]
                local = self.geometry.clearance[max(0, r-radius):r+radius+1, max(0, c-radius):c+radius+1]
                rank = np.mean(local <= self.geometry.clearance[r, c])
                if (s.mode == 'distance_guided' and self.rng.random() >= s.uniform_fraction
                        and rank < s.acceptance_fraction):
                    continue
                accepted += self.add(points[index], 'sampled', priority=fresh)
            self.repair()
        neighbours = self.adjacency()
        for node in list(self.nodes):
            self.isolation_age[node] = self.isolation_age.get(node, 0)+1 if not neighbours[node] else 0
            if self.sources[node] != 'travelled' and self.isolation_age[node] >= s.isolation_patience:
                self.cooldown[self.world_key(self.nodes[node])] = self.cycle+s.isolation_patience
                self.remove(node)
                self.retired_isolates += 1
        self.groups = self.components()
        metrics = self.metrics()
        metrics.update(added_nodes=self.next_id-before_id, removed_nodes=len(invalid),
                       budget_rejections=self.budget_rejections, replacements=self.replacements,
                       retired_isolates=self.retired_isolates)
        return metrics

    def adjacency(self):
        neighbours = {i: [] for i in self.nodes}
        for (a, b), cost in self.edges.items():
            neighbours[a].append((b, cost))
            neighbours[b].append((a, cost))
        return neighbours

    def components(self):
        neighbours = self.adjacency()
        groups = {}
        for root in self.nodes:
            if root in groups:
                continue
            stack = [root]
            groups[root] = root
            while stack:
                for other, _ in neighbours[stack.pop()]:
                    if other not in groups:
                        groups[other] = root
                        stack.append(other)
        return groups

    def route(self, start, goal):
        """A* over validated edges; [] if no route (IDs must exist)."""
        if start not in self.nodes or goal not in self.nodes:
            return []
        neighbours = self.adjacency()
        costs, parents = {start: 0.0}, {}
        queue = [(math.dist(self.nodes[start], self.nodes[goal]), 0.0, start)]
        while queue:
            _, cost, node = heapq.heappop(queue)
            if cost > costs[node]:
                continue
            if node == goal:
                path = [goal]
                while path[-1] != start:
                    path.append(parents[path[-1]])
                return path[::-1]
            for other, edge_cost in neighbours[node]:
                new_cost = cost + edge_cost
                if new_cost < costs.get(other, math.inf):
                    costs[other], parents[other] = new_cost, node
                    heapq.heappush(queue, (new_cost + math.dist(self.nodes[other], self.nodes[goal]),
                                          new_cost, other))
        return []

    def robot_anchor(self):
        for node in sorted(self.nodes, key=lambda i: math.dist(self.robot, self.nodes[i])):
            if math.dist(self.robot, self.nodes[node]) > self.settings.connection_radius:
                break
            if self.edge_valid(self.robot, self.nodes[node]):
                return node
        return None

    def metrics(self):
        cells = np.argwhere(self.geometry.reachable)
        anchor = self.robot_anchor()
        connected = [i for i in self.nodes if anchor is not None and self.groups[i] == self.groups[anchor]]
        coverage = 0.0
        if len(cells) and connected:
            points = np.array([cell_to_world(r, c, *self.origin) for r, c in cells])
            distances, _ = cKDTree([self.nodes[i] for i in connected]).query(points)
            coverage = float(np.mean(distances <= self.settings.coverage_radius))
        # Deterministic spatial sample shared by all strategies. Unlike proximity,
        # a covered cell must have a collision-free link to the anchored component.
        checks = cells[np.linspace(0, len(cells)-1, min(len(cells), self.settings.coverage_samples), dtype=int)] if len(cells) else []
        successes = 0
        for r, c in checks:
            point = cell_to_world(r, c, *self.origin)
            successes += any(math.dist(point, self.nodes[i]) <= self.settings.coverage_radius
                             and self.edge_valid(point, self.nodes[i]) for i in connected)
        neighbours = self.adjacency()
        isolated = [i for i in self.nodes if not neighbours[i]]
        diagnostics = []
        for node in isolated[:20]:
            distances = sorted(math.dist(self.nodes[node], p) for i, p in self.nodes.items() if i != node)
            nearby = sum(d <= self.settings.connection_radius for d in distances)
            diagnostics.append({'id': node, 'position': self.nodes[node],
                'clearance_m': self.lower_clearance(self.nodes[node]),
                'nearest_node_m': distances[0] if distances else None,
                'nearby_candidates': nearby, 'age_updates': self.isolation_age.get(node, 0),
                'reason': 'all_nearby_edges_blocked' if nearby else 'no_nearby_nodes'})
        unvisited = 0
        if self.nodes and self.trajectory:
            distances, _ = cKDTree(self.trajectory).query(list(self.nodes.values()))
            unvisited = int(np.count_nonzero(distances > self.settings.visit_radius))
        return {'isolated_nodes': len(isolated), 'isolated_diagnostics': diagnostics,
                'beyond_sampled_trajectory_nodes': unvisited,
                'collision_checked_coverage': successes/len(checks) if len(checks) else 0.0,
                'coverage_test_cells': len(checks), 'mode': self.settings.mode, 'nodes': len(self.nodes), 'edges': len(self.edges),
                'components': len(set(self.groups.values())), 'robot_connected_nodes': len(connected),
                'reachable_area_m2': float(len(cells) * self.origin[3]**2),
                'node_proximity_coverage': coverage, 'coverage_radius_m': self.settings.coverage_radius,
                'at_node_limit': len(self.nodes) >= self.settings.max_nodes}

    def query_route(self, start, goal):
        """Exact requested endpoints; virtual attachments, never snap a goal."""
        if (self.lower_clearance(start) < self.settings.robot_radius+self.settings.safety_margin or
                self.lower_clearance(goal) < self.settings.robot_radius+self.settings.safety_margin):
            return [], 'unsafe_endpoint'
        neighbours = self.adjacency()
        positions = dict(self.nodes)
        positions[-1], positions[-2] = tuple(start), tuple(goal)
        neighbours[-1], neighbours[-2] = [], []
        for endpoint in (-1, -2):
            for node, point in self.nodes.items():
                distance = math.dist(positions[endpoint], point)
                if distance <= self.settings.connection_radius and self.edge_valid(positions[endpoint], point):
                    neighbours[endpoint].append((node, distance))
                    neighbours[node].append((endpoint, distance))
        if not neighbours[-1] or not neighbours[-2]:
            return [], 'endpoint_connection_failed'
        queue, costs, parents = [(0., -1)], {-1: 0.}, {}
        while queue:
            cost, node = heapq.heappop(queue)
            if cost > costs[node]:
                continue
            if node == -2:
                path = [-2]
                while path[-1] != -1:
                    path.append(parents[path[-1]])
                return [positions[i] for i in path[::-1]], 'ok'
            for other, weight in neighbours[node]:
                value = cost+weight
                if value < costs.get(other, math.inf):
                    costs[other], parents[other] = value, node
                    heapq.heappush(queue, (value, other))
        return [], 'disconnected_graph'
