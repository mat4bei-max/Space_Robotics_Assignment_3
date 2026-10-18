"""Replay identical online inputs, evaluate exact endpoint queries, export evidence.

python -m advanced_functions.roadmap_evaluation --recording DIR --pairs pairs.json --output DIR
python -m advanced_functions.roadmap_evaluation --demo --output /tmp/roadmap_demo
"""
import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from advanced_functions.cave_geometry import cell_to_world
from advanced_functions.online_roadmap import OnlineRoadmap, Settings


def load_pairs(path):
    pairs = json.loads(Path(path).read_text())
    if not isinstance(pairs, list) or not pairs:
        raise ValueError('pairs must be a nonempty JSON list')
    names = set()
    for pair in pairs:
        if not isinstance(pair.get('name'), str) or pair['name'] in names:
            raise ValueError('each pair requires a unique string name')
        names.add(pair['name'])
        for key in ('start', 'goal'):
            point = pair[key]
            if len(point) != 2 or not all(isinstance(v, (float, int)) and math.isfinite(v) for v in point):
                raise ValueError('endpoints must be finite [x, y] coordinates')
    return pairs


class GridReference:
    """Eight-connected cell-centre reference with the SAME continuous edge check.

    It is a discrete reference, not a continuous-space optimum. Endpoints are
    attached with checked short segments; their requested coordinates never move.
    """
    def __init__(self, graph):
        self.graph = graph
        required = graph.settings.robot_radius+graph.settings.safety_margin+graph.origin[3]/4
        cells = np.argwhere(graph.geometry.clearance >= required)
        self.points = [cell_to_world(r,c,*graph.origin) for r,c in cells]
        self.tree = cKDTree(self.points) if self.points else None
        ids = {tuple(cell): i for i,cell in enumerate(cells)}
        rows, cols, costs = [], [], []
        for (r,c), node in ids.items():
            for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
                other = ids.get((r+dr,c+dc))
                if other is not None and graph.edge_valid(self.points[node], self.points[other]):
                    cost = math.dist(self.points[node], self.points[other])
                    rows.extend([node,other]); cols.extend([other,node]); costs.extend([cost,cost])
        self.matrix = csr_matrix((costs,(rows,cols)), shape=(len(cells),len(cells)))

    def query(self, start, goal):
        if self.tree is None:
            return None
        attachments = []
        for point in (start,goal):
            attachments.append([(i,math.dist(point,self.points[i]))
                for i in self.tree.query_ball_point(point, self.graph.origin[3]*2)
                if self.graph.edge_valid(point,self.points[i])])
        a,b = attachments
        if not a or not b:
            return None
        distances = dijkstra(self.matrix, indices=[i for i,_ in a], directed=False)
        best = min(distances[k,j]+cost_a+cost_b for k,(_,cost_a) in enumerate(a) for j,cost_b in b)
        return float(best) if math.isfinite(best) else None


def demo_inputs():
    """Progressively reveal two chambers joined by a turning passage."""
    grid = np.full((50,80),100,dtype=np.int16)
    grid[5:45,3:30]=0
    grid[19:31,29:45]=0
    grid[19:39,38:47]=0
    grid[30:46,46:77]=0
    origin=(0.,0.,0.,0.25)
    poses=[(2.,5.),(4.,5.),(6.,5.),(8.,6.),(10.5,6.),(10.5,8.5),(13.,9.),(16.,9.)]
    for i,pose in enumerate(poses):
        observed=grid.copy()
        observed[:, min(80,30+i*8):]=-1
        for repeat in range(2):
            yield observed, origin, pose, (i*2+repeat)*2., asdict(Settings())


def read_inputs(directory):
    files=sorted(Path(directory).glob('*.npz'))
    if not files:
        raise ValueError('recording directory must contain snapshot .npz files (select a session subdirectory)')
    last_time=-math.inf
    last_frame=None
    for path in files:
        with np.load(path,allow_pickle=False) as data:
            time=float(data['time_ns'])/1e9
            frame=str(data['frame'])
            if time < last_time or (last_frame is not None and last_frame != frame):
                raise ValueError('recording crosses a simulation reset/frame change; use one session')
            last_time,last_frame=time,frame
            yield data['grid'].copy(), tuple(data['origin']), tuple(data['robot']), time, json.loads(str(data['settings']))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evaluate(inputs, pairs, output, seeds, budgets):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    inputs=list(inputs)
    if not inputs:
        raise ValueError('no input snapshots')
    history, routes, summaries=[],[],[]
    # Shared reference from the final observed map, independent of PRM outcomes.
    grid,origin,robot,_,config=inputs[-1]
    reference_graph=OnlineRoadmap(Settings(**dict(config,mode='baseline')))
    reference_graph.update(grid,origin,robot)
    reference=GridReference(reference_graph)
    reference_lengths={p['name']:reference.query(p['start'],p['goal']) for p in pairs}
    for budget in budgets:
        for seed in seeds:
            for mode in ('baseline','uniform','distance_guided'):
                graph=OnlineRoadmap(Settings(**dict(config,mode=mode,seed=seed,max_nodes=budget)))
                for grid,origin,robot,time,_ in inputs:
                    start=perf_counter()
                    metrics=graph.update(grid,origin,robot)
                    duration=(perf_counter()-start)*1000
                    history.append(dict(mode=mode,seed=seed,budget=budget,time_s=time,
                        nodes=metrics['nodes'],edges=metrics['edges'],isolated=metrics['isolated_nodes'],
                        connected_nodes=metrics['robot_connected_nodes'],
                        coverage=metrics['collision_checked_coverage'],update_ms=duration,
                        replacements=metrics['replacements'],budget_rejections=metrics['budget_rejections']))
                successes=0
                feasible_successes=0
                feasible_count=sum(v is not None for v in reference_lengths.values())
                for pair in pairs:
                    start=perf_counter()
                    path,status=graph.query_route(pair['start'],pair['goal'])
                    duration=(perf_counter()-start)*1000
                    length=sum(math.dist(a,b) for a,b in zip(path,path[1:])) if path else None
                    ref=reference_lengths[pair['name']]
                    successes+=bool(path)
                    feasible_successes += bool(path) and ref is not None
                    routes.append(dict(mode=mode,seed=seed,budget=budget,pair=pair['name'],
                        start=json.dumps(pair['start']),goal=json.dumps(pair['goal']),status=status,
                        length_m=length,reference_length_m=ref,
                        relative_to_grid=length/ref if length is not None and ref else None,query_ms=duration))
                summaries.append(dict(mode=mode,seed=seed,budget=budget,success_fraction=successes/len(pairs),
                    reference_reachable_success_fraction=feasible_successes/feasible_count if feasible_count else None,
                    nodes=len(graph.nodes),coverage=metrics['collision_checked_coverage'],
                    isolated=metrics['isolated_nodes']))
                if seed == seeds[0]:
                    draw_graph(graph, pairs, output/f'map_{mode}_{budget}.png')
                print(f'{mode}, seed={seed}, budget={budget}: {successes}/{len(pairs)} routes, '
                      f'{len(graph.nodes)} nodes, {metrics["isolated_nodes"]} isolated',flush=True)
    write_csv(output/'history.csv',history)
    write_csv(output/'routes.csv',routes)
    (output/'summary.json').write_text(json.dumps({'pairs':pairs,'settings':config,'seeds':seeds,
        'budgets':budgets,'snapshots':len(inputs),'results':summaries,
        'notes':['Synthetic demo is not cave validation.','Grid reference uses identical clearance assumptions.',
                 'Route ratios may be below 1: the grid reference is discretised.',
                 'Coverage uses a deterministic subset of reachable cell centres.',
                 'Equal capacity does not guarantee equal accepted node counts.']},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for mode in ('baseline','uniform','distance_guided'):
        for axis,key,label in zip(axes,('reference_reachable_success_fraction','coverage','isolated'),
                                  ('Reference-reachable route success','Checked coverage estimate','Isolated nodes')):
            means=[np.mean([r[key] for r in summaries if r['mode']==mode and r['budget']==b and r[key] is not None]) for b in budgets]
            stds=[np.std([r[key] for r in summaries if r['mode']==mode and r['budget']==b and r[key] is not None]) for b in budgets]
            axis.errorbar(budgets,means,yerr=stds,marker='o',label=mode,capsize=3)
            axis.set_xlabel('Node capacity'); axis.set_ylabel(label); axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/'comparison.png',dpi=160); plt.close(fig)


def draw_graph(graph, pairs, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig,axis=plt.subplots(figsize=(9,6))
    r,c=np.indices((graph.grid.shape[0]+1,graph.grid.shape[1]+1))
    ox,oy,yaw,res=graph.origin
    x=ox+np.cos(yaw)*c*res-np.sin(yaw)*r*res
    y=oy+np.sin(yaw)*c*res+np.cos(yaw)*r*res
    image=np.where(graph.grid<0,0.5,np.where(graph.grid<=graph.settings.free_threshold,1.,0.))
    axis.pcolormesh(x,y,image,cmap='gray',vmin=0,vmax=1,rasterized=True)
    axis.add_collection(LineCollection([[graph.nodes[a],graph.nodes[b]] for a,b in graph.edges],
                                      colors='steelblue',linewidths=0.5,alpha=0.6))
    adjacency=graph.adjacency()
    for isolated,colour,label in ((False,'blue','Connected nodes'),(True,'red','Isolated nodes')):
        points=[p for i,p in graph.nodes.items() if (not adjacency[i])==isolated]
        if points:
            axis.scatter(*np.array(points).T,c=colour,s=10,label=label)
    for pair in pairs:
        path,status=graph.query_route(pair['start'],pair['goal'])
        if path:
            axis.plot(*np.array(path).T,linewidth=1.7,label=pair['name'])
        else:
            axis.scatter(*pair['goal'],marker='x',c='red')
            axis.annotate(pair['name']+': '+status,pair['goal'],fontsize=7)
    axis.set_aspect('equal'); axis.set_xlabel('Map x (m)'); axis.set_ylabel('Map y (m)')
    axis.set_title(f'{graph.settings.mode}: {len(graph.nodes)} nodes / capacity {graph.settings.max_nodes}')
    axis.legend(fontsize=7); fig.tight_layout(); fig.savefig(filename,dpi=150); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--recording'); group.add_argument('--demo',action='store_true')
    parser.add_argument('--pairs'); parser.add_argument('--output',required=True)
    parser.add_argument('--seeds',default='7,19,31'); parser.add_argument('--budgets',default='120,240')
    args=parser.parse_args()
    if not args.demo and not args.pairs:
        parser.error('--pairs is required for a recorded cave run')
    pairs=load_pairs(args.pairs) if args.pairs else [
        {'name':'room_to_passage','start':[2.,5.],'goal':[10.5,6.]},
        {'name':'through_bend','start':[2.,5.],'goal':[16.,9.]},
        {'name':'far_chamber','start':[10.5,6.],'goal':[18.,10.]},
        {'name':'blocked_goal','start':[2.,5.],'goal':[12.,2.]}]
    evaluate(demo_inputs() if args.demo else read_inputs(args.recording),pairs,args.output,
             [int(x) for x in args.seeds.split(',')],[int(x) for x in args.budgets.split(',')])


if __name__=='__main__':
    main()
