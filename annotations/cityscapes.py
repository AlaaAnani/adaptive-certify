import json
from pycocotools.coco import COCO
from graphviz import Digraph

categories = []
ds = 'acdc'
# leaves

categories.append({'id':0, 'name': 'road', 'level': 0, 'supercategory': 'everything'})
categories.append({'id':1, 'name': 'sidewalk', 'level': 0, 'supercategory': 'flat obstacle' })
categories.append({'id':2, 'name': 'building', 'level': 0, 'supercategory': 'construction and vegetation' })
categories.append({'id':3, 'name': 'wall', 'level': 0, 'supercategory': 'construction and vegetation' })
categories.append({'id':4, 'name': 'fence', 'level': 0, 'supercategory': 'construction and vegetation' })
categories.append({'id':5, 'name': 'pole', 'level': 0, 'supercategory': 'construction and vegetation' })
categories.append({'id':6, 'name': 'traffic light', 'level': 0, 'supercategory': 'traffic-sign' })
categories.append({'id':7, 'name': 'traffic sign', 'level': 0, 'supercategory': 'traffic-sign' })
categories.append({'id':8, 'name': 'vegetation', 'level': 0, 'supercategory': 'construction and vegetation' })
categories.append({'id':9, 'name': 'terrain', 'level':0,  'supercategory': 'flat obstacle' })
categories.append({'id':10, 'name': 'sky', 'level':0, 'supercategory': 'construction and vegetation' })
categories.append({'id':11, 'name': 'person', 'level':0, 'supercategory': 'human' })
categories.append({'id':12, 'name': 'rider', 'level':0, 'supercategory': 'human' })
categories.append({'id':13, 'name': 'car', 'level':0, 'supercategory': 'vehicle' })
categories.append({'id':14, 'name': 'truck', 'level':0, 'supercategory': 'vehicle' })
categories.append({'id':15, 'name': 'bus', 'level':0, 'supercategory': 'vehicle' })
categories.append({'id':16, 'name': 'train', 'level':0, 'supercategory': 'vehicle' })
categories.append({'id':17, 'name': 'motorcycle', 'level':0, 'supercategory': 'vehicle' })
categories.append({'id':18, 'name': 'bicycle', 'level':0, 'supercategory': 'vehicle' })

categories.append({'id':19, 'name': 'construction and vegetation', 'level': 1, 'supercategory': 'static obstacle'})
categories.append({'id':20, 'name': 'traffic-sign', 'level': 1, 'supercategory': 'static obstacle'})
categories.append({'id':21, 'name': 'human', 'level': 1, 'supercategory': 'dynamic obstacle'})
categories.append({'id':22, 'name': 'vehicle', 'level': 1, 'supercategory': 'dynamic obstacle'})


categories.append({'id':23, 'name': 'static obstacle', 'level': 2, 'supercategory': 'obstacle'})
categories.append({'id':24, 'name': 'dynamic obstacle', 'level': 2, 'supercategory': 'obstacle'})
categories.append({'id':25, 'name': 'flat obstacle', 'level': 2, 'supercategory': 'obstacle'})

categories.append({'id':26, 'name': 'obstacle', 'level': 3, 'supercategory': 'everything'})


categories.append({'id':27, 'name': 'everything', 'level': 4, 'supercategory': None})



nodes = categories
for n in nodes:
    n['train_id'] = n['id'] 

hierarchy_dict = {'nodes': categories}

parent_by_level = {}
nodes_by_level = {}
parent_child_relationships = {}

nodes = hierarchy_dict['nodes']

max_id = -1
for node in nodes:
    if node['level'] not in nodes_by_level:
        nodes_by_level[node['level']] = []
    parent_name = node['supercategory']
    for n in nodes: 
        if parent_name == n['name']:
            parent_id = n['train_id']
    if parent_id not in parent_child_relationships:
        parent_child_relationships[parent_id] = []
    nodes_by_level[node['level']].append(node)
    parent_child_relationships[parent_id].append(node['train_id'])
    if node['level'] == 0:
        parent_child_relationships[node['train_id']] = []
    if node['train_id'] > max_id:
        max_id = node['train_id']

parent_child_relationships[max_id] = [node['train_id'] for node in nodes if node['train_id'] != max_id]

# Adjusted recursive function with a base condition
def collect_descendants_iterative(node_id, parent_to_children):
    descendants = []
    stack = [node_id]
    while stack:
        current_node = stack.pop()
        children = parent_to_children.get(current_node, [])
        for child in children:
            descendants.append(child)
            stack.append(child)
    return descendants

node_to_descendants = {}
for node in nodes:
    node_to_descendants[node['train_id']] = list(set(collect_descendants_iterative(node['train_id'], parent_child_relationships)))

for node in nodes:
    if node['level'] == 0:
        leaf_id = node['train_id']
        for level, level_nodes in nodes_by_level.items():
            found_parent_in_level = False
            for level_node in level_nodes:
                descendants = node_to_descendants[level_node['train_id']]
                if leaf_id in descendants:
                    parent_by_level[(leaf_id, level)] = level_node['train_id']
                    found_parent_in_level = True 
            if not found_parent_in_level:
                parent_by_level[(leaf_id, level)] = leaf_id

lookup_tables = {}

for level in nodes_by_level.keys():
    lookup_tables[level] = [0]*(len(nodes_by_level[0]))
    for i in range(len(nodes_by_level[0])):
        lookup_tables[level][i] = parent_by_level[(i, level)]

hierarchy_dict['lookup_tables'] = lookup_tables

node_info_gain_lookup = {}
# get the # descentant leaf nodes from each node in the DAG (to calculate the CIG later)
cnt = 0
for node in nodes:
    descendants = set(node_to_descendants[node['train_id']])
    num_leaves = 0
    for d_node in descendants:
        for leaf in nodes_by_level[0]:
            if d_node == leaf['train_id']:
                num_leaves +=1
    if num_leaves == 0: num_leaves = 1
    node_info_gain_lookup[int(node['train_id'])] = num_leaves
    

hierarchy_dict['node_info_gain_lookup'] = node_info_gain_lookup
filepath = f'HRNet-Semantic-Segmentation/data/{ds}/{ds}_hierarchy.json'
print('Saving created hierarchy DAG into', filepath)
json.dump(hierarchy_dict, open(filepath, 'w'), indent=4)

# Create a directed graph with horizontal layout
graph = Digraph(format='png', node_attr={'shape': 'box'}, graph_attr={'rankdir': 'TB'})

# Bucket the categories by level
levels = {}
for cat in nodes:
    level = cat['level']
    name = cat['name']
    if level not in levels:
        levels[level] = [name]
    else:
        levels[level].append(name)

# Create subgraphs to align nodes of the same level
for level in levels:
    with graph.subgraph() as s:
        s.attr(rank='same')
        for name in levels[level]:
            s.node(name, label=name, fontsize='10')

# Add edges based on relationships
for cat in nodes:
    if cat['supercategory']:
        graph.edge(cat['supercategory'], cat['name'])


# Visualize the graph
graph.render(f'annotations/{ds}',)
