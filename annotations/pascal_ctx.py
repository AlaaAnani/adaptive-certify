# %%
import json
from graphviz import Digraph

categories = []

# leaves
categories.append({ 'name': 'aeroplane', 'level': 0, 'supercategory': 'vehicle' })
categories.append({'name': 'bicycle','level': 0,'supercategory': 'vehicle'})
categories.append({'name': 'bird','level': 0,'supercategory': 'animal'})
categories.append({'name': 'boat','level': 0,'supercategory': 'vehicle'})
categories.append({'name': 'bottle','level': 0,'supercategory': 'household-items'})
categories.append({'name': 'bus','level': 0,'supercategory': 'vehicle'})
categories.append({'name': 'car','level': 0,'supercategory': 'vehicle'})
categories.append({'name': 'cat','level': 0,'supercategory': 'animal'})
categories.append({'name': 'chair','level': 0,'supercategory': 'furniture'})
categories.append({'name': 'cow','level': 0,'supercategory': 'animal'})
categories.append({'name': 'table','level': 0,'supercategory': 'furniture'})
categories.append({'name': 'dog','level': 0,'supercategory': 'animal'})
categories.append({'name': 'horse','level': 0,'supercategory': 'animal'})
categories.append({'name': 'motorbike','level': 0,'supercategory': 'vehicle'})
categories.append({'name': 'person','level': 0,'supercategory': 'living-being'})
categories.append({'name': 'pottedplant','level': 0,'supercategory': 'household-items'})
categories.append({'name': 'sheep','level': 0,'supercategory': 'animal'})
categories.append({'name': 'sofa','level': 0,'supercategory': 'furniture'})
categories.append({'name': 'train','level': 0,'supercategory': 'vehicle'})
categories.append({'name': 'tvmonitor','level': 0,'supercategory': 'electronic'})
categories.append({'name': 'bag','level': 0,'supercategory': 'household-items'})
categories.append({'name': 'bed','level': 0,'supercategory': 'furniture'})
categories.append({'name': 'bench','level': 0,'supercategory': 'furniture'})
categories.append({'name': 'book', 'level': 0,'supercategory': 'household-items'})
categories.append({'name': 'building', 'level': 0,'supercategory': 'construction'})
categories.append({'name': 'cabinet', 'level': 0,'supercategory': 'furniture'})
categories.append({'name': 'ceiling', 'level': 0,'supercategory': 'construction'})
categories.append({'name': 'cloth', 'level': 0,'supercategory': 'household-items'})
categories.append({'name': 'computer', 'level': 0,'supercategory': 'electronic'})
categories.append({'name': 'cup', 'level': 0,'supercategory': 'household-items'})
categories.append({'name': 'door', 'level': 0,'supercategory': 'construction'})
categories.append({'name': 'fence', 'level': 0,'supercategory': 'construction'})
categories.append({'name': 'floor', 'level': 0,'supercategory': 'surface'})
categories.append({'name': 'flower', 'level': 0,'supercategory': 'nature'})
categories.append({'name': 'food', 'level': 0,'supercategory': 'household-items'})
categories.append({'name': 'grass', 'level': 0,'supercategory': 'nature'})
categories.append({'name': 'ground', 'level': 0,'supercategory': 'surface'})
categories.append({'name': 'keyboard', 'level': 0,'supercategory': 'electronic'})
categories.append({'name': 'light', 'level': 0,'supercategory': 'furniture'})
categories.append({'name': 'mountain', 'level': 0,'supercategory': 'nature'})
categories.append({'name': 'mouse', 'level': 0,'supercategory': 'electronic'})
categories.append({'name': 'curtain', 'level': 0,'supercategory': 'furniture'})
categories.append({'name': 'platform', 'level': 0,'supercategory': 'construction'})
categories.append({'name': 'sign', 'level': 0,'supercategory': 'construction'})
categories.append({'name': 'plate', 'level': 0,'supercategory': 'household-items'})
categories.append({'name': 'road', 'level': 0,'supercategory': 'surface'})
categories.append({'name': 'rock', 'level': 0,'supercategory': 'nature'})
categories.append({'name': 'shelves', 'level': 0,'supercategory': 'furniture'})
categories.append({'name': 'sidewalk', 'level': 0,'supercategory': 'surface'})
categories.append({'name': 'sky', 'level': 0,'supercategory': 'nature'})
categories.append({'name': 'snow', 'level': 0,'supercategory': 'nature'})
categories.append({'name': 'bedclothes', 'level': 0,'supercategory': 'household-items'})
categories.append({'name': 'track', 'level': 0,'supercategory': 'surface'})
categories.append({'name': 'tree', 'level': 0,'supercategory': 'nature'})
categories.append({'name': 'truck', 'level': 0,'supercategory': 'vehicle'})
categories.append({'name': 'wall', 'level': 0,'supercategory': 'construction'})
categories.append({'name': 'water', 'level': 0,'supercategory': 'nature'})
categories.append({'name': 'window', 'level': 0,'supercategory': 'furniture'})
categories.append({'name': 'wood', 'level': 0,'supercategory': 'nature'})
file_path = '/BS/mlcysec2/work/hierarchical-certification/HRNet-Semantic-Segmentation/data/pascal_ctx/labels_59_ordered.txt'
items = []
with open(file_path, 'r') as file:
    for line in file:
        item = line.split(':')[-1].strip()
        items.append(item)
for j, c in enumerate(categories):
    for i, it in enumerate(items):
        if c['name'] == it:
            categories[j]['id'] = i
print(categories)
categories.append({'id': 60,'name': 'vehicle', 'level': 1,'supercategory': 'outdoor'})
categories.append({'id': 61,'name': 'animal', 'level': 1,'supercategory': 'living-being'})
categories.append({'id': 62,'name': 'household-items', 'level': 1,'supercategory': 'house'})
categories.append({'id': 63,'name': 'furniture', 'level': 1,'supercategory': 'house'})
categories.append({'id': 64,'name': 'construction', 'level': 1,'supercategory': 'structure'})
categories.append({'id': 65,'name': 'electronic', 'level': 1,'supercategory': 'house'})
categories.append({'id': 66,'name': 'surface', 'level': 1,'supercategory': 'structure'})
categories.append({'id': 67,'name': 'nature', 'level': 1,'supercategory': 'outdoor'})


categories.append({'id': 68,'name': 'living-being', 'level': 2,'supercategory': 'outdoor'})
categories.append({'id': 69,'name': 'house', 'level': 2,'supercategory': 'indoor'})
categories.append({'id': 70,'name': 'structure', 'level': 2,'supercategory': 'outdoor'})

categories.append({'id': 71,'name': 'outdoor', 'level': 3,'supercategory': 'everything'})
categories.append({'id': 72,'name': 'indoor', 'level': 3,'supercategory': 'everything'})

categories.append({'id': 73,'name': 'everything', 'level': 4,'supercategory': None})


nodes = categories
for n in nodes:
    n['train_id'] = n['id'] - 1


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

for l, ns in nodes_by_level.items():
    print(l)
    for n_ in ns:
        print(n_['name'])
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
    

hierarchy_dict['node_info_gain_lookup'] = dict(sorted(node_info_gain_lookup.items(), key=lambda item: item[0]))
filepath = 'HRNet-Semantic-Segmentation/data/pascal_ctx/pascal_ctx_hierarchy.json'
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
graph.render('annotations/pascal_ctx',)