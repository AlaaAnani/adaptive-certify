import json
from pycocotools.coco import COCO
from graphviz import Digraph
from tqdm import tqdm
import os

annotations_path = 'HRNet-Semantic-Segmentation/data/cocostuff/cocostuff-10k-v1.1.json'
coco = COCO(annotations_path)

clsID_to_trID = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    13: 12,
    14: 13,
    15: 14,
    16: 15,
    17: 16,
    18: 17,
    19: 18,
    20: 19,
    21: 20,
    22: 21,
    23: 22,
    24: 23,
    25: 24,
    27: 25,
    28: 26,
    31: 27,
    32: 28,
    33: 29,
    34: 30,
    35: 31,
    36: 32,
    37: 33,
    38: 34,
    39: 35,
    40: 36,
    41: 37,
    42: 38,
    43: 39,
    44: 40,
    46: 41,
    47: 42,
    48: 43,
    49: 44,
    50: 45,
    51: 46,
    52: 47,
    53: 48,
    54: 49,
    55: 50,
    56: 51,
    57: 52,
    58: 53,
    59: 54,
    60: 55,
    61: 56,
    62: 57,
    63: 58,
    64: 59,
    65: 60,
    67: 61,
    70: 62,
    72: 63,
    73: 64,
    74: 65,
    75: 66,
    76: 67,
    77: 68,
    78: 69,
    79: 70,
    80: 71,
    81: 72,
    82: 73,
    84: 74,
    85: 75,
    86: 76,
    87: 77,
    88: 78,
    89: 79,
    90: 80,
    92: 81,
    93: 82,
    94: 83,
    95: 84,
    96: 85,
    97: 86,
    98: 87,
    99: 88,
    100: 89,
    101: 90,
    102: 91,
    103: 92,
    104: 93,
    105: 94,
    106: 95,
    107: 96,
    108: 97,
    109: 98,
    110: 99,
    111: 100,
    112: 101,
    113: 102,
    114: 103,
    115: 104,
    116: 105,
    117: 106,
    118: 107,
    119: 108,
    120: 109,
    121: 110,
    122: 111,
    123: 112,
    124: 113,
    125: 114,
    126: 115,
    127: 116,
    128: 117,
    129: 118,
    130: 119,
    131: 120,
    132: 121,
    133: 122,
    134: 123,
    135: 124,
    136: 125,
    137: 126,
    138: 127,
    139: 128,
    140: 129,
    141: 130,
    142: 131,
    143: 132,
    144: 133,
    145: 134,
    146: 135,
    147: 136,
    148: 137,
    149: 138,
    150: 139,
    151: 140,
    152: 141,
    153: 142,
    154: 143,
    155: 144,
    156: 145,
    157: 146,
    158: 147,
    159: 148,
    160: 149,
    161: 150,
    162: 151,
    163: 152,
    164: 153,
    165: 154,
    166: 155,
    167: 156,
    168: 157,
    169: 158,
    170: 159,
    171: 160,
    172: 161,
    173: 162,
    174: 163,
    175: 164,
    176: 165,
    177: 166,
    178: 167,
    179: 168,
    180: 169,
    181: 170,
    182: 171
}

# Get category information from COCO-Stuff
categories = coco.loadCats(coco.getCatIds())
print(f'Loaded {len(categories)} leaf classes')
nms = [cat['name'] for cat in categories]
#print('COCO categories: \n{}\n'.format(' '.join(nms)), f'count={len(nms)}')

for idx, cat in enumerate(categories[:]): 
    categories[idx]['level'] = 0
    categories[idx]['cls_id'] =  categories[idx]['id'] 
    categories[idx]['id'] = clsID_to_trID[categories[idx]['id']]
    if cat['name'] == 'indoor' or cat['name'] == 'outdoor':
        categories.remove(cat)
    if cat['supercategory'] == 'indoor' or cat['supercategory'] == 'outdoor':
        categories[idx]['supercategory'] += '-things'
    if cat['name'] == 'person':
        categories[idx]['supercategory'] = 'outdoor-things'

    
# adding levels to the hierarchy
supercategories = {
    'indoor-things': ['appliance', 'electronic', 'furniture', 'food', 'kitchen', 'toothbrush', 
                      'hair brush', 'hair drier', 'teddy bear', 'scissors', 'vase', 'clock',
                      'book', ],
    'outdoor-things': ['sports', 'accessory', 'animal', 'vehicle', 'person',
                       'bench', 'parking meter', 'stop sign', 'fire hydrant', 'traffic light'],
    'indoor-stuff': ['textile', 'furniture-stuff', 'window', 'floor', 'ceiling', 'wall', 'raw-material', 'food-stuff'],
    'outdoor-stuff': ['water', 'ground', 'solid', 'sky', 'plant', 'structural', 'building'],
    'things': ['indoor-things', 'outdoor-things'],
    'stuff': ['indoor-stuff', 'outdoor-stuff']
}
i = 172

for super_cat_name in set(cat['supercategory'] for cat in categories):
    for k, v in supercategories.items():
        if super_cat_name in v:
            if 'outdoor' in super_cat_name or 'indoor' in super_cat_name: continue
            categories.append(
                {
                    'id': i,
                    'name': super_cat_name,
                    'level': 1 if super_cat_name not in list(supercategories.keys()) else 2,
                    'supercategory': k
                }
            )
            i +=1

for super_cat_name in set(cat['supercategory'] for cat in categories):
    for cat_name in ['things', 'stuff']:
        if super_cat_name in supercategories[cat_name]:
            categories.append(
                {
                    'id': i,
                    'name': super_cat_name,
                    'level': 2,
                    'supercategory': cat_name
                }
            )
            i+=1
categories.append(
    {
        'id': i,
        'name': 'things',
        'level': 3,
        'supercategory': 'everything'
        
    }
)
i+=1
categories.append(
    {
        'id': i,
        'name': 'stuff',
        'level': 3,
        'supercategory': 'everything'
        
    }
)      
i+=1   
categories.append(
    {
        'id': i,
        'name': 'everything',
        'level': 4,
        'supercategory': None
        
    }
)    
print('Number of nodes in the DAG hierarchy', i)
names = []
unique_categories = []

# Remove repeated dictionaries
for d in categories:
    if d['name'] not in names:
        unique_categories.append(d)
        names.append(d['name'])

nodes = unique_categories 
for n in nodes:
    n['train_id'] = n['id'] - 1


hierarchy_dict = {'nodes': 
    categories}

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
filepath = 'HRNet-Semantic-Segmentation/data/cocostuff/cocostuff_hierarchy.json'
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
graph.render('annotations/cocostuff',)
