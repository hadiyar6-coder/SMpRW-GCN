import os
import numpy as np
import time
import pandas as pd
import networkx as nx
import scipy.sparse as sp
import torch_geometric
import torch
import pickle
import sys
from deepwalk import DeepWalk
from node2vec import Node2Vec
from role_discovery import Role
from collections import defaultdict

# مسیر پایه شما
# use for data and output path in this SMpRW-GCN directory
BASE_PATH = 'C:/Users/hy/PycharmProjects/pythonProject6/bk3/SMpRW-GCN'
# fix this path from def prepare_dataset in this file for split data

def prepare_data(args):
    if args.dataset in {'cora', 'citeseer', 'pubmed'}:
        node_map, g, features, labels = load_data0(args)
    elif args.dataset in {'brazil', 'europe', 'usa'}:
        node_map, g, features, labels = load_data1(args)
    elif args.dataset in {'actor', 'chameleon', 'cornell', 'squirrel', 'texas', 'wisconsin'}:
        node_map, g, features, labels = load_data2(args)
    return node_map, g, features, labels


def load_data0(args):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    # smart data path resolver (no logic change for the rest)
    # priority:
    # 1) BASE_PATH/data/0   ( ld split-style)
    # 2) BASE_PATH/data/<dataset>
    # 3) BASE_PATH/data     (classic Planetoid)
    cand_paths = [
        os.path.join(BASE_PATH, 'data', '0'),
        os.path.join(BASE_PATH, 'data', args.dataset),
        os.path.join(BASE_PATH, 'data'),
    ]

    data_path = None
    for p in cand_paths:
        if os.path.exists(os.path.join(p, f"ind.{args.dataset}.x")):
            data_path = p
            break

    if data_path is None:
        raise FileNotFoundError(
            f"Could not find ind.{args.dataset}.x in any of these paths:\n" +
            "\n".join(cand_paths)
        )

    for name in names:
        with open(os.path.join(data_path, f"ind.{args.dataset}.{name}"), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pickle.load(f, encoding='latin1'))
            else:
                objects.append(pickle.load(f))

    x, y, tx, ty, allx, ally, graph_data = tuple(objects)

    test_index_path = os.path.join(data_path, f'ind.{args.dataset}.test.index')
    test_idx_reorder = parse_index_file(test_index_path)
    test_idx_range = np.sort(test_idx_reorder)

    if args.dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = features.toarray().astype(np.float32)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels_class = np.argmax(labels, axis=1).reshape(-1, 1)

    node_map = {i: i for i in range(len(labels_class))}

    out_path = os.path.join(BASE_PATH, 'output', args.dataset)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    pd.to_pickle(node_map, os.path.join(out_path, 'node_map.pkl'))

    g = nx.from_dict_of_lists(graph_data)
    if args.feat_flag is False:
        features = create_node_attribute_vector(args, g, 0)

    return node_map, g, features, labels_class


def load_data1(args):
    label_path = os.path.join(BASE_PATH, 'data/1/Category/', args.dataset + '.txt')
    edge_path = os.path.join(BASE_PATH, 'data/1/Edgelist/', args.dataset + '.edgelist')
    node_map = {}
    label_map = {}
    g = nx.Graph()
    with open(label_path) as fp:
        for idx, line in enumerate(fp):
            info = line.strip().split()
            node_map[info[0]] = idx
            label_map[idx] = info[-1]
    label_to_int = {value: index for index, value in enumerate(set(label_map.values()))}
    labels = np.empty((len(node_map), 1), dtype=np.int64)
    for idx in range(len(node_map)): labels[idx] = label_to_int[label_map[idx]]

    out_path = os.path.join(BASE_PATH, 'output', args.dataset)
    if not os.path.exists(out_path): os.makedirs(out_path)
    pd.to_pickle(node_map, out_path + '/node_map.pkl')

    with open(edge_path) as fp:
        for idx, line in enumerate(fp):
            info = line.strip().split()
            x = node_map[info[0]]
            y = node_map[info[1]]
            g.add_edge(x, y)
    features = create_node_attribute_vector(args, g, 1)
    return node_map, g, features, labels


def load_data2(args):
    base_d2 = os.path.join(BASE_PATH, 'data/2', args.dataset)
    edge_file = os.path.join(base_d2, 'out1_graph_edges.txt')
    feat_file = os.path.join(base_d2, 'out1_node_feature_label.txt')
    g = nx.Graph()
    features = {}
    node_map = {}
    label_map = {}
    idx = 0
    with open(feat_file) as f:
        f.readline()
        for line in f:
            line = line.rstrip().split('\t')
            features[idx] = np.array(line[1].split(','), dtype=np.float32)
            node_map[line[0]] = idx
            label_map[idx] = line[2]
            idx += 1
    label_to_int = {value: index for index, value in enumerate(set(label_map.values()))}
    labels = np.empty((len(node_map), 1), dtype=np.int64)
    for idx in range(len(node_map)): labels[idx] = label_to_int[label_map[idx]]

    out_path = os.path.join(BASE_PATH, 'output', args.dataset)
    if not os.path.exists(out_path): os.makedirs(out_path)
    pd.to_pickle(node_map, out_path + '/node_map.pkl')

    with open(edge_file) as f:
        f.readline()
        for line in f:
            line = line.rstrip().split('\t')
            x = node_map[line[0]]
            y = node_map[line[1]]
            g.add_edge(x, y)
    features = list(features.values())
    if args.feat_flag is False: features = create_node_attribute_vector(args, g, 2)
    return node_map, g, features, labels


def generate_structure_similar_matrix(args, graph, flag='generate'):

    cache_path = os.path.join(BASE_PATH, 'output', args.dataset)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    adj_file = os.path.join(cache_path, f'role_adj_{args.feat_generate}.pkl')
    weigh_file = os.path.join(cache_path, f'weigh_list_{args.feat_generate}.pkl')

    file_exists = os.path.exists(adj_file) and os.path.exists(weigh_file)

    if flag == 'generate' or (hasattr(args, "force_regen") and args.force_regen) or not file_exists:
        if not file_exists and flag == 'load':
            print(f"   [System] ⚠️ Cache not found for {args.dataset}. Generating roles now...")

        role = Role(args, graph)
        role_adj_all, weigh_all = role.get_role_adj_weigh()

        with open(adj_file, 'wb') as f:
            pickle.dump(role_adj_all, f)
        with open(weigh_file, 'wb') as f:
            pickle.dump(weigh_all, f)

        return role_adj_all, weigh_all

    else:
        with open(adj_file, 'rb') as f:
            role_adj_all = pickle.load(f)
        with open(weigh_file, 'rb') as f:
            weigh_all = pickle.load(f)

        return role_adj_all, weigh_all


def prepare_dataset(args, labels, i):
    num = len(labels)
    train_num = int(np.ceil(args.train_ra * num / 100.0))
    val_num = int(np.ceil(args.val_ra * num / 100.0))

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             "C:/Users/hy/PycharmProjects/pythonProject6/bk3/SMpRW-GCN"))
    split_dir = os.path.join(base_path, "SMpRWGCN", "tvtsplit", args.dataset, str(args.train_ra / 100))
    os.makedirs(split_dir, exist_ok=True)

    split_path = os.path.join(split_dir, f"{args.dataset}_tvt_index{i}.npy")

    print("\n==================== TVT SPLIT ====================")
    print("[INFO] split_path:", split_path)

    if os.path.exists(split_path):
        rand_indices = np.load(split_path)
        print("[INFO] ✅ Loaded TVT split from cache.")
    else:
        print("[WARNING] ❌ TVT split NOT found -> generating new split...")
        rand_indices = np.random.permutation(num)
        np.save(split_path, rand_indices)
        print("[INFO] ✅ Saved new split:", split_path)

    rand_indices = rand_indices.astype(int)

    train_idx = rand_indices[:train_num]
    val_idx = rand_indices[train_num:train_num + val_num]
    test_idx = rand_indices[train_num + val_num:]

    print(f"[INFO] train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print("===================================================\n")

    return train_idx, val_idx, test_idx


# توابع استاندارد دیگر
def parse_index_file(filename):
    index = []
    for line in open(filename): index.append(int(line.strip()))
    return index


def get_data(graph, feat_data, labels, role_adj_list, weigh_list):
    num_feats = len(feat_data[0])
    num_labels = np.max(labels) + 1
    neigh_edge_index = torch.tensor(list(zip(*graph.edges)), dtype=torch.long)
    neigh_edge_index = torch_geometric.utils.to_undirected(neigh_edge_index)
    neigh_edge_index, _ = torch_geometric.utils.remove_self_loops(neigh_edge_index)
    x = torch.from_numpy(np.array(feat_data)).float()

    y = torch.tensor(labels, dtype=torch.long).view(-1)
    neigh_data = torch_geometric.data.Data(x=x, edge_index=neigh_edge_index, y=y,
                                           edge_weight=torch.ones(neigh_edge_index.size(1)), num_feats=num_feats,
                                           num_labels=num_labels, num_nodes=len(graph.nodes))
    edges, weights = [], []
    for node, neighbors in role_adj_list.items():
        for neighbor in neighbors:
            edges.append((node, neighbor))
            weights.append(weigh_list[(node, neighbor)])
    struc_edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    struc_data = torch_geometric.data.Data(x=x, edge_index=struc_edge_index, y=y, edge_weight=edge_weight,
                                           num_feats=num_feats, num_labels=num_labels, num_nodes=len(graph.nodes))
    return neigh_data, struc_data


def get_merged_data(args, neigh_data, struc_data):
    num_nodes = neigh_data.x.size(0)

    # self-loop edges
    edge_index = torch.arange(0, num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    self_data = torch_geometric.data.Data(x=neigh_data.x, y=neigh_data.y, edge_index=edge_index)

    edge_dict = {}

    def add_edges(data, weight_factor):
        ei = data.edge_index
        ew = data.edge_weight if data.edge_weight is not None else torch.ones(ei.size(1), device=ei.device)

        for i in range(ei.size(1)):
            u = int(ei[0, i].item())   #ensure python int
            v = int(ei[1, i].item())
            w = float(ew[i].item()) * float(weight_factor)  #ensure float

            edge_dict[(u, v)] = edge_dict.get((u, v), 0.0) + w

    if args.alpha != 0:
        add_edges(self_data, args.alpha)
    if args.beta != 0:
        add_edges(neigh_data, args.beta)
    if args.gama != 0:
        add_edges(struc_data, args.gama)

    merged_edge_index = torch.tensor(list(edge_dict.keys()), dtype=torch.long).t()
    merged_edge_weight = torch.tensor(list(edge_dict.values()), dtype=torch.float)

    return torch_geometric.data.Data(
        x=neigh_data.x,
        y=neigh_data.y,
        edge_index=merged_edge_index,
        edge_weight=merged_edge_weight,
        num_feats=neigh_data.num_feats,
        num_labels=neigh_data.num_labels,
        num_nodes=neigh_data.num_nodes
    )



# Dummy create_node_attribute_vector just in case
def create_node_attribute_vector(args, graph, ID):
    base_feat_dir = os.path.join(BASE_PATH, 'data', str(ID), 'Feature')
    if not os.path.exists(base_feat_dir):
        os.makedirs(base_feat_dir)

    file_name = f"{args.dataset}_{args.feat_generate}_feat_data.pkl"
    path = os.path.join(base_feat_dir, file_name)

    if os.path.exists(path) and not (hasattr(args, "force_regen") and args.force_regen):
        print(f'[INFO] Loading cached features from {file_name}')
        feat_data = pd.read_pickle(path)
    else:
        if args.feat_generate == 'onehot':
            print(f'[INFO] Generating fresh One-Hot features...')
            feat_data = np.eye(len(graph.nodes()), dtype=np.float32)
        else:
            print(f'[INFO] Generating features by {args.feat_generate}, dim is {args.embedding_dim}')
            if args.feat_generate == 'node2vec':
                model = Node2Vec(graph, walk_length=10, num_walks=80, workers=18, p=0.25, q=2, use_rejection_sampling=0)
            elif args.feat_generate == 'deepwalk':
                model = DeepWalk(graph, walk_length=10, num_walks=80, workers=18)

            model.train(embed_size=args.embedding_dim)
            embeddings = model.get_embeddings()
            feat_data = np.zeros((len(graph.nodes()), args.embedding_dim))
            for i in range(len(graph.nodes())):
                if i in embeddings:
                    feat_data[i] = embeddings[i]
            feat_data = feat_data.astype(np.float32)

        pd.to_pickle(feat_data, path)
        print(f"[INFO] Features generated and saved to {file_name}")

    return feat_data