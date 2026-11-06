import math
import os
import shutil
import random
from collections import defaultdict
import numpy as np
import torch
from utils import preprocess_nxgraph


class StrucWalkre:
    def __init__(self, graph, workers=4, verbose=40,
                 opt1_reduce_len=True,
                 opt2_reduce_sim_calc=True,
                 opt3_num_layers=None,
                 reuse=False,
                 temp_path="temp_struc2vec/",
                 topk=20,
                 rw_method="node2vec",
                 p=0.25, q=2.0):
        """
         (GPU + Euclidean Cost + Log Scaling + True Node2Vec)
        """
        self.graph = graph
        self.idx2node, self.node2idx = preprocess_nxgraph(graph)
        self.idx = list(range(len(self.idx2node)))
        self.temp_path = temp_path
        self.rw_method = rw_method
        self.topk = int(topk)

        self.p = p
        self.q = q

        self.neigh_cache = {u: list(self.graph[u]) for u in self.graph.nodes()}
        self.neigh_set_cache = {u: set(self.graph[u]) for u in self.graph.nodes()}
        self.deg_cache = {u: len(self.neigh_cache[u]) for u in self.graph.nodes()}

        n = len(self.graph.nodes())

        # فقط squirrel (5201 نود) و pubmed (19717 نود) بالای 5000 هستند؛
        # سایر دیتاست‌ها دقیقاً با همان تنظیمات و تابع قبلی اجرا می‌شوند.
        if n <= 5000:
            self.num_walks, self.walk_length = 10, 25
        else:
            self.num_walks, self.walk_length = 6, 15

        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        if opt3_num_layers is None:
            self.opt3_num_layers = int(math.log(n))
        else:
            self.opt3_num_layers = opt3_num_layers

        self.phi_total = defaultdict(lambda: defaultdict(float))

        if n > 5000:
            self.compute_structural_distance_gpu_candidate(self.opt3_num_layers)
        else:
            pair_distances = self.compute_structural_distance_gpu_full(self.opt3_num_layers)
            for layer in pair_distances:
                if layer == 0: continue
                for u in pair_distances[layer]:
                    for v, dist in pair_distances[layer][u].items():
                        self.phi_total[u][v] += dist

        self.layers_adj, self.layers_distances = self.get_cumulative_rep()

    def deepwalk_walk(self, start_node, walk_length):
        walk = [start_node]
        cur = start_node
        for _ in range(walk_length - 1):
            nbrs = self.neigh_cache[cur]
            if not nbrs: break
            cur = random.choice(nbrs)
            walk.append(cur)
        return walk

    def rwr_walk(self, start_node, walk_length, restart_prob=0.3):
        walk = [start_node]
        cur = start_node
        for _ in range(walk_length - 1):
            if random.random() < restart_prob:
                cur = start_node
            else:
                nbrs = self.neigh_cache[cur]
                if not nbrs:
                    cur = start_node
                else:
                    cur = random.choice(nbrs)
            walk.append(cur)
        return walk

    def walklets_walk(self, start_node, walk_length):
        base = self.deepwalk_walk(start_node, walk_length * 2)
        return base[::2]

    def node2vec_walk(self, start_node, walk_length):
        walk = [start_node]
        cur = start_node
        prev = None
        for _ in range(walk_length - 1):
            nbrs = self.neigh_cache[cur]
            if not nbrs: break

            if prev is None:
                nxt = random.choice(nbrs)
            else:
                prev_nbrs = self.neigh_set_cache[prev]
                weights = []
                for cand in nbrs:
                    if cand == prev:
                        weights.append(1.0 / self.p)
                    elif cand in prev_nbrs:
                        weights.append(1.0)
                    else:
                        weights.append(1.0 / self.q)

                nxt = random.choices(nbrs, weights=weights, k=1)[0]

            walk.append(nxt)
            prev, cur = cur, nxt

        return walk

    def non_backtracking_walk(self, start_node, walk_length):
        walk = [start_node]
        cur = start_node
        prev = None
        for _ in range(walk_length - 1):
            nbrs = self.neigh_cache[cur]
            if not nbrs: break
            if prev is None or len(nbrs) == 1:
                nxt = random.choice(nbrs)
            else:
                nxt = random.choice(nbrs)
                tries = 0
                while nxt == prev and tries < 3:
                    nxt = random.choice(nbrs)
                    tries += 1
            walk.append(nxt)
            prev, cur = cur, nxt
        return walk

    def get_order_degreelist_node(self, root, max_num_layers=None):
        if max_num_layers is None: max_num_layers = self.num_walks - 1
        ordered_dict = {}
        for level in range(max_num_layers + 1):
            if self.rw_method == "node2vec":
                walk = self.node2vec_walk(root, self.walk_length)
            elif self.rw_method == "walklets":
                walk = self.walklets_walk(root, self.walk_length)
            elif self.rw_method == "rwr":
                walk = self.rwr_walk(root, self.walk_length)
            elif self.rw_method == "non_backtracking":
                walk = self.non_backtracking_walk(root, self.walk_length)
            else:
                walk = self.deepwalk_walk(root, self.walk_length)
            ordered_dict[level] = sorted([self.deg_cache[node] for node in walk])
        return ordered_dict, max_num_layers

    def compute_structural_distance_gpu_full(self, max_num_layers):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_nodes = len(self.idx2node)
        degree_sequences = {node: self.get_order_degreelist_node(node, max_num_layers)[0] for node in
                            self.graph.nodes()}
        all_layer_dists = {}
        w_band = 50

        for layer in range(max_num_layers + 1):
            print(f"   [GPU-Scan] Order {layer} Processing (Euclidean + LogBoost)...")
            seqs = [degree_sequences[self.idx2node[i]][layer] for i in range(n_nodes)]
            seq_tensor = torch.log1p(torch.tensor(seqs, dtype=torch.float32, device=device))

            N, L = seq_tensor.shape
            dist_matrix = torch.zeros((N, N), device=device)
            batch_size = 128

            for i in range(0, N, batch_size):
                end_i = min(i + batch_size, N)
                batch_u = seq_tensor[i:end_i].unsqueeze(1)
                dp = torch.full((end_i - i, N, L + 1, L + 1), float('inf'), device=device)
                dp[:, :, 0, 0] = 0
                for r in range(1, L + 1):
                    for c in range(1, L + 1):
                        if abs(r - c) > w_band: continue
                        du, dv = batch_u[:, :, r - 1], seq_tensor[:, c - 1]
                        cost = torch.abs(du - dv)
                        prev_min = torch.min(
                            torch.min(dp[:, :, r - 1, c], dp[:, :, r, c - 1]),
                            dp[:, :, r - 1, c - 1]
                        )
                        dp[:, :, r, c] = cost + prev_min
                dist_matrix[i:end_i, :] = dp[:, :, L, L]

            dist_cpu = dist_matrix.cpu().numpy()
            layer_dist = {}
            for idx_u in range(N):
                u = self.idx2node[idx_u]
                layer_dist[u] = {self.idx2node[idx_v]: float(dist_cpu[idx_u, idx_v]) for idx_v in range(N) if
                                 idx_u != idx_v}
            all_layer_dists[layer] = layer_dist
        return all_layer_dists

    def compute_structural_distance_gpu_candidate(self, max_num_layers):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        n_nodes = len(self.idx2node)
        degree_sequences = {node: self.get_order_degreelist_node(node, max_num_layers)[0] for node in
                            self.graph.nodes()}
        w_band = 50
        K_WINDOW = min(1000, n_nodes)

        for layer in range(max_num_layers + 1):
            print(f"   [GPU-Flash] Order {layer} | Accelerating Large Graphs...")
            if layer == 0:
                continue

            seqs_np = np.array([degree_sequences[self.idx2node[i]][layer] for i in range(n_nodes)], dtype=np.float32)
            means = np.mean(seqs_np, axis=1)
            sorted_indices = np.argsort(means)

            seq_tensor = torch.log1p(torch.from_numpy(seqs_np).to(device))
            N, L = seq_tensor.shape

            batch_size = 512
            for i in range(0, N, batch_size):
                end_i = min(i + batch_size, N)
                curr_batch_indices = sorted_indices[i:end_i]
                B = len(curr_batch_indices)

                win_starts = np.clip(np.arange(i, end_i) - K_WINDOW // 2, 0, N - K_WINDOW)
                cand_idx_matrix = np.lib.stride_tricks.as_strided(
                    sorted_indices,
                    shape=(N - K_WINDOW + 1, K_WINDOW),
                    strides=(sorted_indices.strides[0], sorted_indices.strides[0])
                )[win_starts]

                u_seqs = seq_tensor[curr_batch_indices].unsqueeze(1)
                v_seqs = seq_tensor[cand_idx_matrix]

                cost_matrix = torch.abs(u_seqs.unsqueeze(3) - v_seqs.unsqueeze(2))

                dp = torch.full((B, K_WINDOW, L + 1, L + 1), float('inf'), device=device)
                dp[:, :, 0, 0] = 0.0

                for r in range(1, L + 1):
                    c_start = max(1, r - w_band)
                    c_end = min(L + 1, r + w_band + 1)
                    for c in range(c_start, c_end):
                        prev_min = torch.minimum(
                            torch.minimum(dp[:, :, r - 1, c], dp[:, :, r, c - 1]),
                            dp[:, :, r - 1, c - 1]
                        )
                        dp[:, :, r, c] = cost_matrix[:, :, r - 1, c - 1] + prev_min

                dists_batch = dp[:, :, L, L].cpu().numpy()

                top_m = min(max(self.topk * 3, 60), K_WINDOW)
                part_idx = np.argpartition(dists_batch, top_m - 1, axis=1)[:, :top_m]

                for b_idx in range(B):
                    u_idx = curr_batch_indices[b_idx]
                    u_node = self.idx2node[u_idx]
                    u_dict = self.phi_total[u_node]
                    sel_k = part_idx[b_idx]
                    cands = cand_idx_matrix[b_idx, sel_k]
                    dists = dists_batch[b_idx, sel_k]
                    for c_idx, d_val in zip(cands, dists):
                        if c_idx != u_idx:
                            u_dict[self.idx2node[c_idx]] += float(d_val)

        torch.cuda.empty_cache()
        return {}

    def get_cumulative_rep(self):
        adj, weights = defaultdict(list), defaultdict(list)
        for u in self.phi_total:
            sorted_nodes = sorted(self.phi_total[u].items(), key=lambda x: x[1])
            for v, d in sorted_nodes[:self.topk]:
                adj[u].append(v)
                weights[u].append(d)
        return {0: adj}, {0: weights}

    def get_similarity_weigh(self):
        similarity, weigh_list = defaultdict(set), {}
        neighbors, dists = self.layers_adj[0], self.layers_distances[0]
        all_vals = [d for u in dists for d in dists[u]]
        if not all_vals: return {}, {}

        ds_min = max(np.percentile(all_vals, 5), 1e-4)

        for u, neighs in neighbors.items():
            ws = dists[u]
            for v, wd in zip(neighs, ws):
                w = np.exp(-(wd / ds_min))
                if w > 0.00367:
                    similarity[u].add(v)
                    similarity[v].add(u)
                    weigh_list[(u, v)] = weigh_list[(v, u)] = float(w)
        return {u: list(vs) for u, vs in similarity.items()}, weigh_list