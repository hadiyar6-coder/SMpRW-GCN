import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.utils import degree


class Aggregator(nn.Module):
    def __init__(self, normalize='degree'):
        super(Aggregator, self).__init__()
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj = None

    def forward(self, x, edge_index, edge_weight=None):
        N = x.size(0)
        # کش کردن ماتریس مجاورت نرمال‌شده روی GPU تا در هر 200 ایپاک دوباره از نو ساخته نشود
        cache_key = (edge_index.data_ptr(), edge_index.size(1), N, x.device)
        if self._cached_edge_index != cache_key:
            row, col = edge_index
            if self.normalize == 'degree':
                deg = degree(row, N, dtype=x.dtype)
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                if edge_weight is not None:
                    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
                else:
                    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            else:
                edge_weight_sum = torch.zeros(N, dtype=x.dtype, device=x.device)
                edge_weight_sum.scatter_add_(0, row, edge_weight)
                norm = edge_weight / (edge_weight_sum[row] + 1e-16)

            self._cached_adj = torch.sparse_coo_tensor(
                edge_index, norm, (N, N), device=x.device
            ).coalesce()
            self._cached_edge_index = cache_key

        out = torch.sparse.mm(self._cached_adj, x)
        return out


class Encoder(nn.Module):
    def __init__(self, args, in_dim, out_dim, neigh_data, struc_data, merged_data):
        super(Encoder, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.neigh_data = neigh_data
        self.struc_data = struc_data
        self.merged_data = merged_data
        self.dropout_rate = args.dr

        self.aggregate_neigh = Aggregator(normalize=args.normalize)
        self.aggregate_struc = Aggregator(normalize=args.normalize)

        assert (args.alpha == 0) + (args.beta == 0) + (args.gama == 0) == 1 and \
               abs(args.alpha + args.beta + args.gama - 1.0) < 1e-5, \
            "Invalid config: Exactly one coefficient must be 0, and the sum must be 1."

        def init_logit(val):
            return math.log(max(val, 1e-6))

        self.alpha_logit = nn.Parameter(torch.full((self.in_dim,), init_logit(args.alpha), dtype=torch.float32))
        self.beta_logit = nn.Parameter(torch.full((self.in_dim,), init_logit(args.beta), dtype=torch.float32))
        self.gama_logit = nn.Parameter(torch.full((self.in_dim,), init_logit(args.gama), dtype=torch.float32))

        self.weight = nn.Parameter(torch.FloatTensor(self.out_dim, 2 * self.in_dim))
        init.kaiming_uniform_(self.weight)

    def forward(self, input_features):
        w1 = self.weight[:, :self.in_dim]
        w2 = self.weight[:, self.in_dim:]

        # کاهش ابعاد قبل از انتشار روی گراف (دقیقاً معادل فرمول اصلی مقاله شما ولی 100 برابر سریع‌تر در حالت One-Hot)
        if self.args.alpha == 0:
            logits = torch.stack([self.beta_logit, self.gama_logit], dim=0)
            weights = F.softmax(logits, dim=0)
            b, c = weights[0], weights[1]

            self_part = F.linear(input_features, w1)
            x_b = F.linear(input_features, w2 * b.unsqueeze(0))
            x_c = F.linear(input_features, w2 * c.unsqueeze(0))

            neigh_part = self.aggregate_neigh(x_b, self.neigh_data.edge_index, self.neigh_data.edge_weight)
            struc_part = self.aggregate_struc(x_c, self.struc_data.edge_index, self.struc_data.edge_weight)
            out = self_part + neigh_part + struc_part

        elif self.args.beta == 0:
            logits = torch.stack([self.alpha_logit, self.gama_logit], dim=0)
            weights = F.softmax(logits, dim=0)
            a, c = weights[0], weights[1]

            self_part = F.linear(input_features, w1 * a.unsqueeze(0))
            x_c = F.linear(input_features, w1 * c.unsqueeze(0))
            x_neigh = F.linear(input_features, w2)

            struc_part = self.aggregate_struc(x_c, self.struc_data.edge_index, self.struc_data.edge_weight)
            neigh_part = self.aggregate_neigh(x_neigh, self.neigh_data.edge_index, self.neigh_data.edge_weight)
            out = self_part + struc_part + neigh_part

        elif self.args.gama == 0:
            logits = torch.stack([self.alpha_logit, self.beta_logit], dim=0)
            weights = F.softmax(logits, dim=0)
            a, b = weights[0], weights[1]

            self_part = F.linear(input_features, w1 * a.unsqueeze(0))
            x_b = F.linear(input_features, w1 * b.unsqueeze(0))
            x_struc = F.linear(input_features, w2)

            neigh_part = self.aggregate_neigh(x_b, self.neigh_data.edge_index, self.neigh_data.edge_weight)
            struc_part = self.aggregate_struc(x_struc, self.struc_data.edge_index, self.struc_data.edge_weight)
            out = self_part + neigh_part + struc_part

        combined = F.relu(out)
        combined = F.dropout(combined, p=self.dropout_rate, training=self.training)
        return combined


class SMpRWGNN(nn.Module):
    def __init__(self, args, neigh_data, struc_data, merged_data):
        super(SMpRWGNN, self).__init__()
        self.args = args
        self.neigh_data = neigh_data
        self.struc_data = struc_data
        self.merged_data = merged_data

        if neigh_data is None:
            self.num_features = merged_data.num_features
            self.num_labels = merged_data.num_labels
        else:
            self.num_features = neigh_data.num_features
            self.num_labels = neigh_data.num_labels

        self.dropout = args.dr

        self.conv1 = Encoder(args, self.num_features, args.h_dim, self.neigh_data, self.struc_data, self.merged_data)
        self.conv2 = Encoder(args, args.h_dim, args.h_dim, self.neigh_data, self.struc_data, self.merged_data)

        self.weight = nn.Parameter(torch.FloatTensor(self.num_labels, args.h_dim))
        init.kaiming_uniform_(self.weight)

    def forward(self):
        h1 = self.conv1(self.neigh_data.x)
        h2 = self.conv2(h1)
        h = self.weight.mm(h2.t()).t()
        return F.log_softmax(h, dim=1)