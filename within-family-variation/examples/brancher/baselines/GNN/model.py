"""
Bipartite GCNN branching policy, following the architecture published in
Gasse et al. 2019 ("Exact Combinatorial Optimization with Graph Convolutional
Neural Networks"): separate embeddings for variable and constraint nodes, one
round of bipartite message passing (constraint->variable, then
variable->constraint), and a final scoring MLP over variable embeddings.

No graph library dependency (e.g. PyTorch Geometric) -- message passing is
implemented directly with scatter-add over the edge list, since the graph
structure here (two node types, one edge type) is simple enough not to need one.

NOTE: the paper this baseline reproduces (LLM4Branch) does not itself specify
exact layer counts/dimensions for its GNN baseline -- it cites Gasse et al.
2019's official implementation and default settings. This is our best-effort
PyTorch port of that published architecture, not a verified line-for-line match.
"""
import torch
import torch.nn as nn

VAR_FEATURES = 19
CONS_FEATURES = 5
EDGE_FEATURES = 1
EMB_SIZE = 64


def mlp(in_dim, out_dim, hidden=EMB_SIZE):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
        nn.ReLU(),
    )


class BipartiteConv(nn.Module):
    """One half-convolution: propagates messages along edges from a source
    node type to a target node type, aggregating by sum per target node."""

    def __init__(self):
        super().__init__()
        self.message_mlp = mlp(2 * EMB_SIZE + EDGE_FEATURES, EMB_SIZE)
        self.update_mlp = mlp(2 * EMB_SIZE, EMB_SIZE)

    def forward(self, source_emb, target_emb, edge_index_source, edge_index_target, edge_features, n_target):
        messages_in = torch.cat([
            source_emb[edge_index_source],
            target_emb[edge_index_target],
            edge_features,
        ], dim=-1)
        messages = self.message_mlp(messages_in)
        aggregated = torch.zeros(n_target, EMB_SIZE, device=target_emb.device, dtype=target_emb.dtype)
        aggregated.index_add_(0, edge_index_target, messages)
        return self.update_mlp(torch.cat([target_emb, aggregated], dim=-1))


class GNNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.var_embedding = mlp(VAR_FEATURES, EMB_SIZE)
        self.cons_embedding = mlp(CONS_FEATURES, EMB_SIZE)
        self.cons_to_var = BipartiteConv()
        self.var_to_cons = BipartiteConv()
        self.output_mlp = nn.Sequential(
            nn.Linear(EMB_SIZE, EMB_SIZE),
            nn.ReLU(),
            nn.Linear(EMB_SIZE, 1),
        )

    def forward(self, variable_features, row_features, edge_indices, edge_values):
        var_emb = self.var_embedding(variable_features)
        cons_emb = self.cons_embedding(row_features)

        cons_idx, var_idx = edge_indices[0], edge_indices[1]
        edge_feat = edge_values.unsqueeze(-1)

        cons_emb = self.var_to_cons(
            source_emb=var_emb, target_emb=cons_emb,
            edge_index_source=var_idx, edge_index_target=cons_idx,
            edge_features=edge_feat, n_target=cons_emb.shape[0],
        )
        var_emb = self.cons_to_var(
            source_emb=cons_emb, target_emb=var_emb,
            edge_index_source=cons_idx, edge_index_target=var_idx,
            edge_features=edge_feat, n_target=var_emb.shape[0],
        )

        return self.output_mlp(var_emb)
