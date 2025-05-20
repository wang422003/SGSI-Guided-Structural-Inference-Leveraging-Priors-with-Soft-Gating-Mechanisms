import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
from utils import my_softmax, get_offdiag_indices, gumbel_softmax
from parametric_adj_utils import LearnedAdj

_EPS = 1e-10


def edge2node4gating(edges, rel_rec):
    """
    edges: [b, E, feats]
    rel_rec: [E, N]
    => node features aggregated from incoming edges => [b, N, feats]
    """
    # typically: sum or average edges that go into each node
    b, E, f = edges.size()
    _, N = rel_rec.size()
    # shape => [b, f, E]
    edges_t = edges.transpose(1, 2)
    rel_rec_b = rel_rec.t().unsqueeze(0).to(edges.device)  # [1, N, E]

    # bmm => [b, f, E] x [b, E, N] => [b, f, N]
    # but we need to transpose rel_rec_b => shape [b, E, N]
    rel_rec_b = rel_rec_b.transpose(1, 2)  # => [1, E, N]
    rel_rec_b = rel_rec_b.expand(b, -1, -1)  # => [b, E, N]

    node_in = torch.bmm(edges_t, rel_rec_b)  # => [b, f, N]
    node_in = node_in.transpose(1,2)        # => [b, N, f]

    # average or sum
    denom = rel_rec.sum(0, keepdim=True).unsqueeze(0).transpose(1, 2)  # shape [1, N, 1]
    node_in = node_in / (denom + 1e-8)    # average
    return node_in


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead=4, ff_dim=256, dropout=0.0, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=F.silu,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # No mask by default, but you can add masks if needed
        return self.transformer_encoder(x)



class MLPEdges(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0., max_pool=True):
        super(MLPEdges, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        if max_pool:
            self.pool = nn.MaxPool1d(2)
        else:
            self.pool = nn.AvgPool1d(2)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        # print("====input of batchnorm: {}".format(inputs.size()))
        x = inputs.view(inputs.size()[0] * inputs.size()[1], -1)
        # print("====input after view: {}".format(x.size()))
        x = self.bn(x)
        return x.view(inputs.size()[0], inputs.size()[1], -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        x = self.pool(x)
        return self.batch_norm(x)


class CNN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                                 dilation=1, return_indices=False,
                                 ceil_mode=False)

        self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        # Input shape: [num_sims * num_edges, num_dims, num_timesteps]

        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        pred = self.conv_predict(x)
        attention = my_softmax(self.conv_attention(x), axis=2)

        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob


class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(MLPEncoder, self).__init__()

        self.factor = factor

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        # print("Input shape: {}".format(inputs.size()))
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        # print("x shape: {}".format(x.size()))
        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)


def node2edge(x, rel_rec, rel_send):
    # x: [b, N, d_model]
    # rel_rec: [E, N], rel_send: [E, N]
    # receivers: [b, E, d_model], senders: [b, E, d_model]
    receivers = torch.matmul(rel_rec, x)   # [E, N]*[b, N, d_model] = [b, E, d_model]
    senders = torch.matmul(rel_send, x)
    edges = torch.cat([senders, receivers], dim=2)  # [b, E, 2*d_model]
    return edges


def node2edge_with_gating_tr(node_emb, rel_rec, rel_send, gating):
    """
    node_emb: [b, N, feats]
    rel_rec, rel_send: [E, N]
    gating: [E] in [0,1]
    => edges: [b, E, feats*2], scaled by gating
    """
    b, N, feats = node_emb.shape
    E = rel_rec.size(0)
    device = node_emb.device

    # We'll do a batched approach for senders & receivers
    node_emb_t = node_emb.transpose(1,2)  # => [b, feats, N]

    # Build rel_send_b => [b, N, E]
    rel_send_b = rel_send.t().unsqueeze(0).expand(b, -1, -1).to(device)  # => [b, N, E]
    senders_b = torch.bmm(node_emb_t, rel_send_b)  # => [b, feats, E]
    senders_b = senders_b.transpose(1,2)          # => [b, E, feats]

    # Build rel_rec_b => [b, N, E]
    rel_rec_b = rel_rec.t().unsqueeze(0).expand(b, -1, -1).to(device)
    receivers_b = torch.bmm(node_emb_t, rel_rec_b)
    receivers_b = receivers_b.transpose(1,2)      # => [b, E, feats]

    edges_cat = torch.cat([senders_b, receivers_b], dim=2) # => [b, E, 2*feats]

    gating_ = gating.view(1, -1, 1).expand(b, -1, 2*feats)
    edges_masked = edges_cat * gating_
    return edges_masked


def node2edge_with_gating(x, rel_rec, rel_send, gating):
    """
    node_emb: [b, N, hid]
    rel_rec, rel_send: [E, N]
    gating: [E] in [0,1]
    => edges: [b, E, 2*hid], each scaled by gating[e]
    """
    # standard node2edge
    node_emb = x
    batch, N, feats = node_emb.shape
    E = rel_rec.size(0)

    # senders:
    # shape => [b, feats, E]
    node_emb_t = node_emb.transpose(1,2)  # [b, feats, N]
    # rel_send_b = rel_send.unsqueeze(0).to(node_emb.device)  # [1, E, N]
    rel_send_b = rel_send.unsqueeze(0).expand(batch, -1, -1).to(node_emb.device)  # [b, E, N]
    senders_b = torch.bmm(node_emb_t, rel_send_b.transpose(1,2))  # [b, feats, E]
    senders_b = senders_b.transpose(1,2)  # [b, E, feats]
    # let's define rel_rec_b up front
    # rel_rec_b = rel_rec.unsqueeze(0).to(node_emb.device)
    rel_rec_b = rel_rec.unsqueeze(0).expand(batch, -1, -1).to(node_emb.device)  # [b, E, N]

    # receivers:
    receivers_b = torch.bmm(node_emb_t, rel_rec_b.transpose(1,2))  # need rel_rec_b as well
    receivers_b = receivers_b.transpose(1,2)  # [b, E, feats]

    edges_cat = torch.cat([senders_b, receivers_b], dim=2)  # [b, E, feats*2]

    # gating => shape [E], broadcast => [1, E, 1]
    gating_ = gating.view(1, -1, 1)
    edges_masked = edges_cat * gating_  # scale each edge by gating[e]
    return edges_masked


def node2edge_bmm(node_emb, rel_rec, rel_send):
    """
    node_emb: [b, N, d_model]
    rel_rec, rel_send: [E, N]
    => edges: [b, E, 2*d_model]
    w. batch matmul
    """
    b, N, d = node_emb.shape
    E = rel_rec.size(0)

    rel_rec_batch = rel_rec.unsqueeze(0).expand(b, -1, -1)    # [b, E, N]
    rel_send_batch = rel_send.unsqueeze(0).expand(b, -1, -1)  # [b, E, N]

    receivers = torch.bmm(rel_rec_batch, node_emb)   # [b, E, d_model]
    senders   = torch.bmm(rel_send_batch, node_emb)  # [b, E, d_model]

    edges_cat = torch.cat([senders, receivers], dim=2)  # [b, E, 2*d_model]
    return edges_cat


class MLPEncoderWithLearnedAdj(nn.Module):
    def __init__(self, n_in, n_hid, n_out, n_atoms, factor=True, do_prob=0.0, init_val=-2.0):
        super().__init__()
        self.n_atoms = n_atoms
        self.factor = factor

        # The usual MLP layers
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)

        # Learned adjacency module
        self.learned_adj = LearnedAdj(n_atoms, init_val=init_val)  # from above

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # x: [b, E, feats]
        # rel_rec: [E, N], rel_send: [E, N]
        # Summation or average
        batch, E, feats = x.size()
        _, N = rel_rec.size()

        # reshape x => [batch, E, feats] => [batch, feats, E] for bmm with [E, N]
        x_trans = x.transpose(1, 2)  # shape [b, feats, E]
        rel_rec_b = rel_rec.unsqueeze(0).expand(batch, -1, -1).to(x.device)  # [b, E, N]

        # bmm => [b, feats, E] x [b, E, N] => [b, feats, N]
        incoming_b = torch.bmm(x_trans, rel_rec_b)  # shape [b, feats, N]
        incoming_b = incoming_b.transpose(1, 2)  # [b, N, feats]

        # you might average or sum
        incoming_b = incoming_b / (rel_rec.sum(0, keepdim=True).unsqueeze(0).transpose(1, 2) + 1e-8)  # average
        return incoming_b

    def forward(self, inputs, rel_rec, rel_send):
        """
        inputs: [batch, n_atoms, timesteps, dims]
        rel_rec, rel_send: [E, n_atoms]
        returns: logits => [batch, E, n_out]
        """
        # flatten time dims => [batch, n_atoms, in_feats]
        b, n_atoms, t, d = inputs.size()
        x = inputs.view(b, n_atoms, -1)

        # 1) node-level MLP
        x = self.mlp1(x)  # [b, N, hid]

        # 2) gating => [E]
        gating = self.learned_adj()

        # 3) node2edge with gating
        edges = node2edge_with_gating(x, rel_rec, rel_send, gating)  # [b, E, 2*hid]
        edges = self.mlp2(edges)
        x_skip = edges

        if self.factor:
            # edge2node => [b, N, hid]
            x = self.edge2node(edges, rel_rec, rel_send)
            x = self.mlp3(x)

            # node2edge again
            edges_2 = node2edge_with_gating(x, rel_rec, rel_send, gating)
            # skip connection
            edges_2 = torch.cat((edges_2, x_skip), dim=2)
            edges_2 = self.mlp4(edges_2)
            edges = edges_2
        else:
            edges_2 = self.mlp3(edges)
            edges_2 = torch.cat((edges_2, x_skip), dim=2)
            edges_2 = self.mlp4(edges_2)
            edges = edges_2

        out = self.fc_out(edges)  # => [b, E, n_out]
        return out, gating


class SmallTransformerEncoder(nn.Module):
    """
    A small wrapper around nn.TransformerEncoder for convenience.
    batch_first=True => input shape [batch, seq_len, d_model].
    """
    def __init__(self, d_model=128, nhead=4, ff_dim=256, dropout=0.0, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        """
        x: [batch, seq_len, d_model]
        mask: optional [batch, seq_len, seq_len] or [seq_len, seq_len]
        """
        # If you don't need a mask, pass mask=None
        return self.transformer_encoder(x, mask=mask)


#----------------------------------------------------
# Factor Graph with multiple node-edge iterations
# using Transformers for edges and nodes
#----------------------------------------------------
class FactorGraphTransformerEncoder(nn.Module):
    def __init__(
        self,
        node_in_dim,      # e.g., (timesteps * input_features)
        d_model=64,
        num_iterations=1, # how many (node->edge->node) steps
        nhead=4,
        ff_dim=256,
        dropout=0.0,
        factor=True,
        edge_out_dim=2    # final dimension for edge classification
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.factor = factor
        self.d_model = d_model

        # 1) Input projection for node embeddings
        self.node_in_fc = nn.Linear(node_in_dim, d_model)

        # 2) Edge projection from node2edge: we will do [sender, receiver] => 2*d_model
        #    Then reduce to d_model for edge embedding
        self.edge_in_fc = nn.Linear(2*d_model, d_model)

        # 3) After each edge update, we run an "EdgeTransformerBlock"
        #    We'll define a small transformer for edges
        self.edge_transform = SmallTransformerEncoder(
            d_model=d_model, nhead=nhead, ff_dim=ff_dim, dropout=dropout, num_layers=1
        )

        # 4) After edge2node, we run a "NodeTransformerBlock"
        self.node_transform = SmallTransformerEncoder(
            d_model=d_model, nhead=nhead, ff_dim=ff_dim, dropout=dropout, num_layers=1
        )

        # If factor=True, we do an extra skip connection merging, similar to NRI factor
        # We'll handle that in the forward pass logic

        # Final output for edges
        self.edge_out_fc = nn.Linear(d_model, edge_out_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.node_in_fc.weight)
        self.node_in_fc.bias.data.fill_(0.0)
        nn.init.xavier_normal_(self.edge_in_fc.weight)
        self.edge_in_fc.bias.data.fill_(0.0)
        nn.init.xavier_normal_(self.edge_out_fc.weight)
        self.edge_out_fc.bias.data.fill_(0.0)

    #---------------------------------------------
    # node2edge and edge2node operations
    #---------------------------------------------
    def node2edge(self, node_embeddings, rel_rec, rel_send):
        """
        node_embeddings: [b, N, d_model]
        rel_rec, rel_send: [E, N]
        -> edges: [b, E, 2*d_model]  => then we reduce to [b, E, d_model]
        """
        # receivers = rel_rec @ node_embeddings => [E, N] * [b, N, d_model]
        # but we must broadcast batch
        # We'll do a batch operation via bmm or matmul
        # Trick: If rel_rec is [E, N], we can do: receivers = torch.matmul(rel_rec, node_embeddings)
        # but that would be shape [E, d_model], missing batch dimension.
        # So we expand to [b, E, N], then bmm with [b, N, d_model].
        b, N, d = node_embeddings.shape
        E = rel_rec.size(0)

        # Expand rel_rec: [E, N] -> [b, E, N]
        rel_rec_batch = rel_rec.unsqueeze(0).expand(b, -1, -1)  # [b, E, N]
        receivers = torch.bmm(rel_rec_batch, node_embeddings)   # [b, E, d_model]

        rel_send_batch = rel_send.unsqueeze(0).expand(b, -1, -1)  # [b, E, N]
        senders = torch.bmm(rel_send_batch, node_embeddings)      # [b, E, d_model]

        # Concatenate
        edges_cat = torch.cat([senders, receivers], dim=2)  # [b, E, 2*d_model]
        return edges_cat

    def edge2node(self, edge_embeddings, rel_rec, rel_send):
        """
        edge_embeddings: [b, E, d_model]
        -> node embeddings: [b, N, d_model], where each node is the sum/avg of incoming edges
        """
        # Summation or average over edges that go into the node
        b, E, d = edge_embeddings.shape
        N = rel_rec.size(1)

        # rel_rec^T: [N, E], so we do [b, N, E] * [b, E, d] => [b, N, d]
        rel_rec_batch = rel_rec.t().unsqueeze(0).expand(b, -1, -1)  # [b, N, E]
        incoming = torch.bmm(rel_rec_batch, edge_embeddings)        # [b, N, d]

        # You can average or just sum. We'll do average as in NRI
        num_incoming = torch.sum(rel_rec, dim=0, keepdim=True)  # [1, E]
        # Expand to shape [b, N, 1], but that's trickier. Instead, we can do:
        # We'll do a quick approach: incoming / number_of_incoming_edges
        # For a fully connected graph with no self loops, each node has N-1 incoming edges.
        # But if factor graph is partial or if you want a robust approach:
        incoming = incoming / (num_incoming.unsqueeze(0).transpose(1,2) + 1e-8)  # avoid /0
        return incoming

    #---------------------------------------------
    # The forward pass with multiple iterations
    #---------------------------------------------
    def forward(self, inputs, rel_rec, rel_send):
        """
        inputs: [b, N, time * features]
        rel_rec, rel_send: adjacency definitions, shape [E, N]
        """
        # print("encoder inputs shape", inputs.shape)
        b, N, T_in, f = inputs.size()
        inputs = inputs.view(b, N, -1)  # [b, N, n_in]

        # 1) Project inputs to node embeddings
        node_embeddings = self.node_in_fc(inputs)  # [b, N, d_model]

        # Perform multiple node->edge->node iterations
        for iteration in range(self.num_iterations):
            # node2edge
            edges_cat = self.node2edge(node_embeddings, rel_rec, rel_send)  # [b, E, 2*d_model]

            # reduce dimension to d_model
            edges_in = self.edge_in_fc(edges_cat)  # [b, E, d_model]

            # Edge transformer - treat each edge as a token.
            # shape: [b, E, d_model], no mask by default
            edges_out = self.edge_transform(edges_in)  # [b, E, d_model]

            # skip connection if factor, or just override
            if self.factor:
                # combine edges_in with edges_out
                # for instance, concat or sum. We'll do concat:
                edges_out = torch.cat([edges_out, edges_in], dim=2)  # => [b, E, 2*d_model]
                # we could do another linear down to d_model
                edges_out = nn.Linear(2*self.d_model, self.d_model, bias=True).to(edges_out.device)(edges_out)

            # edge2node
            node_in = self.edge2node(edges_out, rel_rec, rel_send)  # [b, N, d_model]

            # Node transformer - treat each node as a token
            node_out = self.node_transform(node_in)

            # skip connection if desired
            if self.factor:
                node_out = torch.cat([node_out, node_in], dim=2)  # [b, N, 2*d_model]
                node_out = nn.Linear(2*self.d_model, self.d_model, bias=True).to(node_out.device)(node_out)

            node_embeddings = node_out

        # After final iteration, we produce edge predictions from the last edge embeddings
        # node2edge again
        edges_cat_final = self.node2edge(node_embeddings, rel_rec, rel_send)
        edges_in_final = self.edge_in_fc(edges_cat_final)
        # We could pass it again through a small transformer, or just output
        edges_final = self.edge_transform(edges_in_final)  # optional final pass
        edge_logits = self.edge_out_fc(edges_final)  # [b, E, out_dim]

        return edge_logits


class SelfAttentionNRIEncoderWithEdgeResidual(nn.Module):
    """
    Demonstrates:
    1) Node-level self-attention to create node embeddings.
    2) node2edge to create edge embeddings.
    3) Edge-level self-attention with a residual connection from the original edge embeddings.
    4) Final linear projection to produce edge logits.

    This is just one design; adapt as needed for iterative or more advanced message passing.
    """
    def __init__(
        self,
        n_in,           # e.g., time*features per node
        n_hid=128,
        n_out=2,        # number of edge types (binary edge example)
        nhead=4,
        ff_dim=256,
        dropout=0.0,
        num_node_layers=1,  # layers for node self-attn
        num_edge_layers=1   # layers for edge self-attn
    ):
        super().__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_out = n_out

        # 1) Project node inputs to n_hid
        self.node_in_fc = nn.Linear(n_in, n_hid)

        # 2) Node-level self-attention
        self.node_transform = SmallTransformerEncoder(
            d_model=n_hid,
            nhead=nhead,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=num_node_layers
        )

        # 3) Reduce [2*n_hid] => [n_hid] for edges
        self.edge_fc_in = nn.Linear(2*n_hid, n_hid)

        # 4) Edge-level self-attention
        self.edge_transform = SmallTransformerEncoder(
            d_model=n_hid,
            nhead=nhead,
            ff_dim=ff_dim,
            dropout=dropout,
            num_layers=num_edge_layers
        )

        # 5) Final linear layer for edge classification
        self.fc_out = nn.Linear(n_hid, n_out)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.node_in_fc.weight)
        self.node_in_fc.bias.data.fill_(0.0)

        nn.init.xavier_normal_(self.edge_fc_in.weight)
        self.edge_fc_in.bias.data.fill_(0.0)

        nn.init.xavier_normal_(self.fc_out.weight)
        self.fc_out.bias.data.fill_(0.0)

    def forward(self, inputs, rel_rec, rel_send):
        """
        inputs: [b, N, n_in] => node-level inputs
        rel_rec, rel_send: [E, N]

        Returns:
            edge_logits: [b, E, n_out]
        """
        # print("encoder inputs shape", inputs.shape)
        b, N, T_in, f = inputs.size()
        inputs = inputs.view(b, N, -1)  # [b, N, n_in]

        # 1) Project node inputs
        node_emb = self.node_in_fc(inputs)  # [b, N, n_hid]

        # 2) Node-level self-attn => updated node embeddings
        node_emb = self.node_transform(node_emb)  # [b, N, n_hid]

        # 3) node2edge => combine senders + receivers => [b, E, 2*n_hid]
        edges_cat = node2edge_bmm(node_emb, rel_rec, rel_send)

        # 4) Project edges_cat => edges_in
        edges_in = self.edge_fc_in(edges_cat)  # [b, E, n_hid]
        x_skip = edges_in.clone()              # save for residual

        # 5) Edge-level self-attn
        #    We'll treat each edge as a token: shape [b, E, n_hid]
        edges_out = self.edge_transform(edges_in)  # [b, E, n_hid]

        # 6) Residual connection
        #    We simply add the original edges_in to edges_out
        #    (In standard NRI factor-graph skip, we might concat and MLP, but "add" is also valid)
        edges_out = edges_out + x_skip

        # a variant for a similar implementation of NRI:
        # edges_out = torch.cat([edges_out, x_skip], dim=2)  # => [b, E, 2*n_hid]
        # edges_out = self.some_linear(edges_out)  # => [b, E, n_hid]
        # 7) Final linear
        edge_logits = self.fc_out(edges_out)  # [b, E, n_out]

        return edge_logits


class MLPEncoderWithGatingAndKnownEdges(nn.Module):
    """
    Combines:
      - Learned gating param for all edges
      - + partial known adjacency:
        known_present => clamp gating=1
        known_absent => clamp gating=0
    """
    def __init__(self, n_in, n_hid, n_out, n_atoms,
                 known_present, known_absent,
                 # rel_rec, rel_send,
                 factor=True, dropout=0.0, init_gating=-2.0):
        super().__init__()
        self.n_atoms = n_atoms
        self.factor = factor
        # self.rel_rec = rel_rec
        # self.rel_send = rel_send
        self.known_present = known_present  # set of (s, r)
        self.known_absent  = known_absent   # set of (s, r)

        self.mlp1 = MLP(n_in, n_hid, n_hid, dropout)
        self.mlp2 = MLP(n_hid*2, n_hid, n_hid, dropout)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, dropout)
        if factor:
            self.mlp4 = MLP(n_hid*3, n_hid, n_hid, dropout)
        else:
            self.mlp4 = MLP(n_hid*2, n_hid, n_hid, dropout)

        self.fc_out = nn.Linear(n_hid, n_out)

        # full adjacency gating
        self.learned_adj = LearnedAdj(n_atoms, init_val=init_gating)

        # Build an index map for e->(s,r)
        self.edge_index_map = []
        idx = 0
        for s in range(n_atoms):
            for r in range(n_atoms):
                if s != r:
                    self.edge_index_map.append((s, r))
                    idx += 1

    def forward(self, inputs, rel_rec, rel_send):
        """
        inputs: [batch, n_atoms, timesteps, dims]
        returns: logits => [batch, E, n_out], gating => [E]
        """
        b, N, t, d = inputs.size()
        x = inputs.view(b, N, -1)  # flatten time dims => [b, N, in_features]

        # 1) node-level
        x = self.mlp1(x)

        # 2) gating param => shape [E]
        gating_raw = self.learned_adj.gating_param
        gating = torch.sigmoid(gating_raw.clone())  # copy
        gating_clamped = gating.clone()
        # 3) Clamp known edges
        # if (s, r) in known_present => gating[e] = 1
        # if (s, r) in known_absent  => gating[e] = 0
        for e, (s, r) in enumerate(self.edge_index_map):
            if (s, r) in self.known_present:
                gating_clamped[e] = 1.0
            elif (s, r) in self.known_absent:
                gating_clamped[e] = 0.0

        # 4) node2edge with gating => mlp2
        edges = node2edge_with_gating_tr(x, rel_rec, rel_send, gating_clamped)
        edges = self.mlp2(edges)
        x_skip = edges

        if self.factor:
            # edge2node => mlp3 => node2edge => skip => mlp4
            x_node = edge2node4gating(edges, rel_rec)
            x_node = self.mlp3(x_node)
            edges_2 = node2edge_with_gating_tr(x_node, rel_rec, rel_send, gating_clamped)
            edges_2 = torch.cat((edges_2, x_skip), dim=2)
            edges_2 = self.mlp4(edges_2)
            edges = edges_2
        else:
            edges_2 = self.mlp3(edges)
            edges_2 = torch.cat((edges_2, x_skip), dim=2)
            edges_2 = self.mlp4(edges_2)
            edges = edges_2

        out = self.fc_out(edges)  # => [b, E, n_out]
        return out, gating


class SingleTransformerEncoder(nn.Module):
    def __init__(self, n_in, d_model, n_out, nhead=4, ff_dim=256, num_layers=1, dropout=0.0):
        super().__init__()
        self.d_model = d_model

        # Project inputs from n_in to d_model
        self.input_fc = nn.Linear(n_in, d_model)

        # A single transformer encoder over nodes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=F.silu,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # After transformer, we'll do node2edge and a final output layer
        self.fc_out = nn.Linear(d_model*2, n_out)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.input_fc.weight.data)
        self.input_fc.bias.data.fill_(0.1)
        nn.init.xavier_normal_(self.fc_out.weight.data)
        self.fc_out.bias.data.fill_(0.1)

    def forward(self, inputs, rel_rec, rel_send):
        # inputs: [b, N, T_in, feature_dims]
        # Flatten time and feature: [b, N, T_in * feature_dims]
        b, N, T_in, f = inputs.size()
        x = inputs.view(b, N, -1)  # [b, N, n_in]

        # Project to d_model
        x = self.input_fc(x)  # [b, N, d_model]

        # Apply transformer encoder on nodes
        # This models all interactions among nodes (fully connected)
        x = self.transformer_encoder(x)  # [b, N, d_model]

        # Convert node embeddings to edge embeddings
        edges = node2edge(x, rel_rec, rel_send)  # [b, E, 2*d_model]

        # Predict edge features (e.g., edge types)
        out = self.fc_out(edges)  # [b, E, n_out]

        return out



class MixSingleTransformerEncoder(nn.Module):
    def __init__(self, n_in, d_model, n_out, nhead=4, ff_dim=128, num_layers=3, dropout=0.0, feat_dim=4):
        super().__init__()
        self.d_model = d_model

        # Project inputs from n_in to d_model
        self.input_fc = nn.Linear(n_in, d_model)
        print("Number of transformer layers in encoder: ", num_layers)
        print("nhead: ", nhead)
        print("ff_dim: ", ff_dim)

        # A single transformer encoder over nodes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=F.silu,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # After transformer, we'll do node2edge and a final output layer
        self.fc_out = nn.Linear(d_model * 2 * feat_dim, n_out)

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.input_fc.weight.data)
        self.input_fc.bias.data.fill_(0.1)
        nn.init.xavier_normal_(self.fc_out.weight.data)
        self.fc_out.bias.data.fill_(0.1)

    def forward(self, inputs, rel_rec, rel_send):
        # inputs: [b, N, T_in, feature_dims]
        # Flatten time and feature: [b, N * feature_dims, T_in]
        b, N, T_in, f = inputs.size()
        x = inputs.view(b, -1, T_in)  # [b, N * feature_dims, T_in]

        # Project to d_model
        x = self.input_fc(x)  # [b, N * feature_dims, d_model]

        # Apply transformer encoder on nodes
        # This models all interactions among nodes (fully connected)
        x = self.transformer_encoder(x)  # [b, N * feature_dims, d_model]
        # print("X.shape after transformer: ", x.shape)
        x = x.view(b, N, -1)
        # Convert node embeddings to edge embeddings
        edges = node2edge(x, rel_rec, rel_send)  # [b, E, 2*d_model]
        # print("Edges.shape after node2edge: ", edges.shape)
        # Predict edge features (e.g., edge types)
        out = self.fc_out(edges)  # [b, E, n_out]

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True, nhead=4, ff_dim=256, num_layers=1):
        super(TransformerEncoder, self).__init__()
        self.factor = factor
        self.n_hid = n_hid

        # Instead of MLP1: transform node features
        # Input: [num_sims, num_atoms, features]
        # We'll map n_in→n_hid first:
        self.input_fc = nn.Linear(n_in, n_hid)
        # Replace MLP1 with a Transformer block operating at the node level:
        self.node_transform_1 = TransformerBlock(d_model=n_hid, nhead=nhead, ff_dim=ff_dim, dropout=do_prob, num_layers=num_layers)

        # node2edge produces edges: shape [num_sims, E, 2*n_hid]
        # Replace MLP2 with another Transformer that operates on edges:
        self.edge_transform_2 = TransformerBlock(d_model=n_hid, nhead=nhead, ff_dim=ff_dim, dropout=do_prob, num_layers=num_layers)
        # But we must handle the dimension change: we currently have sender/receiver concatenation.
        # Let's define a linear layer to go from 2*n_hid → n_hid before the Transformer:
        self.edge_fc_in = nn.Linear(n_hid*2, n_hid)

        # For factor=True:
        # After edge2node, we apply MLP3 on nodes again.
        # Replace MLP3 with another Transformer at node level
        self.node_transform_3 = TransformerBlock(d_model=n_hid, nhead=nhead, ff_dim=ff_dim, dropout=do_prob, num_layers=num_layers)

        # Then node2edge again, concatenate skip, and MLP4 on edges
        # Replace MLP4 with another Transformer on edges
        # We'll have a skip connection that concatenates x_skip (n_hid) and the new edge features (n_hid),
        # resulting in 2*n_hid or 3*n_hid if factor=True.
        # We'll need a linear to go back to n_hid before Transformer.
        if factor:
            print("Using factor graph Transformer encoder.")
            # For factor=True, after node→edge, we have x of dim n_hid and x_skip dim n_hid,
            # and also one more set: total 3*n_hid
            self.edge_fc_factor = nn.Linear(n_hid*3, n_hid)
        else:
            print("Using Transformer encoder (no factor).")
            self.edge_fc_factor = nn.Linear(n_hid*2, n_hid)

        self.edge_transform_4 = TransformerBlock(d_model=n_hid, nhead=nhead, ff_dim=ff_dim, dropout=do_prob, num_layers=num_layers)

        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.input_fc.weight.data)
        self.input_fc.bias.data.fill_(0.1)
        nn.init.xavier_normal_(self.fc_out.weight.data)
        self.fc_out.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        incoming = torch.matmul(rel_rec.t(), x)  # [E, b_hid] → [N, b_hid]
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        receivers = torch.matmul(rel_rec, x)   # [E, N]*[N, d_model] = [E, d_model]
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)  # [num_sims, E, 2*n_hid]
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # inputs: [num_sims, num_atoms, num_timesteps, num_dims]
        # reshape inputs to [num_sims, num_atoms, -1]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # map to n_hid
        x = self.input_fc(x)  # [num_sims, num_atoms, n_hid]
        # transformer on nodes
        x = self.node_transform_1(x)  # still [num_sims, num_atoms, n_hid]

        # node2edge
        edges = self.node2edge(x, rel_rec, rel_send)  # [num_sims, E, 2*n_hid]
        edges = self.edge_fc_in(edges)  # [num_sims, E, n_hid]
        edges = self.edge_transform_2(edges)

        x_skip = edges

        if self.factor:
            # edge2node
            nodes = self.edge2node(edges, rel_rec, rel_send)  # [num_sims, N, n_hid]
            nodes = self.node_transform_3(nodes)

            # node2edge again
            edges_new = self.node2edge(nodes, rel_rec, rel_send)  # [num_sims, E, 2*n_hid]
            # concatenate with x_skip: now we have edges_new (2*n_hid) + x_skip(n_hid) = 3*n_hid
            edges_cat = torch.cat([edges_new, x_skip], dim=2)  # [num_sims, E, 3*n_hid]
            edges_cat = self.edge_fc_factor(edges_cat)  # back to n_hid
            edges_cat = self.edge_transform_4(edges_cat)
            out = self.fc_out(edges_cat)

        else:
            # no factor
            edges_new = self.node2edge(x, rel_rec, rel_send)  # 2*n_hid
            edges_cat = torch.cat([edges_new, x_skip], dim=2) # [num_sims, E, 3*n_hid] if factor was used, else 2*n_hid
            edges_cat = self.edge_fc_factor(edges_cat)
            edges_cat = self.edge_transform_4(edges_cat)
            out = self.fc_out(edges_cat)

        return out


class CNNEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(CNNEncoder, self).__init__()
        self.dropout_prob = do_prob

        self.factor = factor

        self.cnn = CNN(n_in * 2, n_hid, n_hid, do_prob)
        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)

        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.

        x = inputs.view(inputs.size(0), inputs.size(1), -1)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(inputs.size(0) * receivers.size(1),
                                   inputs.size(2), inputs.size(3))
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.view(inputs.size(0) * senders.size(1),
                               inputs.size(2),
                               inputs.size(3))
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1) - 1) * inputs.size(1), -1)
        x = self.mlp1(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp2(x)

            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp3(x)

        return self.fc_out(x)


class GINEncoder(nn.Module):
    """
    Modified MLP encoder with invariance and other stuffs.
    """
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True, gin=True):
        super(GINEncoder, self).__init__()

        self.factor = factor
        self.gin = gin
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLPEdges(n_hid * 2, n_hid * 2, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLPEdges(n_hid * 3, n_hid * 2, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size()[1]

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_nodes, num_timesteps, num_dims]
        x = inputs.view(inputs.size()[0], inputs.size()[1], -1)
        # New shape: [num_sims, num_nodes, num_timesteps*num_dims]
        # print("x")
        # print(x.size())
        x = self.mlp1(x)  # 2-layer ELU net per node
        node_skip = x
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            if self.gin:
                x = x + node_skip
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)


class SimulationDecoder(nn.Module):
    """Simulation-based decoder."""

    def __init__(self, loc_max, loc_min, vel_max, vel_min, suffix):
        super(SimulationDecoder, self).__init__()

        self.loc_max = loc_max
        self.loc_min = loc_min
        self.vel_max = vel_max
        self.vel_min = vel_min

        self.interaction_type = suffix

        if '_springs' in self.interaction_type:
            print('Using spring simulation decoder.')
            self.interaction_strength = .1
            self.sample_freq = 1
            self._delta_T = 0.1
            self.box_size = 5.
        elif '_charged' in self.interaction_type:
            print('Using charged particle simulation decoder.')
            self.interaction_strength = 1.
            self.sample_freq = 100
            self._delta_T = 0.001
            self.box_size = 5.
        elif '_charged_short' in self.interaction_type:
            print('Using charged particle simulation decoder.')
            self.interaction_strength = .1
            self.sample_freq = 10
            self._delta_T = 0.001
            self.box_size = 1.
        else:
            print("Simulation type could not be inferred from suffix.")

        self.out = None

        # NOTE: For exact reproduction, choose sample_freq=100, delta_T=0.001

        self._max_F = 0.1 / self._delta_T

    def unnormalize(self, loc, vel):
        loc = 0.5 * (loc + 1) * (self.loc_max - self.loc_min) + self.loc_min
        vel = 0.5 * (vel + 1) * (self.vel_max - self.vel_min) + self.vel_min
        return loc, vel

    def renormalize(self, loc, vel):
        loc = 2 * (loc - self.loc_min) / (self.loc_max - self.loc_min) - 1
        vel = 2 * (vel - self.vel_min) / (self.vel_max - self.vel_min) - 1
        return loc, vel

    def clamp(self, loc, vel):
        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        vel[over] = -torch.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        vel[under] = torch.abs(vel[under])

        return loc, vel

    def set_diag_to_zero(self, x):
        """Hack to set diagonal of a tensor to zero."""
        mask = torch.diag(torch.ones(x.size(1))).unsqueeze(0).expand_as(x)
        inverse_mask = torch.ones(x.size(1), x.size(1)) - mask
        if x.is_cuda:
            inverse_mask = inverse_mask.cuda()
        inverse_mask = Variable(inverse_mask)
        return inverse_mask * x

    def set_diag_to_one(self, x):
        """Hack to set diagonal of a tensor to one."""
        mask = torch.diag(torch.ones(x.size(1))).unsqueeze(0).expand_as(x)
        inverse_mask = torch.ones(x.size(1), x.size(1)) - mask
        if x.is_cuda:
            mask, inverse_mask = mask.cuda(), inverse_mask.cuda()
        mask, inverse_mask = Variable(mask), Variable(inverse_mask)
        return mask + inverse_mask * x

    def pairwise_sq_dist(self, x):
        xx = torch.bmm(x, x.transpose(1, 2))
        rx = (x ** 2).sum(2).unsqueeze(-1).expand_as(xx)
        return torch.abs(rx.transpose(1, 2) + rx - 2 * xx)

    def forward(self, inputs, relations, rel_rec, rel_send, pred_steps=1):
        # Input has shape: [num_sims, num_things, num_timesteps, num_dims]
        # Relation mx shape: [num_sims, num_things*num_things]

        # Only keep single dimension of softmax output
        relations = relations[:, :, 1]

        loc = inputs[:, :, :-1, :2].contiguous()
        vel = inputs[:, :, :-1, 2:].contiguous()

        # Broadcasting/shape tricks for parallel processing of time steps
        loc = loc.permute(0, 2, 1, 3).contiguous()
        vel = vel.permute(0, 2, 1, 3).contiguous()
        loc = loc.view(inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2)
        vel = vel.view(inputs.size(0) * (inputs.size(2) - 1), inputs.size(1), 2)

        loc, vel = self.unnormalize(loc, vel)

        offdiag_indices = get_offdiag_indices(inputs.size(1))
        edges = Variable(torch.zeros(relations.size(0), inputs.size(1) *
                                     inputs.size(1)))
        if inputs.is_cuda:
            edges = edges.cuda()
            offdiag_indices = offdiag_indices.cuda()

        edges[:, offdiag_indices] = relations.float()

        edges = edges.view(relations.size(0), inputs.size(1),
                           inputs.size(1))

        self.out = []

        for _ in range(0, self.sample_freq):
            x = loc[:, :, 0].unsqueeze(-1)
            y = loc[:, :, 1].unsqueeze(-1)

            xx = x.expand(x.size(0), x.size(1), x.size(1))
            yy = y.expand(y.size(0), y.size(1), y.size(1))
            dist_x = xx - xx.transpose(1, 2)
            dist_y = yy - yy.transpose(1, 2)

            if '_springs' in self.interaction_type:
                forces_size = -self.interaction_strength * edges
                pair_dist = torch.cat(
                    (dist_x.unsqueeze(-1), dist_y.unsqueeze(-1)),
                    -1)

                # Tricks for parallel processing of time steps
                pair_dist = pair_dist.view(inputs.size(0), (inputs.size(2) - 1),
                                           inputs.size(1), inputs.size(1), 2)
                forces = (
                        forces_size.unsqueeze(-1).unsqueeze(1) * pair_dist).sum(
                    3)
            else:  # charged particle sim
                e = (-1) * (edges * 2 - 1)
                forces_size = -self.interaction_strength * e

                l2_dist_power3 = torch.pow(self.pairwise_sq_dist(loc), 3. / 2.)
                l2_dist_power3 = self.set_diag_to_one(l2_dist_power3)

                l2_dist_power3 = l2_dist_power3.view(inputs.size(0),
                                                     (inputs.size(2) - 1),
                                                     inputs.size(1),
                                                     inputs.size(1))
                forces_size = forces_size.unsqueeze(1) / (l2_dist_power3 + _EPS)

                pair_dist = torch.cat(
                    (dist_x.unsqueeze(-1), dist_y.unsqueeze(-1)),
                    -1)
                pair_dist = pair_dist.view(inputs.size(0), (inputs.size(2) - 1),
                                           inputs.size(1), inputs.size(1), 2)
                forces = (forces_size.unsqueeze(-1) * pair_dist).sum(3)

            forces = forces.view(inputs.size(0) * (inputs.size(2) - 1),
                                 inputs.size(1), 2)

            if '_charged' in self.interaction_type:  # charged particle sim
                # Clip forces
                forces[forces > self._max_F] = self._max_F
                forces[forces < -self._max_F] = -self._max_F

            # Leapfrog integration step
            vel = vel + self._delta_T * forces
            loc = loc + self._delta_T * vel

            # Handle box boundaries
            loc, vel = self.clamp(loc, vel)

        loc, vel = self.renormalize(loc, vel)

        loc = loc.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)
        vel = vel.view(inputs.size(0), (inputs.size(2) - 1), inputs.size(1), 2)

        loc = loc.permute(0, 2, 1, 3)
        vel = vel.permute(0, 2, 1, 3)

        out = torch.cat((loc, vel), dim=-1)
        # Output has shape: [num_sims, num_things, num_timesteps-1, num_dims]

        return out


class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                 rel_type.size(2)]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        # assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 curr_rel_type)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class RNNDecoder(nn.Module):
    """Recurrent decoder module."""

    def __init__(self, n_in_node, edge_types, n_hid,
                 do_prob=0., skip_first=False):
        super(RNNDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, inputs, rel_rec, rel_send,
                            rel_type, hidden):

        # node2edge
        receivers = torch.matmul(rel_rec, hidden)
        senders = torch.matmul(rel_send, hidden)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        self.msg_out_shape))
        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1.
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2,
                                                                        -1)
        agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average

        # GRU-style gated aggregation
        r = F.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
        i = F.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
        n = F.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        pred = inputs + pred

        return pred, hidden

    def forward(self, data, rel_type, rel_rec, rel_send, pred_steps=1,
                burn_in=False, burn_in_steps=1, dynamic_graph=False,
                encoder=None, temp=None):

        inputs = data.transpose(1, 2).contiguous()

        time_steps = inputs.size(1)

        # inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # rel_type has shape:
        # [batch_size, num_atoms*(num_atoms-1), num_edge_types]

        hidden = Variable(
            torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape))
        if inputs.is_cuda:
            hidden = hidden.cuda()

        pred_all = []

        for step in range(0, inputs.size(1) - 1):

            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]
            else:
                assert (pred_steps <= time_steps)
                # Use ground truth trajectory input vs. last prediction
                if not step % pred_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]

            if dynamic_graph and step >= burn_in_steps:
                # NOTE: Assumes burn_in_steps = args.timesteps
                logits = encoder(
                    data[:, :, step - burn_in_steps:step, :].contiguous(),
                    rel_rec, rel_send)
                rel_type = gumbel_softmax(logits, tau=temp, hard=True)

            pred, hidden = self.single_step_forward(ins, rel_rec, rel_send,
                                                    rel_type, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds.transpose(1, 2).contiguous()
