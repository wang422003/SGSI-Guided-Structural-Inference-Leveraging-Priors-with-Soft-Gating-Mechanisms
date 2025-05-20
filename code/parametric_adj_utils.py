import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from utils import netsim_edges
import scipy.io

from torch.autograd import Variable


class LearnedAdj(nn.Module):
    def __init__(self, n_atoms, init_val=-2.0):
        """
        n_atoms: number of nodes
        init_val: float, initial gating_param value (e.g. -2 => gating ~ 0.12)
                  or 0 => gating ~ 0.5
        """
        super().__init__()
        E = n_atoms * n_atoms  # directed edges w. self loops
        # gating_param shape [E], each ~ init_val
        # We'll fill it with a constant
        param_init = torch.full((E,), init_val)
        self.gating_param = nn.Parameter(param_init)  # learnable

    def forward(self):
        """
        Returns gating in [0,1] for each edge: shape [E]
        i.e. alpha_e = sigmoid(gating_param[e])
        """
        return torch.sigmoid(self.gating_param)

    def gating_sum(self):
        """
        Returns sum of gating values => sum_e alpha_e
        Could be used for a mild L1 penalty or similar.
        """
        gating = torch.sigmoid(self.gating_param)
        return gating.sum()


def target_sparsity_penalty(gating, target_sparsity, lambda_sparsity=1e-3):
    """
    gating: [E] in [0,1], each gating[e] is the activation for edge e.
    target_sparsity: float in [0,1], desired average gating
    lambda_sparsity: penalty coefficient

    Return: A scalar penalty encouraging average gating ~ target_sparsity
    """
    # current average gating
    current_sparsity = gating.mean()  # shape: scalar
    # we penalize deviation from the desired target
    # e.g. L1 distance:
    penalty = (current_sparsity - target_sparsity).abs()
    return lambda_sparsity * penalty


def target_degree_penalty(gating, rel_send, target_degree, lambda_degree=1e-3):
    """
    gating: [E], gating[e] in [0,1]
    rel_send: [E, N], 1 if edge e is from node i
    target_degree: float, desired out-degree per node
    lambda_degree: penalty coefficient

    We penalize the L1 difference between each node’s sum of gating and target_degree.
    """
    device = gating.device
    E, N = rel_send.size()

    # For node i, out-degree_i = sum of gating[e where rel_send[e,i]=1]
    # We'll accumulate these in a loop or vector approach
    out_degs = []
    for i in range(N):
        # mask for edges from node i
        mask_e = (rel_send[:, i] == 1).float().to(device)  # shape [E]
        out_degree_i = (gating * mask_e).sum()  # scalar
        out_degs.append(out_degree_i)
    out_degs = torch.stack(out_degs)  # shape [N]

    # penalize difference from target_degree
    penalty = (out_degs - target_degree).abs().mean()  # L1 average
    return lambda_degree * penalty


def node_specific_degree_penalty(
    gating,        # shape [E]
    rel_send,      # [E, N], 1 if edge e sends from node i
    rel_rec,       # [E, N], 1 if edge e receives into node j
    target_out=None,  # shape [N] or None
    target_in=None,   # shape [N] or None
    lambda_degree=1e-3
):
    """
    gating: [E], each gating[e] in [0,1]
    rel_send, rel_rec: shape [E, N]
    target_out: optional array of shape [N], node-specific out-degree targets
    target_in:  optional array of shape [N], node-specific in-degree targets
    lambda_degree: penalty coefficient

    Returns a scalar penalty encouraging each node's out-degree or in-degree
    to match the specified targets.
    """
    device = gating.device
    E, N = rel_send.size()

    penalty = torch.tensor(0.0, device=device)

    if target_out:
        # out-degree of node i => sum of gating[e] for edges e where rel_send[e,i]=1
        # We'll gather those sums and compare to target_out[i]
        out_degs = []
        for i in range(N):
            mask_e = (rel_send[:, i] == 1).float().to(device)  # shape [E]
            deg_i = (gating * mask_e).sum()                    # scalar
            out_degs.append(deg_i)
        out_degs = torch.stack(out_degs)  # [N]
        # penalty for out-degree mismatch
        # e.g. L1 distance average
        tgt_out_t = torch.tensor(target_out, dtype=out_degs.dtype, device=device)
        penalty_out = (out_degs - tgt_out_t).abs().mean()
        penalty = penalty + penalty_out

    if target_in:
        # in-degree of node i => sum of gating[e] for edges e where rel_rec[e,i]=1
        in_degs = []
        for i in range(N):
            mask_e = (rel_rec[:, i] == 1).float().to(device)
            deg_i = (gating * mask_e).sum()
            in_degs.append(deg_i)
        in_degs = torch.stack(in_degs)  # [N]
        tgt_in_t = torch.tensor(target_in, dtype=in_degs.dtype, device=device)
        penalty_in = (in_degs - tgt_in_t).abs().mean()
        penalty = penalty + penalty_in

    return lambda_degree * penalty


def partial_node_specific_degree_penalty(
    gating,            # shape [E], gating[e] in [0,1]
    rel_send,          # [E, N], 1 if edge e is from node i
    rel_rec,           # [E, N], 1 if edge e is into node j
    known_degree_out,  # dict {node_idx: target_out_degree}
    known_degree_in,   # dict {node_idx: target_in_degree}
    lambda_degree=1e-3
):
    """
    gating: [E], each gating[e] is in [0,1].
    known_degree_out: {i: out_degree_i}, only for nodes we have out-degree info
    known_degree_in: {j: in_degree_j}, only for nodes we have in-degree info
    returns a scalar penalty that encourages each known node's out/in-degree
    to match the specified value.
    """
    device = gating.device
    E, N = rel_send.size()  # or rel_rec.size()

    penalty = torch.tensor(0.0, device=device)

    # 1) Out-degree penalty for nodes in known_degree_out
    if known_degree_out:
        for node_i, target_out_deg in known_degree_out.items():
            # edges from node_i => rel_send[e, node_i] == 1
            mask_e = (rel_send[:, node_i] == 1).float().to(device)  # shape [E]
            out_degree_i = (gating * mask_e).sum()                   # scalar
            penalty += (out_degree_i - target_out_deg).abs()

    # 2) In-degree penalty for nodes in known_degree_in
    if known_degree_in:
        for node_j, target_in_deg in known_degree_in.items():
            # edges into node_j => rel_rec[e, node_j] == 1
            mask_e = (rel_rec[:, node_j] == 1).float().to(device)
            in_degree_j = (gating * mask_e).sum()
            penalty += (in_degree_j - target_in_deg).abs()

    # Optionally average the penalty if you like
    total_known = 0
    if known_degree_out:
        total_known += len(known_degree_out)
    if known_degree_in:
        total_known += len(known_degree_in)
    # total_known = len(known_degree_out) + len(known_degree_in)
    if total_known > 0:
        penalty = penalty / total_known

    return lambda_degree * penalty


def grab_known_edges(
    adjacency,
    num_known_present,
    num_known_absent,
    random_seed=None
):
    """
    From a ground-truth adjacency matrix, randomly pick a specified number
    of present edges and absent edges to treat as "known".

    Args:
        adjacency: np.ndarray of shape [N, N], 0/1 for absent/present edges
        num_known_present: int, how many edges with adjacency=1 to select
        num_known_absent: int, how many edges with adjacency=0 to select
        random_seed: optional int for reproducible random selection

    Returns:
        known_present: set of (i, j) edges that are definitely present
        known_absent: set of (i, j) edges that are definitely absent
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    N = adjacency.shape[0]
    present_edges = []
    absent_edges = []

    # Collect candidate edges
    for i in range(N):
        for j in range(N):
            # if i == j:
            #     continue  # skip self-loops if not used in your setting
            if adjacency[i, j] == 1:
                present_edges.append((i, j))
            else:
                absent_edges.append((i, j))

    # Shuffle them so we can pick randomly
    np.random.shuffle(present_edges)
    np.random.shuffle(absent_edges)

    # Slice the desired number
    known_present = set(present_edges[:num_known_present])
    known_absent = set(absent_edges[:num_known_absent])

    return known_present, known_absent


def trial_grab_known_n_absent_from_adj_spring_simulations(args):
    edges_train = np.load('./data/physics_simulations/edges_train_springs.npy')
    return grab_known_edges(
        edges_train,
        num_known_present=args.num_known_present,
        num_known_absent=args.num_known_absent,
        random_seed=args.seed
    )


def trial_grab_known_n_absent_from_adj_netsim2(args):
    mat = scipy.io.loadmat('./data/netsims/sim2.mat')
    info_connection = mat['net']
    edges_train = netsim_edges(info_connection[0])
    return grab_known_edges(
        edges_train,
        num_known_present=args.num_known_present,
        num_known_absent=args.num_known_absent,
        random_seed=args.seed
    )


def trial_grab_known_n_absent_from_adj_springs(args):
    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'

    edges_train = np.load(root_str + 'edges_train_' + keep_str)
    return grab_known_edges(
        edges_train,
        num_known_present=args.num_known_present,
        num_known_absent=args.num_known_absent,
        random_seed=args.seed
    )


def trial_grab_known_n_absent_from_adj_netsims(args):
    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'

    edges_train = np.load(root_str + 'edges_train_' + keep_str)
    return grab_known_edges(
        edges_train,
        num_known_present=args.num_known_present,
        num_known_absent=args.num_known_absent,
        random_seed=args.seed
    )


def trial_grab_known_n_absent_from_adj_genetic(args):
    edges_train = np.load('./data/Synthetic-H/sampled_data/' + args.suffix + '/edges.npy')
    return grab_known_edges(
        edges_train,
        num_known_present=args.num_known_present,
        num_known_absent=args.num_known_absent,
        random_seed=args.seed
    )


def num_nodes(data_suffix='LI'):
    if data_suffix == 'LI':
        return 7
    elif data_suffix == 'LL':
        return 18
    elif data_suffix == 'CY':
        return 6
    elif data_suffix == 'BF':
        return 7
    elif data_suffix == 'TF':
        return 8
    elif data_suffix == 'BF-CV':
        return 10
    elif data_suffix == 'springs_simulations':
        return 10
    elif data_suffix == 'netsims':
        return 10
    else:
        raise ValueError("Check the suffix of the dataset!")


if __name__ == "__main__":
    print("This is parametric_adj_utils.py")