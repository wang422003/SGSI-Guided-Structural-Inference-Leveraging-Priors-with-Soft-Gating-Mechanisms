import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.io


def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    # soft_max_1d = F.softmax(trans_input)
    soft_max_1d = F.softmax(trans_input, dim=-1)  # modified on Dec.26, 2024
    return soft_max_1d.transpose(axis, 0)


def binary_concrete(logits, tau=1, hard=False, eps=1e-10):
    y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = Variable(y_hard.data - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def binary_concrete_sample(logits, tau=1, eps=1e-10):
    logistic_noise = sample_logistic(logits.size(), eps=eps)
    if logits.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = logits + Variable(logistic_noise)
    return F.sigmoid(y / tau)


def sample_logistic(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return torch.log(uniform + eps) - torch.log(1 - uniform + eps)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def binary_accuracy(output, labels):
    preds = output > 0.5
    correct = preds.type_as(labels).eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def portion_data(raw_data, data_portion, time_steps, shuffle):
    if data_portion == 1.0 and time_steps == 49:
        return raw_data
    if shuffle:
        np.random.shuffle(raw_data)
    num_trajs = raw_data.shape[0]
    num_times = raw_data.shape[0]
    return raw_data[:int(num_trajs * data_portion), :int(time_steps), :, :]


def load_data_genetic(batch_size=1, data_type='LI',
                      self_loops=True, distributed_flag=False, norm_flag=False,
                      data_portion=1.0, time_steps=49, shuffle_data=False):
    train_traj = np.load('./data/Synthetic-H/sampled_data/' + data_type + '/train.npy')
    # shape:[num_simulations, num_genes, time_steps]
    train_traj = np.transpose(train_traj, [0, 2, 1])  # change to [num_simulations, timesteps, num_genes]
    train_traj = train_traj[..., np.newaxis]  # shape: [num_sim, timesteps, num_genes, dimension]
    train_traj = np.transpose(train_traj, [0, 1, 3, 2])  # shape: [num_sim, timesteps, dimension, num_genes]
    train_traj = portion_data(train_traj, data_portion, time_steps, shuffle_data)

    n_train = train_traj.shape[0]
    edges_train = np.load('./data/Synthetic-H/sampled_data/' + data_type + '/edges.npy')
    edges_train = np.tile(edges_train, (n_train, 1, 1))

    valid_traj = np.load('./data/Synthetic-H/sampled_data/' + data_type + '/valid.npy')
    valid_traj = np.transpose(valid_traj, [0, 2, 1])  # change to [num_simulations, timesteps, num_genes]
    valid_traj = valid_traj[..., np.newaxis]  # shape: [num_sim, timesteps, num_genes, dimension]
    valid_traj = np.transpose(valid_traj, [0, 1, 3, 2])
    valid_traj = portion_data(valid_traj, data_portion, time_steps, shuffle_data)

    n_valid = valid_traj.shape[0]
    edges_valid = np.load('./data/Synthetic-H/sampled_data/' + data_type + '/edges.npy')
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))

    test_traj = np.load('./data/Synthetic-H/sampled_data/' + data_type + '/test.npy')
    test_traj = np.transpose(test_traj, [0, 2, 1])  # change to [num_simulations, timesteps, num_genes]
    test_traj = test_traj[..., np.newaxis]  # shape: [num_sim, timesteps, num_genes, dimension]
    test_traj = np.transpose(test_traj, [0, 1, 3, 2])

    n_test = test_traj.shape[0]
    edges_test = np.load('./data/Synthetic-H/sampled_data/' + data_type + '/edges.npy')
    edges_test = np.tile(edges_test, (n_test, 1, 1))

    # [num_sim, timesteps, dimension, num_genes]
    num_nodes = train_traj.shape[3]

    loc_max = train_traj.max()
    loc_min = train_traj.min()

    if norm_flag:
        # Normalize to [-1, 1]
        norm_train = (train_traj - loc_min) * 2 / (loc_max - loc_min) - 1

        norm_valid = (valid_traj - loc_min) * 2 / (loc_max - loc_min) - 1

        norm_test = (test_traj - loc_min) * 2 / (loc_max - loc_min) - 1
    else:
        norm_train = train_traj
        norm_valid = valid_traj
        norm_test = test_traj

    # Reshape to: [num_sims, num_genes, num_timesteps, num_dims]

    # NOTE: added normalization on Jun.29
    # feat_train = np.transpose(train_traj, [0, 3, 1, 2])  # without normalization
    feat_train = np.transpose(norm_train, [0, 3, 1, 2])
    edges_train = np.reshape(edges_train, [-1, num_nodes ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    # feat_valid = np.transpose(valid_traj, [0, 3, 1, 2])  # without normalization
    feat_valid = np.transpose(norm_valid, [0, 3, 1, 2])
    edges_valid = np.reshape(edges_valid, [-1, num_nodes ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    # feat_test = np.transpose(test_traj, [0, 3, 1, 2])  # without normalization
    feat_test = np.transpose(norm_test, [0, 3, 1, 2])
    edges_test = np.reshape(edges_test, [-1, num_nodes ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    if not self_loops:
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)),
            [num_nodes, num_nodes])
        edges_train = edges_train[:, off_diag_idx]
        edges_valid = edges_valid[:, off_diag_idx]
        edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    if distributed_flag:
        train_sampler = DistributedSampler(train_data)
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, sampler=train_sampler)

    else:
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def load_data_phy2(batch_size=1, suffix='springs', self_loops=True, data_portion=1.0, time_steps=49, shuffle_data=False):
    loc_train = np.load('./data/physics_simulations/loc_train_' + suffix + '.npy')
    vel_train = np.load('./data/physics_simulations/vel_train_' + suffix + '.npy')
    edges_train = np.load('./data/physics_simulations/edges_train_' + suffix + '.npy')

    loc_valid = np.load('./data/physics_simulations/loc_valid_' + suffix + '.npy')
    vel_valid = np.load('./data/physics_simulations/vel_valid_' + suffix + '.npy')
    edges_valid = np.load('./data/physics_simulations/edges_valid_' + suffix + '.npy')

    loc_test = np.load('./data/physics_simulations/loc_test_' + suffix + '.npy')
    vel_test = np.load('./data/physics_simulations/vel_test_' + suffix + '.npy')
    edges_test = np.load('./data/physics_simulations/edges_test_' + suffix + '.npy')

    loc_train = portion_data(loc_train, data_portion, time_steps, shuffle_data)
    vel_train = portion_data(vel_train, data_portion, time_steps, shuffle_data)

    loc_valid = portion_data(loc_valid, data_portion, time_steps, shuffle_data)
    vel_valid = portion_data(vel_valid, data_portion, time_steps, shuffle_data)

    # [num_samples, num_timesteps, num_dims, num_nodes]
    num_nodes = loc_train.shape[3]

    n_train = loc_train.shape[0]
    n_test = loc_test.shape[0]
    n_valid = loc_valid.shape[0]

    edges_train = np.tile(edges_train, (n_train, 1, 1))
    edges_test = np.tile(edges_test, (n_test, 1, 1))
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_nodes, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.reshape(edges_train, [-1, num_nodes ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.reshape(edges_valid, [-1, num_nodes ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.reshape(edges_test, [-1, num_nodes ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    if not self_loops:
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)),
            [num_nodes, num_nodes])
        edges_train = edges_train[:, off_diag_idx]
        edges_valid = edges_valid[:, off_diag_idx]
        edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def netsim_features(raw_features, valid=False):
    n_total_raw = raw_features.shape[0]
    n_time_raw = raw_features.shape[2]
    if not valid:
        n_time_new = n_time_raw - 49
    else:
        n_time_new = n_time_raw - 99
    features_new = list()
    for i in range(n_time_new):
        features_new.append(raw_features[:, :, i: i + 49, :])
    features_new = np.concatenate(features_new)
    np.random.shuffle(features_new)
    return features_new


def netsim_edges(connection):
    edges = np.zeros_like(connection)
    for i, row in enumerate(connection):
        for j, element in enumerate(row):
            if element > 0:
                edges[i][j] = 1
    return edges


def load_data_netsim(batch_size=1, suffix='', self_loops=True, data_portion=1.0, time_steps=49, shuffle_data=False):
    mat = scipy.io.loadmat('./data/netsims/sim2.mat')

    num_nodes = mat['Nnodes'][0][0]
    num_sims = mat['Nsubjects'][0][0]
    num_time = mat['Ntimepoints'][0][0]

    feat_raw = mat['ts']
    feat_raw = feat_raw.reshape(num_sims, num_nodes, num_time, 1)
    np.random.shuffle(feat_raw)
    feat_train = feat_raw[0: int(num_sims * 0.8)]
    feat_test = feat_raw[int(num_sims * 0.8): int(num_sims * 0.9)]
    feat_valid = feat_raw[int(num_sims * 0.9):]

    feat_train = netsim_features(feat_train)
    feat_test = netsim_features(feat_test)
    feat_valid = netsim_features(feat_valid, valid=True)

    feat_train = portion_data(feat_train, data_portion, time_steps, shuffle_data)
    feat_valid = portion_data(feat_valid, data_portion, time_steps, shuffle_data)
    feat_test = portion_data(feat_test, data_portion, time_steps, shuffle_data)

    n_train = feat_train.shape[0]
    n_test = feat_test.shape[0]
    n_valid = feat_valid.shape[0]

    info_connection = mat['net']
    edges = netsim_edges(info_connection[0])
    edges_train = np.tile(edges, (n_train, 1, 1))
    edges_test = np.tile(edges, (n_test, 1, 1))
    edges_valid = np.tile(edges, (n_valid, 1, 1))

    # [num_samples, num_timesteps, num_dims, num_nodes]

    # Reshape to: [num_sims, num_atoms * num_atoms]
    edges_train = np.reshape(edges_train, [-1, num_nodes ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_nodes ** 2])
    edges_test = np.reshape(edges_test, [-1, num_nodes ** 2])

    edges_train = edges_train / np.max(edges_train)
    edges_valid = edges_valid / np.max(edges_valid)
    edges_test = edges_test / np.max(edges_test)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    if not self_loops:
        off_diag_idx = np.ravel_multi_index(
            np.where(np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)),
            [num_nodes, num_nodes])
        edges_train = edges_train[:, off_diag_idx]
        edges_valid = edges_valid[:, off_diag_idx]
        edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def load_customized_springs_data(args):

    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'

    loc_train = np.load(root_str + 'loc_train_' + keep_str)
    vel_train = np.load(root_str + 'vel_train_' + keep_str)
    edges_train = np.load(root_str + 'edges_train_' + keep_str)
    edges_train[edges_train > 0] = 1

    loc_valid = np.load(root_str + 'loc_valid_' + keep_str)
    vel_valid = np.load(root_str + 'vel_valid_' + keep_str)
    edges_valid = np.load(root_str + 'edges_valid_' + keep_str)
    edges_valid[edges_valid > 0] = 1

    loc_test = np.load(root_str + 'loc_test_' + keep_str)
    vel_test = np.load(root_str + 'vel_test_' + keep_str)
    edges_test = np.load(root_str + 'edges_test_' + keep_str)
    edges_test[edges_test > 0] = 1

    loc_train = portion_data(loc_train, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_train = portion_data(vel_train, args.b_portion, args.b_time_steps, args.b_shuffle)

    loc_valid = portion_data(loc_valid, args.b_portion, args.b_time_steps, args.b_shuffle)
    vel_valid = portion_data(vel_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    num_nodes = loc_train.shape[3]

    n_train = loc_train.shape[0]
    n_test = loc_test.shape[0]
    n_valid = loc_valid.shape[0]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_timesteps, num_nodes, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.tile(edges_train, (n_train, 1, 1))
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.tile(edges_test, (n_test, 1, 1))
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges: discarded
    # off_diag_idx = np.ravel_multi_index(
    #     np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
    #     [num_atoms, num_atoms])
    # edges_train = edges_train[:, off_diag_idx]
    # edges_valid = edges_valid[:, off_diag_idx]
    # edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=args.batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=args.batch_size)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min


def load_customized_netsims_data(args):

    keep_str = args.data_path.split('/')[-1].split('_', 2)[-1]
    root_str = args.data_path[::-1].split('/', 1)[1][::-1] + '/'

    bold_train = np.load(root_str + 'bold_train_' + keep_str)
    edges_train = np.load(root_str + 'edges_train_' + keep_str)
    edges_train[edges_train > 0] = 1

    bold_valid = np.load(root_str + 'bold_valid_' + keep_str)
    edges_valid = np.load(root_str + 'edges_valid_' + keep_str)
    edges_valid[edges_valid > 0] = 1

    bold_test = np.load(root_str + 'bold_test_' + keep_str)
    edges_test = np.load(root_str + 'edges_test_' + keep_str)
    edges_test[edges_test > 0] = 1

    bold_train = portion_data(bold_train, args.b_portion, args.b_time_steps, args.b_shuffle)

    bold_valid = portion_data(bold_valid, args.b_portion, args.b_time_steps, args.b_shuffle)

    num_nodes = bold_train.shape[3]

    n_train = bold_train.shape[0]
    n_test = bold_test.shape[0]
    n_valid = bold_valid.shape[0]

    bold_max = bold_train.max()
    bold_min = bold_train.min()

    bold_train = (bold_train - bold_min) * 2 / (bold_max - bold_min) - 1

    bold_valid = (bold_valid - bold_min) * 2 / (bold_max - bold_min) - 1

    bold_test = (bold_test - bold_min) * 2 / (bold_max - bold_min) - 1

    # Reshape to: [num_sims, num_timesteps, num_nodes, num_dims]
    feat_train = np.transpose(bold_train, [0, 3, 1, 2])
    edges_train = np.tile(edges_train, (n_train, 1, 1))
    edges_train = np.reshape(edges_train, [-1, num_nodes ** 2])

    feat_valid = np.transpose(bold_valid, [0, 3, 1, 2])
    edges_valid = np.tile(edges_valid, (n_valid, 1, 1))
    edges_valid = np.reshape(edges_valid, [-1, num_nodes ** 2])

    feat_test = np.transpose(bold_test, [0, 3, 1, 2])
    edges_test = np.tile(edges_test, (n_test, 1, 1))
    edges_test = np.reshape(edges_test, [-1, num_nodes ** 2])

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges: discarded
    # off_diag_idx = np.ravel_multi_index(
    #     np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
    #     [num_atoms, num_atoms])
    # edges_train = edges_train[:, off_diag_idx]
    # edges_valid = edges_valid[:, off_diag_idx]
    # edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=args.batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=args.batch_size)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, bold_max, bold_min, bold_max, bold_min


def load_data(batch_size=1, suffix=''):
    loc_train = np.load('data/loc_train' + suffix + '.npy')
    vel_train = np.load('data/vel_train' + suffix + '.npy')
    edges_train = np.load('data/edges_train' + suffix + '.npy')

    loc_valid = np.load('data/loc_valid' + suffix + '.npy')
    vel_valid = np.load('data/vel_valid' + suffix + '.npy')
    edges_valid = np.load('data/edges_valid' + suffix + '.npy')

    loc_test = np.load('data/loc_test' + suffix + '.npy')
    vel_test = np.load('data/vel_test' + suffix + '.npy')
    edges_test = np.load('data/edges_test' + suffix + '.npy')

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min


def load_kuramoto_data(batch_size=1, suffix=''):
    feat_train = np.load('data/feat_train' + suffix + '.npy')
    edges_train = np.load('data/edges_train' + suffix + '.npy')
    feat_valid = np.load('data/feat_valid' + suffix + '.npy')
    edges_valid = np.load('data/edges_valid' + suffix + '.npy')
    feat_test = np.load('data/feat_test' + suffix + '.npy')
    edges_test = np.load('data/edges_test' + suffix + '.npy')

    # [num_sims, num_atoms, num_timesteps, num_dims]
    num_atoms = feat_train.shape[1]

    # Normalize each feature dim. individually
    feat_max = feat_train.max(0).max(0).max(0)
    feat_min = feat_train.min(0).min(0).min(0)

    feat_max = np.expand_dims(np.expand_dims(np.expand_dims(feat_max, 0), 0), 0)
    feat_min = np.expand_dims(np.expand_dims(np.expand_dims(feat_min, 0), 0), 0)

    # Normalize to [-1, 1]
    feat_train = (feat_train - feat_min) * 2 / (feat_max - feat_min) - 1
    feat_valid = (feat_valid - feat_min) * 2 / (feat_max - feat_min) - 1
    feat_test = (feat_test - feat_min) * 2 / (feat_max - feat_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def load_kuramoto_data_old(batch_size=1, suffix=''):
    feat_train = np.load('data/old_kuramoto/feat_train' + suffix + '.npy')
    edges_train = np.load('data/old_kuramoto/edges_train' + suffix + '.npy')
    feat_valid = np.load('data/old_kuramoto/feat_valid' + suffix + '.npy')
    edges_valid = np.load('data/old_kuramoto/edges_valid' + suffix + '.npy')
    feat_test = np.load('data/old_kuramoto/feat_test' + suffix + '.npy')
    edges_test = np.load('data/old_kuramoto/edges_test' + suffix + '.npy')

    # [num_sims, num_atoms, num_timesteps, num_dims]
    num_atoms = feat_train.shape[1]

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def load_motion_data(batch_size=1, suffix=''):
    feat_train = np.load('data/motion_train' + suffix + '.npy')
    feat_valid = np.load('data/motion_valid' + suffix + '.npy')
    feat_test = np.load('data/motion_test' + suffix + '.npy')
    adj = np.load('data/motion_adj' + suffix + '.npy')

    # NOTE: Already normalized

    # [num_samples, num_nodes, num_timesteps, num_dims]
    num_nodes = feat_train.shape[1]

    edges_train = np.repeat(np.expand_dims(adj.flatten(), 0),
                            feat_train.shape[0], axis=0)
    edges_valid = np.repeat(np.expand_dims(adj.flatten(), 0),
                            feat_valid.shape[0], axis=0)
    edges_test = np.repeat(np.expand_dims(adj.flatten(), 0),
                           feat_test.shape[0], axis=0)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(np.array(edges_train, dtype=np.int64))
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(np.array(edges_valid, dtype=np.int64))
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(np.array(edges_test, dtype=np.int64))

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)),
        [num_nodes, num_nodes])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader


def to_2d_idx(idx, num_cols):
    idx = np.array(idx, dtype=np.int64)
    y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
    x_idx = idx % num_cols
    return x_idx, y_idx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def get_minimum_distance(data):
    data = data[:, :, :, :2].transpose(1, 2)
    data_norm = (data ** 2).sum(-1, keepdim=True)
    dist = data_norm + \
           data_norm.transpose(2, 3) - \
           2 * torch.matmul(data, data.transpose(2, 3))
    min_dist, _ = dist.min(1)
    return min_dist.view(min_dist.size(0), -1)


def get_buckets(dist, num_buckets):
    dist = dist.cpu().data.numpy()

    min_dist = np.min(dist)
    max_dist = np.max(dist)
    bucket_size = (max_dist - min_dist) / num_buckets
    thresholds = bucket_size * np.arange(num_buckets)

    bucket_idx = []
    for i in range(num_buckets):
        if i < num_buckets - 1:
            idx = np.where(np.all(np.vstack((dist > thresholds[i],
                                             dist <= thresholds[i + 1])), 0))[0]
        else:
            idx = np.where(dist > thresholds[i])[0]
        bucket_idx.append(idx)

    return bucket_idx, thresholds


def get_correct_per_bucket(bucket_idx, pred, target):
    pred = pred.cpu().numpy()[:, 0]
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def get_correct_per_bucket_(bucket_idx, pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))


def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1))


def edge_accuracy(preds, target):
    _, preds = preds.max(-1)
    correct = preds.float().data.eq(
        target.float().data.view_as(preds)).cpu().sum()
    return float(correct) / (target.size(0) * target.size(1))
