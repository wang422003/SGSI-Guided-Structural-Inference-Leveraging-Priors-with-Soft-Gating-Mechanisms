from __future__ import division
from __future__ import print_function

import numpy as np
import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import *
from modules import *
from parametric_adj_utils import *


def parse_dict(arg):
    """
    Parse a dictionary from a string argument.
    Example input: "1:0.5,2:0.3,3:0.2"
    """
    return {int(k): float(v) for k, v in (item.split(":") for item in arg.split(","))}


t_begin = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=3407, help='Random seed.')  # 42
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--encoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--num-atoms', type=int, default=5,
                    help='Number of atoms in simulation.')
parser.add_argument('--encoder', type=str, default='softgate',
                    help='Type of path encoder model (mlp, cnn, gin, gate or softgate).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--no-factor', action='store_true', default=False,
                    help='Disables factor graph model.')
parser.add_argument('--suffix', type=str, default='',
                    help='Suffix for training data (e.g. "_charged".')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--edge-types', type=int, default=2,
                    help='The number of edge types to infer.')
parser.add_argument('--dims', type=int, default=4,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=49,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--skip-first', action='store_true', default=True,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')

# for benchmark:
parser.add_argument('--save-probs', action='store_true', default=False,
                    help='Save the probs during test.')
parser.add_argument('--b-portion', type=float, default=1.0,
                    help='Portion of data to be used in benchmarking.')
parser.add_argument('--b-time-steps', type=int, default=49,
                    help='Portion of time series in data to be used in benchmarking.')
parser.add_argument('--b-shuffle', action='store_true', default=False,
                    help='Shuffle the data for benchmarking?.')
parser.add_argument('--b-manual-nodes', type=int, default=0,
                    help='The number of nodes if changed from the original dataset.')
parser.add_argument('--data-path', type=str, default='',
                    help='Where to load the data. May input the paths to edges_train of the data.')
parser.add_argument('--b-network-type', type=str, default='',
                    help='What is the network type of the graph.')
parser.add_argument('--b-directed', action='store_true', default=False,
                    help='Default choose trajectories from undirected graphs.')
parser.add_argument('--b-simulation-type', type=str, default='',
                    help='Either springs or netsims.')
parser.add_argument('--b-suffix', type=str, default='',
    help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1.'
         ' Or "50r1" for 50 nodes, rep 1 and noise free.')
# remember to disable this for submission
parser.add_argument('--b-walltime', action='store_true', default=True,
                    help='Set wll time for benchmark training and testing. (Max time = 2 days)')

# for gating sparsity
parser.add_argument('--gating_lambda', type=float, default=0.0001,
                    help="The weight to encourage sparsity of the gating parameters.")
parser.add_argument('--beta', type=float, default=1.0,
                    help="The weight to for KL.")
# for priors:
parser.add_argument('--known-degree-out', type=parse_dict, default=None,
                    help='The known out-degree of the nodes.')
parser.add_argument('--known-degree-in', type=parse_dict, default=None,
                    help='The known in-degree of the nodes.')
parser.add_argument('--prior-sparsity', type=float, default=-1.0,
                    help='If positive, the sparsity of the prior.')
# for prior absent / present edges:
parser.add_argument("--num_known_present", type=int, default=2,
                    help="How many known-present edges to grab.")
parser.add_argument("--num_known_absent", type=int, default=2,
                    help="How many known-absent edges to grab.")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.factor = not args.no_factor
print(args)

if args.suffix == "":
    args.suffix = args.b_simulation_type
    args.timesteps = args.b_time_steps

# set flags:
if args.known_degree_out:
    args.prior_degree_out_flag = True
else:
    args.prior_degree_out_flag = False
if args.known_degree_in:
    args.prior_degree_in_flag = True
else:
    args.prior_degree_in_flag = False
if args.prior_sparsity != -1.0:
    args.prior_sparsity_flag = True
else:
    args.prior_sparsity_flag = False

if args.b_simulation_type == 'springs' or args.b_simulation_type == 'springs_simulations':
    args.dims = 4
elif args.b_simulation_type == 'netsims':
    args.dims = 1
else:
    args.dims = 1

if args.data_path == "" and args.b_network_type != "":
    if args.b_directed:
        dir_str = 'directed'
    else:
        dir_str = 'undirected'
    # args.data_path = (os.path.dirname(os.getcwd()) + '/data/StructInfer/VN_' + args.b_simulation_type + '_15-100'
    #                   + '/edges_train_' + args.b_simulation_type + args.b_suffix + '.npy')
    args.data_path = ('/data/StructInfer/VN_' + args.b_simulation_type + '_15-100'
                      + '/edges_train_' + args.b_simulation_type + args.b_suffix + '.npy')

    if args.b_network_type == 'vascular_networks':
        args.b_manual_nodes = int(args.b_suffix.split('r')[0])
    else:
        args.b_suffix = num_nodes(args.b_network_type)
        args.b_manual_nodes = int(args.b_suffix)
    # print("args.data_path: ", args.data_path)
if args.data_path != '':
    args.num_atoms = args.b_manual_nodes
# if args.data_path != '':
#     args.suffix = args.data_path.split('/')[-1].split('_', 2)[-1]

print("suffix: ", args.suffix)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dynamic_graph:
    print("Testing with dynamically re-computed graph.")

# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
    name_str = args.data_path.split('/')[-4] + '_' + args.data_path.split('/')[-3] + '_' + \
               args.data_path.split('/')[-1].split('_', 2)[-1].split('.')[0]
    # save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    save_folder = './{}/NRI-{}-E{}-D{}-exp{}/'.format(args.save_folder, name_str, args.encoder,
                                                      args.decoder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')
    res_folder = save_folder + 'results/'
    os.mkdir(res_folder)
    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

if args.prediction_steps > args.timesteps:
    args.prediction_steps = args.timesteps

if args.suffix == 'springs_simulations':
    print("Loading Springs Simulations Datasets: ")
    train_loader, valid_loader, test_loader = load_data_phy2(
        batch_size=args.batch_size
    )
    known_present, known_absent = trial_grab_known_n_absent_from_adj_spring_simulations(args)
elif args.suffix == 'netsims':
    print("Loading NetSims Datasets: ")
    train_loader, valid_loader, test_loader = load_data_netsim(
        batch_size=args.batch_size
    )
    known_present, known_absent = trial_grab_known_n_absent_from_adj_netsim2(args)
elif args.suffix in ['LI', 'LL', 'CY', 'BF', 'TF', 'BF-CV']:
    print("Loading genetic datasets ", args.suffix, ": ")
    train_loader, valid_loader, test_loader = load_data_genetic(
        batch_size=args.batch_size,
        data_type=args.suffix,
        shuffle_data=args.b_shuffle
    )
    known_present, known_absent = trial_grab_known_n_absent_from_adj_genetic(args)
elif args.suffix == "springs":
    print("Loading VN_SP Datasets: ")
    train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_customized_springs_data(
        args)
    known_present, known_absent = trial_grab_known_n_absent_from_adj_springs(args)
else:
    print("Loading VN_NS Datasets: ")
    train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_customized_netsims_data(
        args)
    known_present, known_absent = trial_grab_known_n_absent_from_adj_netsims(args)

# original:
# train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data(
#     args.batch_size, args.suffix)

# Generate off-diagonal interaction graph: discarded
# off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)
print("num_atoms: ", args.num_atoms)
off_diag = np.ones([args.num_atoms, args.num_atoms])

rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

if args.encoder == 'mlp':
    encoder = MLPEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)
elif args.encoder == 'cnn':
    encoder = CNNEncoder(args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)
elif args.encoder == 'gin':
    encoder = GINEncoder(args.timesteps * args.dims, args.encoder_hidden,
                         args.edge_types,
                         args.encoder_dropout, args.factor)
elif args.encoder == 'gate':
    encoder = MLPEncoderWithLearnedAdj(
        n_in=args.timesteps * args.dims,
        n_hid=args.encoder_hidden,
        n_out=args.edge_types,
        n_atoms=args.num_atoms,
        init_val=-2.0
    )

elif args.encoder == 'softgate':
    encoder = MLPEncoderWithGatingAndKnownEdges(
        n_in=args.timesteps * args.dims,
        n_hid=args.encoder_hidden,
        n_out=args.edge_types,
        n_atoms=args.num_atoms,
        known_present=known_present,
        known_absent=known_absent,
        init_gating=-2.0
    )

if args.decoder == 'mlp':
    decoder = MLPDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         msg_hid=args.decoder_hidden,
                         msg_out=args.decoder_hidden,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'rnn':
    decoder = RNNDecoder(n_in_node=args.dims,
                         edge_types=args.edge_types,
                         n_hid=args.decoder_hidden,
                         do_prob=args.decoder_dropout,
                         skip_first=args.skip_first)
elif args.decoder == 'sim':
    decoder = SimulationDecoder(loc_max, loc_min, vel_max, vel_min, args.suffix)

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))

    args.save_folder = False


# check the number of parameters
print("-" * 35)
# get the parameters counts of encoder and decoder
print("Encoder parameters: ", sum(p.numel() for p in encoder.parameters()))
print("Decoder parameters: ", sum(p.numel() for p in decoder.parameters()))
# print("-------Breakdown--------")
# # for each layer:
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name}: {param.numel()} parameters")
print("-" * 35)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

# Linear indices of an upper triangular mx, used for acc calculation
triu_indices = get_triu_offdiag_indices(args.num_atoms)
tril_indices = get_tril_offdiag_indices(args.num_atoms)

if args.prior:
    prior = np.array([0.91, 0.03, 0.03, 0.03])  # TODO: hard coded for now
    print("Using prior")
    print(prior)
    log_prior = torch.FloatTensor(np.log(prior))
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = torch.unsqueeze(log_prior, 0)
    log_prior = Variable(log_prior)

    if args.cuda:
        log_prior = log_prior.cuda()

if args.cuda:
    encoder.cuda()
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()
    triu_indices = triu_indices.cuda()
    tril_indices = tril_indices.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)


def train(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    acc_train = []
    kl_train = []
    mse_train = []
    gl_train = []
    sp_train = []
    deg_train = []

    encoder.train()
    decoder.train()
    scheduler.step()
    for batch_idx, (data, relations) in enumerate(train_loader):

        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data), Variable(relations)

        optimizer.zero_grad()

        logits, gating = encoder(data, rel_rec, rel_send)
        edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
        prob = my_softmax(logits, -1)

        if args.decoder == 'rnn':
            output = decoder(data, edges, rel_rec, rel_send, 100,
                             burn_in=True,
                             burn_in_steps=args.timesteps - args.prediction_steps)
        else:
            output = decoder(data, edges, rel_rec, rel_send,
                             args.prediction_steps)

        target = data[:, :, 1:, :]

        loss_nll = nll_gaussian(output, target, args.var)

        if args.prior:
            loss_kl = kl_categorical(prob, log_prior, args.num_atoms)
        else:
            loss_kl = kl_categorical_uniform(prob, args.num_atoms,
                                             args.edge_types)
        # gating regularization
        # gating shape [E] => sum => scalar
        gating_sum = gating.sum()  # sum of gating over all edges
        loss_gating = args.gating_lambda * gating_sum
        loss = loss_nll + args.beta * loss_kl + loss_gating
        # add optional constraints
        if args.prior_sparsity_flag:
            penalty_spars = target_sparsity_penalty(
                gating=gating,
                target_sparsity=args.prior_sparsity,
                lambda_sparsity=0.0001
            )
            sp_train.append(penalty_spars.item())
            loss = loss + penalty_spars

        if args.prior_degree_out_flag or args.prior_degree_in_flag:
            penalty_deg = partial_node_specific_degree_penalty(
                gating=gating,
                rel_send=rel_send,
                rel_rec=rel_rec,
                known_degree_out=args.known_degree_out,
                known_degree_in=args.known_degree_in,
                lambda_degree=0.0001
                )
            deg_train.append(penalty_deg.item())
            loss = loss + penalty_deg
        # print("Logits shape: ", logits.shape)
        # print("Relations shape: ", relations.shape)
        acc = edge_accuracy(logits, relations)
        acc_train.append(acc)

        loss.backward()
        optimizer.step()

        mse_train.append(F.mse_loss(output, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        gl_train.append(loss_gating.item())

    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []
    gl_val = []
    sp_val = []
    deg_val = []

    encoder.eval()
    decoder.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        # data, relations = Variable(data, volatile=True), Variable(
        #     relations, volatile=True)
        with torch.no_grad():
            logits, gating = encoder(data, rel_rec, rel_send)
            edges = gumbel_softmax(logits, tau=args.temp, hard=True)
            prob = my_softmax(logits, -1)

            # validation output uses teacher forcing
            output = decoder(data, edges, rel_rec, rel_send, 1)

            target = data[:, :, 1:, :]
            loss_nll = nll_gaussian(output, target, args.var)
            loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)

            # gating regularization
            # gating shape [E] => sum => scalar
            gating_sum = gating.sum()  # sum of gating over all edges
            loss_gating = args.gating_lambda * gating_sum

            if args.prior_sparsity_flag:
                penalty_spars = target_sparsity_penalty(
                    gating=gating,
                    target_sparsity=args.prior_sparsity,
                    lambda_sparsity=0.0001
                )
                sp_val.append(penalty_spars.item())
                # loss = loss + penalty_spars

            if args.prior_degree_out_flag or args.prior_degree_in_flag:
                penalty_deg = partial_node_specific_degree_penalty(
                    gating=gating,
                    rel_send=rel_send,
                    rel_rec=rel_rec,
                    known_degree_out=args.known_degree_out,
                    known_degree_in=args.known_degree_in,
                    lambda_degree=0.0001
                )
                deg_val.append(penalty_deg.item())
                # loss = loss + penalty_deg

            acc = edge_accuracy(logits, relations)
            acc_val.append(acc)

            mse_val.append(F.mse_loss(output, target).item())
            nll_val.append(loss_nll.item())
            kl_val.append(loss_kl.item())
            gl_val.append(loss_gating.item())

    print('Epoch: {:04d}'.format(epoch),
          'nll_train: {:.10f}'.format(np.mean(nll_train)),
          'kl_train: {:.10f}'.format(np.mean(kl_train)),
          'gl_train: {:.10f}'.format(np.mean(gl_train)),
          'sp_train: {:.10f}'.format(np.mean(sp_train)),
          'deg_train: {:.10f}'.format(np.mean(deg_train)),
          'mse_train: {:.10f}'.format(np.mean(mse_train)),
          'acc_train: {:.10f}'.format(np.mean(acc_train)),
          'nll_val: {:.10f}'.format(np.mean(nll_val)),
          'kl_val: {:.10f}'.format(np.mean(kl_val)),
          'gl_val: {:.10f}'.format(np.mean(gl_val)),
          'sp_val: {:.10f}'.format(np.mean(sp_val)),
          'deg_val: {:.10f}'.format(np.mean(deg_val)),
          'mse_val: {:.10f}'.format(np.mean(mse_val)),
          'acc_val: {:.10f}'.format(np.mean(acc_val)),
          'time: {:.4f}s'.format(time.time() - t))
    if args.save_folder and np.mean(nll_val) < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
        print('Epoch: {:04d}'.format(epoch),
              'nll_train: {:.10f}'.format(np.mean(nll_train)),
              'kl_train: {:.10f}'.format(np.mean(kl_train)),
              'gl_train: {:.10f}'.format(np.mean(gl_train)),
              'sp_train: {:.10f}'.format(np.mean(sp_train)),
              'deg_train: {:.10f}'.format(np.mean(deg_train)),
              'mse_train: {:.10f}'.format(np.mean(mse_train)),
              'acc_train: {:.10f}'.format(np.mean(acc_train)),
              'nll_val: {:.10f}'.format(np.mean(nll_val)),
              'kl_val: {:.10f}'.format(np.mean(kl_val)),
              'gl_val: {:.10f}'.format(np.mean(gl_val)),
              'sp_val: {:.10f}'.format(np.mean(sp_val)),
              'deg_val: {:.10f}'.format(np.mean(deg_val)),
              'mse_val: {:.10f}'.format(np.mean(mse_val)),
              'acc_val: {:.10f}'.format(np.mean(acc_val)),
              'time: {:.4f}s'.format(time.time() - t), file=log)
        log.flush()
    return np.mean(nll_val)


def test():
    acc_test = []
    nll_test = []
    kl_test = []
    gl_test = []
    sp_test = []
    deg_test = []
    mse_test = []
    prob_test = []
    auroc_test = []
    auprc_test = []

    tot_mse = 0
    counter = 0

    encoder.eval()
    decoder.eval()
    encoder.load_state_dict(torch.load(encoder_file))
    decoder.load_state_dict(torch.load(decoder_file))
    for batch_idx, (data, relations) in enumerate(test_loader):
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        # data, relations = Variable(data, volatile=True), Variable(
        #     relations, volatile=True)
        with torch.no_grad():
            if args.suffix not in ['netsims', 'LI', 'LL', 'CY', 'BF', 'TF', 'BF-CV']:
                assert (data.size(2) - args.timesteps) >= args.timesteps

            data_encoder = data[:, :, :args.timesteps, :].contiguous()
            data_decoder = data[:, :, -args.timesteps:, :].contiguous()

            logits, gating = encoder(data_encoder, rel_rec, rel_send)
            edges = gumbel_softmax(logits, tau=args.temp, hard=True)

            prob = my_softmax(logits, -1)
            prob_np = prob.detach().cpu().numpy()
            relations_np = relations.detach().cpu().numpy()
            prob_test.append(prob_np)
            output = decoder(data_decoder, edges, rel_rec, rel_send, 1)

            target = data_decoder[:, :, 1:, :]
            loss_nll = nll_gaussian(output, target, args.var)
            loss_kl = kl_categorical_uniform(prob, args.num_atoms, args.edge_types)
            # gating regularization
            # gating shape [E] => sum => scalar
            gating_sum = gating.sum()  # sum of gating over all edges
            loss_gating = args.gating_lambda * gating_sum

            if args.prior_sparsity_flag:
                penalty_spars = target_sparsity_penalty(
                    gating=gating,
                    target_sparsity=args.prior_sparsity,
                    lambda_sparsity=0.0001
                )
                sp_test.append(penalty_spars.item())
                # loss = loss + penalty_spars

            if args.prior_degree_out_flag or args.prior_degree_in_flag:
                penalty_deg = partial_node_specific_degree_penalty(
                    gating=gating,
                    rel_send=rel_send,
                    rel_rec=rel_rec,
                    known_degree_out=args.known_degree_out,
                    known_degree_in=args.known_degree_in,
                    lambda_degree=0.0001
                )
                deg_test.append(penalty_deg.item())
                # loss = loss + penalty_deg

            acc = edge_accuracy(logits, relations)
            acc_test.append(acc)

            preds_np = prob_np[:, :, 1]
            for i in range(len(prob_np)):
                # print("relation: ", relations_np[i])
                # print("preds: ", preds_np[i])
                auroc = roc_auc_score(relations_np[i], preds_np[i], average=None)
                auroc_test.append(auroc)
            auprc = average_precision_score(relations_np.flatten(), prob_np[:, :, 1].flatten())
            auprc_test.append(auprc)

            mse_test.append(F.mse_loss(output, target).item())
            nll_test.append(loss_nll.item())
            kl_test.append(loss_kl.item())
            gl_test.append(loss_gating.item())

            # For plotting purposes
            if args.decoder == 'rnn':
                if args.dynamic_graph:
                    output = decoder(data, edges, rel_rec, rel_send, 100,
                                     burn_in=True, burn_in_steps=args.timesteps,
                                     dynamic_graph=True, encoder=encoder,
                                     temp=args.temp)
                else:
                    output = decoder(data, edges, rel_rec, rel_send, 100,
                                     burn_in=True, burn_in_steps=args.timesteps)
                output = output[:, :, args.timesteps:, :]
                target = data[:, :, -args.timesteps:, :]
            else:
                data_plot = data[:, :, args.timesteps:args.timesteps + 21,
                            :].contiguous()
                output = decoder(data_plot, edges, rel_rec, rel_send, 20)
                target = data_plot[:, :, 1:, :]

            mse = ((target - output) ** 2).mean(dim=0).mean(dim=0).mean(dim=-1)
            tot_mse += mse.data.cpu().numpy()
            counter += 1
        if args.save_probs:
            np.save(save_folder + 'results/edges_test.npy', np.concatenate(prob_test))
            print("edges_test saved at: " + save_folder + 'results/edges_test.npy')
        mean_mse = tot_mse / counter
        mse_str = '['
        for mse_step in mean_mse[:-1]:
            mse_str += " {:.12f} ,".format(mse_step)
        if args.suffix != 'netsims':
            mse_str += " {:.12f} ".format(mean_mse[-1])
            mse_str += ']'

    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print('nll_test: {:.10f}'.format(np.mean(nll_test)),
          'kl_test: {:.10f}'.format(np.mean(kl_test)),
          'gl_test: {:.10f}'.format(np.mean(gl_test)),
          'sp_test: {:.10f}'.format(np.mean(sp_test)),
          'deg_test: {:.10f}'.format(np.mean(deg_test)),
          'mse_test: {:.10f}'.format(np.mean(mse_test)),
          'acc_test: {:.10f}'.format(np.mean(acc_test)),
          'auroc_test: {:.10f}'.format(np.mean(auroc_test)),
          'auprc_test: {:.10f}'.format(np.mean(auprc_test)))
    if args.suffix != 'netsims':
        print('MSE: {}'.format(mse_str))
    if args.save_folder:
        print('--------------------------------', file=log)
        print('--------Testing-----------------', file=log)
        print('--------------------------------', file=log)
        print('nll_test: {:.10f}'.format(np.mean(nll_test)),
              'kl_test: {:.10f}'.format(np.mean(kl_test)),
              'gl_test: {:.10f}'.format(np.mean(gl_test)),
              'sp_test: {:.10f}'.format(np.mean(sp_test)),
              'deg_test: {:.10f}'.format(np.mean(deg_test)),
              'mse_test: {:.10f}'.format(np.mean(mse_test)),
              'acc_test: {:.10f}'.format(np.mean(acc_test)),
              'auroc_test: {:.10f}'.format(np.mean(auroc_test)),
              'auprc_test: {:.10f}'.format(np.mean(auprc_test)),
              file=log)
        if args.suffix != 'netsims':
            print('MSE: {}'.format(mse_str), file=log)
        log.flush()

    print("Finished.")
    print("Dataset: ", args.suffix)
    print("Ground truth graph locates at: ", args.data_path)
    print("With portion: ", args.b_portion)
    print("With ", args.b_time_steps, " time steps")

# Train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0
for epoch in range(args.epochs):
    t_epoch_start = time.time()
    val_loss = train(epoch, best_val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - t_epoch_start
    if args.b_walltime:
        if epoch_end_time - t_begin < 171900 - epoch_time:
            continue
        else:
            break
print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()

test()
if log is not None:
    print(save_folder)
    log.close()
