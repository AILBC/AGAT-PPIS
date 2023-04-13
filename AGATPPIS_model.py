import pickle
import math
import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter

import warnings
warnings.filterwarnings("ignore")


# Feature Path
Feature_Path = "./Feature/"
# Seed
SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

# model parameters
BASE_MODEL_TYPE = 'AGAT'  # agat/gcn
ADD_NODEFEATS = 'all'  # all/atom_feats/psepose_embedding/no
USE_EFEATS = True  # True/False
if BASE_MODEL_TYPE == 'GCN':
    USE_EFEATS = False
MAP_CUTOFF = 14
DIST_NORM = 15

# INPUT_DIM
if ADD_NODEFEATS == 'all':  # add atom features and psepose embedding
    INPUT_DIM = 54 + 7 + 1
elif ADD_NODEFEATS == 'atom_feats':  # only add atom features
    INPUT_DIM = 54 + 7
elif ADD_NODEFEATS == 'psepose_embedding':  # only add psepose embedding
    INPUT_DIM = 54 + 1
elif ADD_NODEFEATS == 'no':
    INPUT_DIM = 54
HIDDEN_DIM = 256  # hidden size of node features
LAYER = 8  # the number of AGAT layers
DROPOUT = 0.1
ALPHA = 0.7
LAMBDA = 1.5

LEARNING_RATE = 1E-3
WEIGHT_DECAY = 0
BATCH_SIZE = 1
NUM_CLASSES = 2  # [not bind, bind]
NUMBER_EPOCHS = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def embedding(sequence_name):
    pssm_feature = np.load(Feature_Path + "pssm/" + sequence_name + '.npy')
    hmm_feature = np.load(Feature_Path + "hmm/" + sequence_name + '.npy')
    seq_embedding = np.concatenate([pssm_feature, hmm_feature], axis=1)
    return seq_embedding.astype(np.float32)


def get_dssp_features(sequence_name):
    dssp_feature = np.load(Feature_Path + "dssp/" + sequence_name + '.npy')
    return dssp_feature.astype(np.float32)

def get_res_atom_features(sequence_name):
    res_atom_feature = np.load(Feature_Path + "resAF/" + sequence_name + '.npy')
    return res_atom_feature.astype(np.float32)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def cal_edges(sequence_name, radius=MAP_CUTOFF):  # to get the index of the edges
    dist_matrix = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dist_matrix >= 0) * (dist_matrix <= radius))
    adjacency_matrix = mask.astype(np.int)
    radius_index_list = np.where(adjacency_matrix == 1)
    radius_index_list = [list(nodes) for nodes in radius_index_list]
    return radius_index_list

def load_graph(sequence_name):
    dismap = np.load(Feature_Path + "distance_map_SC/" + sequence_name + ".npy")
    mask = ((dismap >= 0) * (dismap <= MAP_CUTOFF))
    adjacency_matrix = mask.astype(np.int)
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))
    return norm_matrix


def graph_collate(samples):
    sequence_name, sequence, label, node_features, G, adj_matrix = map(list, zip(*samples))
    label = torch.Tensor(label)
    G_batch = dgl.batch(G)
    node_features = torch.cat(node_features)
    adj_matrix = torch.Tensor(adj_matrix)
    return sequence_name, sequence, label, node_features, G_batch, adj_matrix


class ProDataset(Dataset):
    def __init__(self, dataframe, radius=MAP_CUTOFF, dist=DIST_NORM, psepos_path='./Feature/psepos/Train335_psepos_SC.pkl'):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.residue_psepos = pickle.load(open(psepos_path, 'rb'))
        self.radius = radius
        self.dist = dist


    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        nodes_num = len(sequence)
        pos = self.residue_psepos[sequence_name]
        reference_res_psepos = pos[0]
        pos = pos - reference_res_psepos
        pos = torch.from_numpy(pos)

        sequence_embedding = embedding(sequence_name)
        structural_features = get_dssp_features(sequence_name)
        node_features = np.concatenate([sequence_embedding, structural_features], axis=1)
        node_features = torch.from_numpy(node_features)
        if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'atom_feats':
            res_atom_features = get_res_atom_features(sequence_name)
            res_atom_features = torch.from_numpy(res_atom_features)
            node_features = torch.cat([node_features, res_atom_features], dim=-1)
        if ADD_NODEFEATS == 'all' or ADD_NODEFEATS == 'psepose_embedding':
            node_features = torch.cat([node_features, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)

        radius_index_list = cal_edges(sequence_name, MAP_CUTOFF)
        edge_feat = self.cal_edge_attr(radius_index_list, pos)

        G = dgl.DGLGraph()
        G.add_nodes(nodes_num)
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = edge_feat.squeeze(1)

        self.add_edges_custom(G,
                              radius_index_list,
                              edge_feat
                              )

        adj_matrix = load_graph(sequence_name)
        node_features = node_features.detach().numpy()
        node_features = node_features[np.newaxis, :, :]
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)

        return sequence_name, sequence, label, node_features, G, adj_matrix

    def __len__(self):
        return len(self.labels)

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2,keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)

        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        radius_attr_list = np.array([distance, cos])
        return radius_attr_list

    def add_edges_custom(self, G, radius_index_list, edge_features):
        src, dst = radius_index_list[1], radius_index_list[0]
        if len(src) != len(dst):
            print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
            raise Exception
        G.add_edges(src, dst)
        G.edata['ex'] = torch.tensor(edge_features)

class AGATLayer(nn.Module):
    def __init__(self, nfeats_in_dim, nfeats_out_dim, edge_dim=2, use_efeats=USE_EFEATS):
        super(AGATLayer, self).__init__()
        self.use_efeats = use_efeats
        self.fc = nn.Linear(nfeats_in_dim, nfeats_out_dim, bias=False)
        if self.use_efeats:
            self.attn_fc = nn.Linear(2 * nfeats_out_dim + edge_dim, 1, bias=False)
            self.fc_edge_for_att_calc = nn.Linear(edge_dim, edge_dim, bias=False)
            self.fc_eFeatsDim_to_nFeatsDim = nn.Linear(edge_dim, nfeats_out_dim, bias=False)
        else:
            self.attn_fc = nn.Linear(2 * nfeats_out_dim, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        if self.use_efeats:
            nn.init.xavier_normal_(self.fc_edge_for_att_calc.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_eFeatsDim_to_nFeatsDim.weight, gain=gain)

    def edge_attention(self, edges):
        if self.use_efeats:
            z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['ex']], dim=1)
            a = self.attn_fc(z2)
        else:
            z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
            a = self.attn_fc(z2)

        if self.use_efeats:
            ez = self.fc_eFeatsDim_to_nFeatsDim(edges.data['ex'])
            return {'e': F.leaky_relu(a), 'ez': ez}
        else:
            return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        if self.use_efeats:
            return {'z': edges.src['z'], 'e': edges.data['e'], 'ez': edges.data['ez']}
        else:
            return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        attn_w = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(attn_w * nodes.mailbox['z'], dim=1)
        if self.use_efeats:
            h = h + torch.sum(attn_w * nodes.mailbox['ez'], dim=1)
        return {'h': h}

    def forward(self, g, h, e):
        z = self.fc(h)
        g.ndata['z'] = z
        if self.use_efeats:
            ex = self.fc_edge_for_att_calc(e)
            g.edata['ex'] = ex
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)

        return g.ndata.pop('h')

class BaseModule(nn.Module):
    def __init__(self, in_features, out_features):
        super(BaseModule, self).__init__()
        self.AGAT = AGATLayer(in_features, out_features)
        self.in_features = 2*in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, h0, lamda, alpha, l, adj_matrix=None, graph=None, efeats=None):
        theta = min(1, math.log(lamda/l+1))
        if adj_matrix is not None:
            hi = torch.sparse.mm(adj_matrix, input)
        elif graph is not None and efeats is not None:
            hi = self.AGAT(graph, input, efeats)
        else:
            print('ERROR:adj_matrix, graph and efeats must not be None at the same time! Please input the value of adj_matrix or the value of graph and efeats.')
            raise ValueError
        support = torch.cat([hi,h0],1)
        r = (1-alpha)*hi+alpha*h0
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        output = output+input
        return output


class deepAGAT(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha):
        super(deepAGAT, self).__init__()
        self.baseModules = nn.ModuleList()
        for _ in range(nlayers):
            self.baseModules.append(BaseModule(nhidden, nhidden))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj_matrix=None, graph=None, efeats=None):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,baseMod in enumerate(self.baseModules):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            if graph is not None and efeats is not None:
                layer_inner = baseMod(input=layer_inner, h0=_layers[0], lamda=self.lamda, alpha=self.alpha, l=i+1, graph=graph, efeats=efeats)
            elif adj_matrix is not None:
                layer_inner = baseMod(input=layer_inner, h0=_layers[0], lamda=self.lamda, alpha=self.alpha, l=i+1, adj_matrix=adj_matrix)
            else:
                print('ERROR:adj_matrix, graph and efeats must not be None at the same time! Please input the value of adj_matrix or the value of graph and efeats.')
                raise ValueError
            layer_inner = self.act_fn(layer_inner)
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner


class AGATPPIS(nn.Module):
    def __init__(self, nlayers, nfeat, nhidden, nclass, dropout, lamda, alpha):
        super(AGATPPIS, self).__init__()

        self.deep_agat = deepAGAT(nlayers=nlayers, nfeat=nfeat, nhidden=nhidden, nclass=nclass,
                                  dropout=dropout, lamda=lamda, alpha=alpha)
        self.criterion = nn.CrossEntropyLoss()  # automatically do softmax to the predicted value and one-hot to the label
        self.optimizer = torch.optim.Adam(self.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=10, min_lr=1e-6)

    def forward(self, x, graph, adj_matrix):
        x = x.float()
        x = x.view([x.shape[0]*x.shape[1], x.shape[2]])
        if BASE_MODEL_TYPE=='GCN':
            output = self.deep_agat(x=x, adj_matrix=adj_matrix)
        elif BASE_MODEL_TYPE=='AGAT':
            output = self.deep_agat(x=x, graph=graph, efeats=graph.edata['ex'])
        else:
            print('ERROR: The value of BASE_MODEL_TYPE is wrong!')
            raise ValueError
        return output
