import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time
from tqdm import tqdm

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

class HyperConv(Module):
    def __init__(self, layers,dataset,emb_size=100):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset

    def forward(self, adjacency, embedding):
        acc = embedding
        out = embedding
        for _ in range(self.layers):
            acc = torch.sparse.mm(adjacency, acc)
            out = out + acc
        return out / (self.layers + 1)


class LineConv(Module):
    def __init__(self, layers,batch_size,emb_size=100):
        super(LineConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers
    def forward(self, item_embedding, D, A, session_item, session_len):
        zeros = torch.zeros(1, self.emb_size, device=item_embedding.device)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        seq_h1 = item_embedding[session_item]
        session_emb_lgcn = seq_h1.sum(1) / session_len.float().clamp(min=1)
        session = [session_emb_lgcn]
        DA = torch.mm(D, A).float()
        for i in range(self.layers):
            session_emb_lgcn = torch.mm(DA, session_emb_lgcn)
            session.append(session_emb_lgcn)
        #session1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in session]))
        #session_emb_lgcn = torch.sum(session1, 0)
        session_emb_lgcn = torch.stack(session, dim=0).sum(0) / (self.layers + 1)
        return session_emb_lgcn


class DHCN(Module):
    def __init__(self, adjacency, n_node,lr, layers,l2, beta,dataset,emb_size=100, batch_size=100):
        super(DHCN, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.dataset = dataset

        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        if dataset == 'Nowplaying':
            index_fliter = (values < 0.05).nonzero()
            values = np.delete(values, index_fliter)
            indices1 = np.delete(indices[0], index_fliter)
            indices2 = np.delete(indices[1], index_fliter)
            indices = [indices1, indices2]
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        self.adjacency = adjacency
        if torch.cuda.is_available():
            self.adjacency = self.adjacency.cuda()
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.pos_embedding = nn.Embedding(200, self.emb_size)
        self.HyperGraph = HyperConv(self.layers,dataset)
        self.LineGraph = LineConv(self.layers, self.batch_size)
        self.w_1 = nn.Linear(2 * self.emb_size, self.emb_size)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

     
    def generate_sess_emb(self,item_embedding, session_item, session_len, reversed_sess_item, mask):
        device = item_embedding.device
        zeros = torch.zeros(1, self.emb_size, device=device)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        batch_size = reversed_sess_item.shape[0]
        seq_h = item_embedding[reversed_sess_item]
        hs = seq_h.sum(1) / session_len.float().clamp(min=1)
        mask = mask.float().unsqueeze(-1)
        seq_len = seq_h.shape[1]
        max_pos = self.pos_embedding.num_embeddings
        if seq_len > max_pos:
            seq_h = seq_h[:, :max_pos, :]
            mask = mask[:, :max_pos, :]
            seq_len = max_pos
        pos_emb = self.pos_embedding.weight[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)

        hs = hs.unsqueeze(-2).expand(-1, seq_len, -1)
        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    def generate_sess_emb_npos(self,item_embedding, session_item, session_len, reversed_sess_item, mask):
        device = item_embedding.device
        zeros = torch.zeros(1, self.emb_size, device=device)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        seq_h = item_embedding[reversed_sess_item]
        hs = seq_h.sum(1) / session_len.float().clamp(min=1)
        mask = mask.float().unsqueeze(-1)
        seq_len = seq_h.shape[1]

        hs = hs.unsqueeze(-2).expand(-1, seq_len, -1)
        nh = seq_h
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:,torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos = score(sess_emb_hgnn, sess_emb_lgcn)
        neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
        one = torch.ones(neg1.shape[0], device=neg1.device)
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        return con_loss

    def forward(self, session_item, session_len, D, A, reversed_sess_item, mask):
        item_embeddings_hg = self.HyperGraph(self.adjacency, self.embedding.weight)
        if self.dataset == 'Tmall':
            sess_emb_hgnn = self.generate_sess_emb_npos(item_embeddings_hg, session_item, session_len, reversed_sess_item, mask)
        else:
            sess_emb_hgnn = self.generate_sess_emb(item_embeddings_hg, session_item, session_len, reversed_sess_item, mask)
        session_emb_lg = self.LineGraph(self.embedding.weight, D, A, session_item, session_len)
        con_loss = self.SSL(sess_emb_hgnn, session_emb_lg)
        return item_embeddings_hg, sess_emb_hgnn, self.beta*con_loss


def forward(model, i, data):
    tar, session_len, session_item, reversed_sess_item, mask = data.get_slice(i)
    device = next(model.parameters()).device
    A_hat, D_hat = data.get_overlap_tensors(i, device=device)
    session_item = torch.as_tensor(session_item, dtype=torch.long, device=device)
    session_len = torch.as_tensor(session_len, dtype=torch.long, device=device)
    tar = torch.as_tensor(tar, dtype=torch.long, device=device)
    mask = torch.as_tensor(mask, dtype=torch.long, device=device)
    reversed_sess_item = torch.as_tensor(reversed_sess_item, dtype=torch.long, device=device)
    item_emb_hg, sess_emb_hgnn, con_loss = model(session_item, session_len, D_hat, A_hat, reversed_sess_item, mask)
    scores = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1,0))
    return tar, scores, con_loss


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now(), flush=True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i in tqdm(slices, desc='train', mininterval=1.0):
        model.zero_grad()
        targets, scores, con_loss = forward(model, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets)
        loss = loss + con_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss, flush=True)
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now(), flush=True)

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    with torch.inference_mode():
        for i in tqdm(slices, desc='test', mininterval=1.0):
            tar, scores, _ = forward(model, i, test_data)
            sub_scores = scores.topk(20, dim=1).indices.cpu().numpy()
            tar = tar.cpu().numpy()
            for K in top_K:
                for prediction, target in zip(sub_scores[:, :K], tar):
                    metrics['hit%d' % K].append(np.isin(target, prediction))
                    hit_pos = np.where(prediction == target)[0]
                    if len(hit_pos) == 0:
                        metrics['mrr%d' % K].append(0)
                    else:
                        metrics['mrr%d' % K].append(1 / (hit_pos[0] + 1))
    return metrics, total_loss


