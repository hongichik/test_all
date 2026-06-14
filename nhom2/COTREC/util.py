import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from operator import itemgetter
import random


def _jaccard_overlap(unique_lists):
    """Ma trận Jaccard (n,n) + degree — vectorized, ~ms cho batch 100."""
    n = len(unique_lists)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    cols = sorted({x for u in unique_lists for x in u})
    col_idx = {c: j for j, c in enumerate(cols)}
    m = max(len(cols), 1)
    nnz = sum(len(u) for u in unique_lists)
    if nnz == 0:
        matrix = np.eye(n, dtype=np.float32)
        return matrix, np.diag(1.0 / np.maximum(matrix.sum(axis=1), 1.0)).astype(np.float32)

    rows = np.fromiter((i for i, u in enumerate(unique_lists) for _ in u), dtype=np.int32, count=nnz)
    cols_i = np.fromiter((col_idx[x] for u in unique_lists for x in u), dtype=np.int32, count=nnz)
    B = csr_matrix((np.ones(nnz, dtype=np.float32), (rows, cols_i)), shape=(n, m))
    inter = (B @ B.T).toarray().astype(np.float32)
    sizes = np.asarray(B.sum(axis=1)).ravel()
    union = sizes[:, None] + sizes[None, :] - inter
    matrix = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    np.fill_diagonal(matrix, 1.0)
    deg = matrix.sum(axis=1)
    deg = np.where(deg > 0, deg, 1.0)
    return matrix, np.diag(1.0 / deg).astype(np.float32)

def data_masks(all_sessions, n_node):
    adj = dict()
    for sess in all_sessions:
        for i, item in enumerate(sess):
            if i == len(sess)-1:
                break
            else:
                if sess[i] - 1 not in adj.keys():
                    adj[sess[i]-1] = dict()
                    adj[sess[i]-1][sess[i]-1] = 1
                    adj[sess[i]-1][sess[i+1]-1] = 1
                else:
                    if sess[i+1]-1 not in adj[sess[i]-1].keys():
                        adj[sess[i] - 1][sess[i + 1] - 1] = 1
                    else:
                        adj[sess[i]-1][sess[i+1]-1] += 1
    row, col, data = [], [], []
    for i in adj.keys():
        item = adj[i]
        for j in item.keys():
            row.append(i)
            col.append(j)
            data.append(adj[i][j])
    coo = coo_matrix((data, (row, col)), shape=(n_node, n_node))
    return coo

class Data():
    def __init__(self, data, all_train, shuffle=False, n_node=None):
        self.raw = np.array(data[0], dtype=object)
        adj = data_masks(all_train, n_node)
        # # print(adj.sum(axis=0))
        self.adjacency = adj.multiply(1.0/adj.sum(axis=0).reshape(1, -1))
        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle

    def get_overlap(self, sessions):
        unique_lists = [list(dict.fromkeys(int(x) for x in s if x)) for s in sessions]
        return _jaccard_overlap(unique_lists)

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length-batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        mask = []
        # item_set = set()
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            # item_set.update(set([t-1 for t in session]))
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])
        # item_set = list(item_set)
        # index_list = [item_set.index(a) for a in self.targets[index]-1]
        diff_mask = np.ones(shape=[100, self.n_node]) * (1/(self.n_node - 1))
        for count, value in enumerate(self.targets[index]-1):
            diff_mask[count][value] = 1
        return self.targets[index]-1, session_len,items, reversed_sess_item, mask, diff_mask
