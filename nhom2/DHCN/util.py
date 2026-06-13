import numpy as np
import torch
from scipy.sparse import csr_matrix
from operator import itemgetter

def data_masks(all_sessions, n_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i]-1)
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node))

    return matrix

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def _jaccard_overlap(unique_lists):
    """Ma trận Jaccard (n,n) + degree — scipy CSR trên CPU, ~ms cho batch 100-256."""
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


class Data():
    def __init__(self, data, shuffle=False, n_node=None):
        self.raw = np.array(data[0], dtype=object)
        self._unique = [
            list(dict.fromkeys(int(x) for x in s if x))
            for s in self.raw
        ]
        H_T = data_masks(self.raw, n_node)
        BH_T = H_T.T.multiply(1.0/H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        DH = H.T.multiply(1.0/H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        DHBH_T = np.dot(DH,BH_T)

        self.adjacency = DHBH_T.tocoo()
        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle

    def get_overlap(self, sessions):
        unique_lists = [list(dict.fromkeys(x for x in s if x)) for s in sessions]
        return _jaccard_overlap(unique_lists)

    def get_overlap_tensors(self, row_indices, device=None):
        """Overlap line-graph: scipy CPU rồi copy lên GPU (nhanh hơn build tensor trên GPU)."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        unique_lists = [self._unique[int(idx)] for idx in row_indices]
        matrix, degree = _jaccard_overlap(unique_lists)
        A = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        D = torch.as_tensor(degree, dtype=torch.float32, device=device)
        return A, D

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            self._unique = [self._unique[i] for i in shuffled_arg]
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
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1]*len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])


        return self.targets[index]-1, session_len,items, reversed_sess_item, mask
