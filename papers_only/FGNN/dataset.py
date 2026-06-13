# -*- coding: utf-8 -*-
"""
Created on 31/3/2019
@author: RuihongQiu
"""

import os
import pickle
import torch
import collections
from torch_geometric.data import InMemoryDataset, Data


class MultiSessionsGraph(InMemoryDataset):
    """Every session is a graph."""
    def __init__(self, root, phrase, transform=None, pre_transform=None):
        assert phrase in ['train', 'test']
        self.phrase = phrase
        super(MultiSessionsGraph, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        """Data retailrocket nằm thẳng trong root/, không có thư mục raw/."""
        return self.root
     
    @property
    def raw_file_names(self):
        return [self.phrase + '.txt']
    
    @property
    def processed_file_names(self):
        return [self.phrase + '.pt']
    
    def download(self):
        pass
    
    def process(self):
        data = pickle.load(open(self.raw_dir + '/' + self.raw_file_names[0], 'rb'))
        data_list = []
        limit = int(os.environ.get('NCS_SMOKE_SAMPLES', '0')) if os.environ.get('NCS_SMOKE') else 0

        for i, (sequence, y) in enumerate(zip(data[0], data[1])):
            if limit and i >= limit:
                break
            # sequence = [1, 3, 2, 2, 1, 3, 4]
            i = 0
            nodes = {}    # dict{15: 0, 16: 1, 18: 2, ...}
            senders = []
            x = []
            for node in sequence:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]
            
            if len(senders) != 1:
                del senders[-1]    # the last item is a receiver
                del receivers[0]    # the first item is a sender

            # undirected
            # senders, receivers = senders + receivers, receivers + senders

            pair = {}
            sur_senders = senders[:]
            sur_receivers = receivers[:]
            i = 0
            for sender, receiver in zip(sur_senders, sur_receivers):
                if str(sender)+'-'+str(receiver) in pair:
                    pair[str(sender)+'-'+str(receiver)] += 1
                    del senders[i]
                    del receivers[i]
                else:
                    pair[str(sender)+'-'+str(receiver)] = 1
                    i += 1

            count = collections.Counter(senders)
            out_degree_inv = torch.tensor([1/count[i] for i in senders], dtype=torch.float)

            count = collections.Counter(receivers)
            in_degree_inv = torch.tensor([1 / count[i] for i in receivers], dtype=torch.float)
            
            edge_attr = torch.tensor([pair[str(senders[i])+'-'+str(receivers[i])] for i in range(len(senders))],
                                     dtype=torch.float)

            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor([y], dtype=torch.long)
            sequence = torch.tensor(sequence, dtype=torch.long)
            sequence_len = torch.tensor([len(sequence)], dtype=torch.long)
            session_graph = Data(x=x, y=y,
                                 edge_index=edge_index, edge_attr=edge_attr,
                                 sequence=sequence, sequence_len=sequence_len,
                                 out_degree_inv=out_degree_inv, in_degree_inv=in_degree_inv)
            data_list.append(session_graph)
            
        self.save(data_list, self.processed_paths[0])
