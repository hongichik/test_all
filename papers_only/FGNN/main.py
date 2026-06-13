# -*- coding: utf-8 -*-
"""
Created on 4/4/2019
@author: RuihongQiu
"""

import os
import sys
import argparse
import logging
import torch
import time
from pathlib import Path
from tqdm import tqdm
from dataset import MultiSessionsGraph

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from ncs_data import count_nodes_from_pickle, load_pickle

from torch_geometric.data import DataLoader
from set2set_model import Set2SetModel

from train import forward

try:
    from tensorboardX import SummaryWriter
except ImportError:
    class SummaryWriter:  # noqa: D101 — stub khi thiếu tensorboardX/protobuf
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass


# Logger configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--top_k', type=int, default=20, help='top K indicator for evaluation')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
opt = parser.parse_args()
logging.warning(opt)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cur_dir = os.getcwd()
    ds_root = cur_dir + '/datasets/' + opt.dataset
    train_dataset = MultiSessionsGraph(ds_root, phrase='train')
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataset = MultiSessionsGraph(ds_root, phrase='test')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    log_dir = cur_dir + '/../log/' + str(opt.dataset) + '/' + str(opt) + '_s2s3_linear_gat8-1_noleaky_' + time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.warning('logging to {}'.format(log_dir))
    writer = SummaryWriter(log_dir)

    if opt.dataset == 'diginetica':
        n_node = 43097
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37483
    elif opt.dataset == 'retailrocket':
        tr = load_pickle('FGNN', opt.dataset, 'train.txt', f'papers_only/FGNN/datasets/{opt.dataset}/train.txt')
        te = load_pickle('FGNN', opt.dataset, 'test.txt', f'papers_only/FGNN/datasets/{opt.dataset}/test.txt')
        n_node = count_nodes_from_pickle(tr, te)
    else:
        n_node = 309

    # model = GNNModel(hidden_size=opt.hidden_size, n_node=n_node).to(device)
    # model = SortPoolModel(hidden_size=opt.hidden_size, n_node=n_node).to(device)
    model = Set2SetModel(hidden_size=opt.hidden_size, n_node=n_node).to(device)
    # model = GINSet2SetModel(hidden_size=opt.hidden_size, n_node=n_node).to(device)
    # model = VirtualNodeModel(hidden_size=opt.hidden_size, n_node=n_node).to(device)
    # model = Set2SetATTModel(hidden_size=opt.hidden_size, n_node=n_node).to(device)
    # model = VirtualNodeRNNModel(hidden_size=opt.hidden_size, n_node=n_node).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.l2, momentum=opt.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

    logging.warning(model)
    
    for epoch in tqdm(range(opt.epoch)):
        scheduler.step()
        forward(model, train_loader, device, writer, epoch, top_k=opt.top_k, optimizer=optimizer, train_flag=True)
        with torch.no_grad():
            forward(model, test_loader, device, writer, epoch, top_k=opt.top_k, train_flag=False)


if __name__ == '__main__':
    main()
