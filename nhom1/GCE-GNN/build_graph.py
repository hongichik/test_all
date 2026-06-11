import pickle
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Tmall/Nowplaying/retailrocket')
parser.add_argument('--sample_num', type=int, default=12)
opt = parser.parse_args()

dataset = opt.dataset
sample_num = opt.sample_num

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
from ncs_data import load_pickle
from ncs_paths import data_dir

legacy = f'nhom1/GCE-GNN/datasets/{dataset}'
seq = load_pickle('GCE-GNN', dataset, 'all_train_seq.txt', f'{legacy}/all_train_seq.txt')
num = max(max(s) for s in seq) + 1 if seq else 3
out_dir = data_dir('GCE-GNN', dataset)
out_dir.mkdir(parents=True, exist_ok=True)

relation = []
neighbor = [] * num

all_test = set()

adj1 = [dict() for _ in range(num)]
adj = [[] for _ in range(num)]

for i in range(len(seq)):
    data = seq[i]
    for k in range(1, 4):
        for j in range(len(data)-k):
            relation.append([data[j], data[j+k]])
            relation.append([data[j+k], data[j]])

for tup in relation:
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1

weight = [[] for _ in range(num)]

for t in range(num):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj[t] = [v[0] for v in x]
    weight[t] = [v[1] for v in x]

for i in range(num):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]

pickle.dump(adj, open(out_dir / f'adj_{sample_num}.pkl', 'wb'))
pickle.dump(weight, open(out_dir / f'num_{sample_num}.pkl', 'wb'))
print(f'Saved graph -> {out_dir}')
