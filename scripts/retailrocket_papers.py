"""
Chuyển RetailRocket -> format cho papers_only (FGNN, CM-HGNN, CCT-GNN, HGCAN).
Output: Data/<TênPaper>/retailrocket/
"""
from __future__ import annotations

import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ncs_paths import REPO_ROOT, data_dir
from scripts.retailrocket_base import (
    RAW_RETAILROCKET,
    build_sessions,
    encode_sessions_with_cat,
    filter_sessions,
    load_item_categories,
    prefix_labels,
    prefix_labels_with_cat,
    save_gce_graph,
    save_pickle_session,
    temporal_split,
)

PAPER_LINKS = {
    "FGNN": REPO_ROOT / "papers_only" / "FGNN" / "datasets" / "retailrocket",
    "CM-HGNN": REPO_ROOT / "papers_only" / "CM-HGNN" / "datasets" / "retailrocket",
    "CCT-GNN": REPO_ROOT / "papers_only" / "CCT-GNN" / "datasets" / "retailrocket",
    "HGCAN": REPO_ROOT / "papers_only" / "HGCAN" / "datasets" / "retailrocket",
}


def _load_cat_sessions() -> Tuple[
    Dict[str, List[str]],
    Dict[str, float],
    List[Tuple[str, float]],
    List[Tuple[str, float]],
    List[Tuple[str, float]],
    Dict[str, str],
]:
    raw = RAW_RETAILROCKET
    item_to_cat = load_item_categories(
        raw / "item_properties_part1.csv",
        raw / "item_properties_part2.csv",
    )
    sess_clicks, sess_date = build_sessions(raw / "events.csv")
    sess_clicks, sess_date, _ = filter_sessions(sess_clicks, sess_date)
    train_s, valid_s, test_s = temporal_split(sess_date)
    return sess_clicks, sess_date, train_s, valid_s, test_s, item_to_cat


def _item_category_map(
    item_dict: Dict[str, int],
    cat_dict: Dict[str, int],
    item_to_cat: Dict[str, str],
) -> Dict[int, int]:
    return {
        item_dict[raw]: cat_dict.get(item_to_cat.get(raw, "0"), 0)
        for raw in item_dict
    }


def _cct_process_seqs(
    seqs: List[List[int]],
    dates: List[float],
) -> Tuple[List[int], List[List[int]], List[int], List[float]]:
    ids, out_seqs, labs, out_dates = [], [], [], []
    idx = 0
    for seq, date in zip(seqs, dates):
        for i in range(1, len(seq)):
            out_seqs.append(seq[:-i])
            labs.append(seq[-i])
            out_dates.append(date)
            ids.append(idx)
        idx += 1
    return ids, out_seqs, labs, out_dates


def link_papers_datasets() -> None:
    """Symlink Data/<Paper>/retailrocket -> papers_only/<Paper>/datasets/retailrocket"""
    for project, dst in PAPER_LINKS.items():
        src = data_dir(project, "retailrocket").resolve()
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        rel = Path("../../../Data") / project / "retailrocket"
        dst.symlink_to(rel)
        print(f"[link] {dst} -> {rel}")


def save_fgnn(out_dir: Path | None = None) -> int:
    out = out_dir or data_dir("FGNN", "retailrocket")
    return save_pickle_session(out)


def save_cm_hgnn(out_dir: Path | None = None, sample_num: int = 12) -> None:
    out = out_dir or data_dir("CM-HGNN", "retailrocket")
    n_items = save_pickle_session(out)

    sess_clicks, _, train_s, valid_s, _, item_to_cat = _load_cat_sessions()
    item_dict: Dict[str, int] = {}
    cat_dict: Dict[str, int] = {}
    encode_sessions_with_cat(train_s, sess_clicks, item_dict, cat_dict, item_to_cat, True)
    encode_sessions_with_cat(valid_s, sess_clicks, item_dict, cat_dict, item_to_cat, False)

    category = _item_category_map(item_dict, cat_dict, item_to_cat)
    pickle.dump(category, open(out / "category.txt", "wb"))
    save_gce_graph(out, sample_num)
    print(f"[CM-HGNN] {out}: items≈{n_items}, cats={len(cat_dict)}")


def save_hgcan(out_dir: Path | None = None, sample_num: int = 12, theta: int = 2) -> None:
    out = out_dir or data_dir("HGCAN", "retailrocket")
    out.mkdir(parents=True, exist_ok=True)

    sess_clicks, _, train_s, valid_s, test_s, item_to_cat = _load_cat_sessions()
    item_dict: Dict[str, int] = {}
    cat_dict: Dict[str, int] = {}

    train_pairs = encode_sessions_with_cat(
        train_s, sess_clicks, item_dict, cat_dict, item_to_cat, True
    )
    train_pairs += encode_sessions_with_cat(
        valid_s, sess_clicks, item_dict, cat_dict, item_to_cat, False
    )
    test_pairs = encode_sessions_with_cat(
        test_s, sess_clicks, item_dict, cat_dict, item_to_cat, False
    )

    tra = [p[0] for p in train_pairs]
    tes = [p[0] for p in test_pairs]
    tra_cat = [p[1] for p in train_pairs]
    tes_cat = [p[1] for p in test_pairs]

    tr_seqs, tr_labs = prefix_labels(tra)
    te_seqs, te_labs = prefix_labels(tes)
    tr_cat = prefix_labels_with_cat(train_pairs)
    te_cat = prefix_labels_with_cat(test_pairs)

    pickle.dump((tr_seqs, tr_labs), open(out / "train.txt", "wb"))
    pickle.dump((te_seqs, te_labs), open(out / "test.txt", "wb"))
    pickle.dump((tr_cat[2], tr_cat[3]), open(out / "train_cat.txt", "wb"))
    pickle.dump((te_cat[2], te_cat[3]), open(out / "test_cat.txt", "wb"))
    pickle.dump(tra, open(out / "all_train_seq.txt", "wb"))
    pickle.dump(tra_cat, open(out / "all_train_seq_cate.txt", "wb"))

    item2cate = _item_category_map(item_dict, cat_dict, item_to_cat)
    cate2item: Dict[int, List[int]] = defaultdict(list)
    for item, cat in item2cate.items():
        if item not in cate2item[cat]:
            cate2item[cat].append(item)
    cate2item = {k: v for k, v in sorted(cate2item.items())}

    pickle.dump(cate2item, open(out / "cate2item.txt", "wb"))
    pickle.dump(item2cate, open(out / "item2cate.txt", "wb"))

    _build_hgcan_graphs(out, sample_num=sample_num, theta=theta)
    print(
        f"[HGCAN] {out}: train={len(tr_seqs)}, test={len(te_seqs)}, "
        f"items={len(item_dict)}, cats={len(cat_dict)}"
    )


def _build_hgcan_graphs(out_dir: Path, sample_num: int = 12, theta: int = 2) -> None:
    seq = pickle.load(open(out_dir / "all_train_seq.txt", "rb"))
    seq_category = pickle.load(open(out_dir / "all_train_seq_cate.txt", "rb"))
    cate2item = pickle.load(open(out_dir / "cate2item.txt", "rb"))

    num = max(max(s) for s in seq) + 1
    num_cate = max(max(s) for s in seq_category) + 1

    adj1 = [dict() for _ in range(num)]
    adj1_cate = [dict() for _ in range(num_cate)]
    adj = [[] for _ in range(num)]
    adj_cate = [[] for _ in range(num_cate)]
    adj_c2i = [[] for _ in range(num_cate)]

    for i in range(len(cate2item)):
        adj_c2i[i + 1] = cate2item[i + 1]

    relation, relation_cate = [], []
    for data in seq:
        for k in range(1, 2 * theta):
            for j in range(len(data) - k):
                relation.append([data[j], data[j + k]])
                relation.append([data[j + k], data[j]])

    for data_cate in seq_category:
        if len(data_cate) == 1:
            relation_cate.append([data_cate[0], data_cate[0]])
        for k in range(1, 2 * theta):
            for j in range(len(data_cate) - k):
                relation_cate.append([data_cate[j], data_cate[j + k]])
                relation_cate.append([data_cate[j + k], data_cate[j]])

    for a, b in relation:
        adj1[a][b] = adj1[a].get(b, 0) + 1
    for a, b in relation_cate:
        adj1_cate[a][b] = adj1_cate[a].get(b, 0) + 1

    weight = [[] for _ in range(num)]
    weight_c2c = [[] for _ in range(num_cate)]
    weight_c2i = [[] for _ in range(num_cate)]

    for t in range(num):
        top = sorted(adj1[t].items(), key=lambda x: x[1], reverse=True)
        adj[t] = [v[0] for v in top]
        weight[t] = [v[1] for v in top]

    for t in range(num_cate):
        top = sorted(adj1_cate[t].items(), key=lambda x: x[1], reverse=True)
        adj_cate[t] = [v[0] for v in top]
        weight_c2c[t] = [v[1] for v in top]

    for t in range(num_cate):
        length_cate = len(adj_c2i[t])
        if length_cate:
            weight_c2i[t] = [1 / length_cate] * length_cate

    for i in range(num):
        adj[i] = adj[i][:sample_num]
        weight[i] = weight[i][:sample_num]
        if weight[i]:
            s = sum(weight[i])
            if s:
                weight[i] = (np.asarray(weight[i]) / s).tolist()

    for i in range(num_cate):
        adj_cate[i] = adj_cate[i][:sample_num]
        weight_c2c[i] = weight_c2c[i][:sample_num]
        if weight_c2c[i]:
            s = sum(weight_c2c[i])
            if s:
                weight_c2c[i] = (np.asarray(weight_c2c[i]) / s).tolist()

    param = {"num_items": num, "max_length": max(len(s) for s in seq), "num_cate": num_cate}
    pickle.dump(adj, open(out_dir / f"adj_{sample_num}.pkl", "wb"))
    pickle.dump(weight, open(out_dir / f"num_{sample_num}.pkl", "wb"))
    pickle.dump(adj_cate, open(out_dir / f"adj_cate_{sample_num}.pkl", "wb"))
    pickle.dump(weight_c2c, open(out_dir / f"num_cate_{sample_num}.pkl", "wb"))
    pickle.dump(adj_c2i, open(out_dir / f"adj_c2i_{sample_num}.pkl", "wb"))
    pickle.dump(weight_c2i, open(out_dir / f"num_c2i_{sample_num}.pkl", "wb"))
    pickle.dump(param, open(out_dir / "parm.pkl", "wb"))
    print(f"[HGCAN-graph] {out_dir}: items={num}, cats={num_cate}")


def save_cct_gnn(
    out_dir: Path | None = None,
    k_neighbors: int = 200,
    max_adj_dist: int = 2,
    itm_adj_sample: int = 12,
    icl: float = 0.25,
) -> None:
    out = out_dir or data_dir("CCT-GNN", "retailrocket")
    out.mkdir(parents=True, exist_ok=True)

    sess_clicks, _, train_s, valid_s, test_s, item_to_cat = _load_cat_sessions()
    item_dict: Dict[str, int] = {}
    cat_dict: Dict[str, int] = {}

    def encode_sess_list(sess_list, allow_new: bool):
        ids, seqs, dates = [], [], []
        ictr = max(item_dict.values(), default=0) + 1
        for sid, date in sess_list:
            enc = []
            for raw in sess_clicks[sid]:
                if raw in item_dict:
                    enc.append(item_dict[raw])
                elif allow_new:
                    item_dict[raw] = ictr
                    enc.append(ictr)
                    ictr += 1
            if len(enc) >= 2:
                ids.append(len(ids))
                seqs.append(enc)
                dates.append(date)
        return ids, seqs, dates, ictr

    _, train_seqs, train_dates, num_items = encode_sess_list(train_s, True)
    _, val_seqs, val_dates, _ = encode_sess_list(valid_s, False)
    _, test_seqs, test_dates, _ = encode_sess_list(test_s, False)

    train_seqs += val_seqs
    train_dates += val_dates

    item_category: Dict[int, int] = {}
    change: Dict[str, int] = {}
    cat_ctr = num_items
    for raw, item_id in sorted(item_dict.items(), key=lambda x: x[1]):
        cat_raw = item_to_cat.get(raw, "0")
        if cat_raw not in change:
            change[cat_raw] = cat_ctr
            cat_ctr += 1
        item_category[item_id] = change[cat_raw]

    tr = _cct_process_seqs(train_seqs, train_dates)
    te = _cct_process_seqs(test_seqs, test_dates)
    all_train_seq = (list(range(len(train_seqs))), train_seqs, ["Nothing"], train_dates)

    pickle.dump(tr, open(out / "train.txt", "wb"))
    pickle.dump(te, open(out / "test.txt", "wb"))
    pickle.dump(all_train_seq, open(out / "all_train_seq.txt", "wb"))
    pickle.dump(item_category, open(out / "category.txt", "wb"))

    _cct_find_neighbors(out, k_neighbors)
    _cct_build_graphs(out, max_adj_dist, itm_adj_sample, icl)
    print(
        f"[CCT-GNN] {out}: train={len(tr[1])}, test={len(te[1])}, "
        f"items={num_items}, cats={cat_ctr - num_items}"
    )


def _cct_find_neighbors(out_dir: Path, k: int) -> None:
    category = pickle.load(open(out_dir / "category.txt", "rb"))
    all_train_seq = pickle.load(open(out_dir / "all_train_seq.txt", "rb"))
    all_items, all_times = all_train_seq[1], all_train_seq[3]
    train = pickle.load(open(out_dir / "train.txt", "rb"))
    test = pickle.load(open(out_dir / "test.txt", "rb"))

    item_sess_map: Dict[int, List[int]] = {}
    for neigh_idx, neigh_items in zip(all_train_seq[0], all_items):
        for itm in neigh_items:
            item_sess_map.setdefault(itm, []).append(neigh_idx)

    def cosine_similarity(neigh_items, sess_items, item_pos_weight):
        intersection = set(neigh_items) & set(sess_items)
        s = sum(item_pos_weight[itm] ** 2 for itm in neigh_items)
        inter_sum = sum(item_pos_weight[itm] for itm in intersection)
        denom = math.sqrt(s) * math.sqrt(len(neigh_items))
        return np.around(inter_sum / denom, 4) if denom else 0.0

    def find_nearest_neighbors(sess_items, sess_time):
        sess_neigh_idxs = []
        for itm in np.unique(sess_items):
            poss = [p for p in item_sess_map.get(itm, []) if all_times[p] < sess_time]
            if len(poss) <= k:
                itm_knn = poss
            else:
                neigh_indexs, neigh_sims = [], []
                item_pos = list(sess_items).index(itm)
                for nidx in poss:
                    neigh_items = all_items[nidx]
                    item_weight = {
                        nitm: math.exp(-abs(neigh_items.index(nitm) - item_pos))
                        for nitm in neigh_items
                    }
                    sim = cosine_similarity(neigh_items, sess_items, item_weight)
                    sim *= math.exp(-abs(sess_time - all_times[nidx]) / 86400)
                    neigh_indexs.append(nidx)
                    neigh_sims.append(sim)
                order = np.argsort(neigh_sims)[::-1]
                itm_knn = np.array(neigh_indexs)[order][:k].tolist()
            sess_neigh_idxs.append(itm_knn)
        return sess_neigh_idxs

    def dump_neighbors(name, data):
        neigh_map = {}
        for sess_idx in sorted(set(data[0])):
            neigh_map[sess_idx] = find_nearest_neighbors(
                all_items[sess_idx], all_times[sess_idx]
            )
        pickle.dump(neigh_map, open(out_dir / f"{name}_neighbors.txt", "wb"))

    dump_neighbors("train", train)
    dump_neighbors("test", test)
    _ = category


def _cct_build_graphs(
    out_dir: Path,
    max_adj_dist: int,
    itm_adj_sample: int,
    icl: float,
) -> None:
    train = pickle.load(open(out_dir / "train.txt", "rb"))
    test = pickle.load(open(out_dir / "test.txt", "rb"))
    all_train_seq = pickle.load(open(out_dir / "all_train_seq.txt", "rb"))
    all_items = all_train_seq[1]
    category = pickle.load(open(out_dir / "category.txt", "rb"))

    def build_graph(name, data, neighbors):
        total = {}
        for sidx in data[0]:
            if sidx in total:
                continue
            edge_count = {item: {} for item in all_items[sidx]}
            sess_items = all_items[sidx]
            for dist in range(1, max_adj_dist + 1):
                for pos, u in enumerate(sess_items):
                    for sign, score_base in ((1, max_adj_dist + 3 - dist), (-1, max_adj_dist + 1 - dist)):
                        vpos = pos + sign * dist
                        if 0 <= vpos < len(sess_items):
                            v = sess_items[vpos]
                            edge_count[u][v] = edge_count[u].get(v, 0) + score_base
            neigh_entry = neighbors.get(sidx, [])
            nidxs = neigh_entry[0] if neigh_entry else []
            for nidx in nidxs:
                nitems = all_items[nidx]
                for dist in range(1, max_adj_dist + 1):
                    for pos, u in enumerate(nitems):
                        if u not in sess_items:
                            continue
                        for sign, score_base in ((1, max_adj_dist + 3 - dist), (-1, max_adj_dist + 1 - dist)):
                            vpos = pos + sign * dist
                            if 0 <= vpos < len(nitems):
                                v = nitems[vpos]
                                edge_count[u][v] = edge_count[u].get(v, 0) + score_base
                                if category.get(u) == category.get(v):
                                    edge_count[u][v] += icl
            new_adjs = {}
            for itm in sess_items:
                edges = sorted(edge_count[itm].items(), key=lambda x: x[1], reverse=True)
                new_adjs[itm] = [
                    [e[0] for e in edges][:itm_adj_sample],
                    [e[1] for e in edges][:itm_adj_sample],
                ]
            total[sidx] = new_adjs
        pickle.dump(total, open(out_dir / f"{name}_adjs.txt", "wb"))

    train_neighbors = pickle.load(open(out_dir / "train_neighbors.txt", "rb"))
    test_neighbors = pickle.load(open(out_dir / "test_neighbors.txt", "rb"))
    build_graph("train", train, train_neighbors)
    build_graph("test", test, test_neighbors)
