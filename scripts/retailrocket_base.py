"""
Chuyển đổi RetailRocket gốc (Data/datagoc/Retailrocket/) sang các format huấn luyện.
"""
from __future__ import annotations

import csv
import datetime
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

from ncs_paths import RAW_RETAILROCKET


def is_same_day(t1: float, t2: float) -> bool:
    if t1 >= 1e10:
        t1 /= 1000
    if t2 >= 1e10:
        t2 /= 1000
    return datetime.datetime.fromtimestamp(t1).date() == datetime.datetime.fromtimestamp(t2).date()


def build_sessions(events_csv: Path) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    user2seq: Dict[str, List[Tuple[int, str]]] = {}
    with open(events_csv, "r", newline="") as f:
        for row in csv.DictReader(f):
            if row["event"] != "view":
                continue
            user2seq.setdefault(row["visitorid"], []).append((int(row["timestamp"]), row["itemid"]))

    for uid in user2seq:
        user2seq[uid].sort(key=lambda x: x[0])

    sess_clicks: Dict[str, List[str]] = {}
    sess_date: Dict[str, float] = {}
    sid = 0
    for clicks in user2seq.values():
        if not clicks:
            continue
        sid += 1
        cur = str(sid)
        sess_clicks[cur] = [clicks[0][1]]
        last_ts = clicks[0][0]
        for ts, item in clicks[1:]:
            if not is_same_day(last_ts, ts):
                sess_date[cur] = last_ts / 1000.0
                sid += 1
                cur = str(sid)
                sess_clicks[cur] = []
            sess_clicks[cur].append(item)
            last_ts = ts
        sess_date[cur] = last_ts / 1000.0
    return sess_clicks, sess_date


def filter_sessions(
    sess_clicks: Dict[str, List[str]],
    sess_date: Dict[str, float],
    min_item_freq: int = 5,
) -> Tuple[Dict[str, List[str]], Dict[str, float], Dict[str, int]]:
    def drop_session(sid: str) -> None:
        sess_clicks.pop(sid, None)
        sess_date.pop(sid, None)

    for s in list(sess_clicks):
        if len(sess_clicks[s]) == 1:
            drop_session(s)

    iid_counts: Dict[str, int] = {}
    for seq in sess_clicks.values():
        for iid in seq:
            iid_counts[iid] = iid_counts.get(iid, 0) + 1

    for s in list(sess_clicks):
        fil = [i for i in sess_clicks[s] if iid_counts.get(i, 0) >= min_item_freq]
        if len(fil) < 2:
            drop_session(s)
        else:
            sess_clicks[s] = fil
    return sess_clicks, sess_date, iid_counts


def temporal_split(
    sess_date: Dict[str, float],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]], List[Tuple[str, float]]]:
    dates = sorted(sess_date.items(), key=lambda x: x[1])
    n = len(dates)
    t1, t2 = int(n * train_ratio), int(n * (train_ratio + valid_ratio))
    return dates[:t1], dates[t1:t2], dates[t2:]


def encode_sessions(
    sess_list: List[Tuple[str, float]],
    sess_clicks: Dict[str, List[str]],
    item_dict: Dict[str, int],
    allow_new: bool,
) -> Tuple[List[List[int]], int]:
    seqs = []
    ctr = max(item_dict.values(), default=0) + 1
    for sid, _ in sess_list:
        enc = []
        for raw in sess_clicks[sid]:
            if raw in item_dict:
                enc.append(item_dict[raw])
            elif allow_new:
                item_dict[raw] = ctr
                enc.append(ctr)
                ctr += 1
        if len(enc) >= 2:
            seqs.append(enc)
    return seqs, ctr


def encode_sessions_with_cat(
    sess_list: List[Tuple[str, float]],
    sess_clicks: Dict[str, List[str]],
    item_dict: Dict[str, int],
    cat_dict: Dict[str, int],
    item_to_cat: Dict[str, str],
    allow_new: bool,
) -> List[Tuple[List[int], List[int]]]:
    pairs = []
    ictr = max(item_dict.values(), default=0) + 1
    cctr = max(cat_dict.values(), default=0) + 1
    for sid, _ in sess_list:
        items, cats = [], []
        for raw in sess_clicks[sid]:
            cat_raw = item_to_cat.get(raw, "0")
            if allow_new:
                if cat_raw not in cat_dict:
                    cat_dict[cat_raw] = cctr
                    cctr += 1
                if raw not in item_dict:
                    item_dict[raw] = ictr
                    ictr += 1
                items.append(item_dict[raw])
                cats.append(cat_dict[cat_raw])
            else:
                if raw not in item_dict:
                    continue
                items.append(item_dict[raw])
                cats.append(cat_dict.get(cat_raw, 0))
        if len(items) >= 2:
            pairs.append((items, cats))
    return pairs


def prefix_labels(seqs: List[List[int]], max_len: int | None = None) -> Tuple[List[List[int]], List[int]]:
    out, labs = [], []
    for seq in seqs:
        for i in range(1, len(seq)):
            p = seq[:-i]
            if max_len:
                p = p[-max_len:]
            out.append(p)
            labs.append(seq[-i])
    return out, labs


def prefix_labels_with_cat(pairs: List[Tuple[List[int], List[int]]]) -> Tuple[List, List, List, List]:
    ip, il, cp, cl = [], [], [], []
    for items, cats in pairs:
        for i in range(1, len(items)):
            ip.append(items[:-i])
            il.append(items[-i])
            cp.append(cats[:-i])
            cl.append(cats[-i])
    return ip, il, cp, cl


def load_item_categories(*parts: Path) -> Dict[str, str]:
    best: Dict[str, Tuple[int, str]] = {}
    for path in parts:
        if not path.exists():
            continue
        with open(path, "r", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("property") != "categoryid":
                    continue
                item, ts, cat = row["itemid"], int(row["timestamp"]), row["value"]
                if item not in best or ts > best[item][0]:
                    best[item] = (ts, cat)
    return {k: v[1] for k, v in best.items()}


def load_raw(raw_dir: Path | None = None) -> Tuple[Dict, Dict, List, List, List, int]:
    raw = raw_dir or RAW_RETAILROCKET
    events = raw / "events.csv"
    if not events.exists():
        raise FileNotFoundError(f"Thiếu {events}. Đặt CSV gốc tại Data/datagoc/Retailrocket/")

    sess_clicks, sess_date = build_sessions(events)
    sess_clicks, sess_date, _ = filter_sessions(sess_clicks, sess_date)
    train_s, valid_s, test_s = temporal_split(sess_date)

    item_dict: Dict[str, int] = {}
    tra, _ = encode_sessions(train_s, sess_clicks, item_dict, True)
    val, _ = encode_sessions(valid_s, sess_clicks, item_dict, False)
    tes, n_items = encode_sessions(test_s, sess_clicks, item_dict, False)
    return sess_clicks, item_dict, tra, val, tes, n_items


def save_pickle_session(out_dir: Path, combine_valid_into_train: bool = True) -> int:
    _, _, tra, val, tes, n_items = load_raw()
    all_train = tra + val if combine_valid_into_train else tra
    tr_seqs, tr_labs = prefix_labels(all_train)
    te_seqs, te_labs = prefix_labels(tes)
    out_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump((tr_seqs, tr_labs), open(out_dir / "train.txt", "wb"))
    pickle.dump((te_seqs, te_labs), open(out_dir / "test.txt", "wb"))
    pickle.dump(tra, open(out_dir / "all_train_seq.txt", "wb"))
    print(f"[pickle] {out_dir}: train={len(tr_seqs)}, test={len(te_seqs)}, items≈{n_items}")
    return n_items


def save_core_inter(out_dir: Path, dataset_name: str = "retailrocket") -> None:
    _, _, tra, val, tes, n_items = load_raw()
    splits = [
        ("train", *prefix_labels(tra, 50)),
        ("valid", *prefix_labels(val, 50)),
        ("test", *prefix_labels(tes, 50)),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, seqs, labs in splits:
        with open(out_dir / f"{dataset_name}.{name}.inter", "w") as f:
            f.write("session_id:token\titem_id_list:token_seq\titem_id:token\n")
            for i, (seq, lab) in enumerate(zip(seqs, labs)):
                f.write(f"{n + i + 1}\t{' '.join(map(str, seq))}\t{lab}\n")
        n += len(seqs)
    print(f"[CORE] {out_dir}: interactions={n}, items≈{n_items}")


def save_duorec_inter(out_dir: Path, dataset_name: str = "retailrocket") -> None:
    raw = RAW_RETAILROCKET
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset_name}.inter"
    n = 0
    with open(raw / "events.csv", "r", newline="") as fin, open(path, "w") as fout:
        reader = csv.DictReader(fin)
        fout.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for row in reader:
            if row["event"] != "view":
                continue
            fout.write(f"{row['visitorid']}\t{row['itemid']}\t1\t{row['timestamp']}\n")
            n += 1
    print(f"[DuoRec] {path}: {n} interactions")


def save_csgnn_pickle(out_dir: Path) -> None:
    raw = RAW_RETAILROCKET
    item_to_cat = load_item_categories(
        raw / "item_properties_part1.csv",
        raw / "item_properties_part2.csv",
    )
    sess_clicks, sess_date = build_sessions(raw / "events.csv")
    sess_clicks, sess_date, _ = filter_sessions(sess_clicks, sess_date)
    train_s, valid_s, test_s = temporal_split(sess_date)

    item_dict: Dict[str, int] = {}
    cat_dict: Dict[str, int] = {}
    train_pairs = encode_sessions_with_cat(train_s, sess_clicks, item_dict, cat_dict, item_to_cat, True)
    train_pairs += encode_sessions_with_cat(valid_s, sess_clicks, item_dict, cat_dict, item_to_cat, False)
    test_pairs = encode_sessions_with_cat(test_s, sess_clicks, item_dict, cat_dict, item_to_cat, False)
    tr = prefix_labels_with_cat(train_pairs)
    te = prefix_labels_with_cat(test_pairs)

    id_dir = out_dir / "id"
    id_dir.mkdir(parents=True, exist_ok=True)
    pickle.dump((tr[0], tr[1]), open(id_dir / "train.txt", "wb"))
    pickle.dump((te[0], te[1]), open(id_dir / "test.txt", "wb"))
    pickle.dump((tr[2], tr[3]), open(id_dir / "category_train.txt", "wb"))
    pickle.dump((te[2], te[3]), open(id_dir / "category_test.txt", "wb"))
    print(f"[CSGNN] {id_dir}: train={len(tr[0])}, test={len(te[0])}, items={len(item_dict)}, cats={len(cat_dict)}")


def build_global_graph(all_train_seq: List[List[int]], sample_num: int = 12) -> Tuple[list, list]:
    num = max(max(s) for s in all_train_seq) + 1
    adj1 = [dict() for _ in range(num)]
    for seq in all_train_seq:
        for k in range(1, 4):
            for j in range(len(seq) - k):
                for a, b in ((seq[j], seq[j + k]), (seq[j + k], seq[j])):
                    adj1[a][b] = adj1[a].get(b, 0) + 1
    adj, weights = [[] for _ in range(num)], [[] for _ in range(num)]
    for i in range(num):
        if not adj1[i]:
            continue
        top = sorted(adj1[i].items(), key=lambda x: x[1], reverse=True)[:sample_num]
        adj[i] = [t[0] for t in top]
        weights[i] = [t[1] for t in top]
    return adj, weights


def save_gce_graph(out_dir: Path, sample_num: int = 12) -> None:
    all_train = pickle.load(open(out_dir / "all_train_seq.txt", "rb"))
    adj, num = build_global_graph(all_train, sample_num)
    pickle.dump(adj, open(out_dir / f"adj_{sample_num}.pkl", "wb"))
    pickle.dump(num, open(out_dir / f"num_{sample_num}.pkl", "wb"))
    print(f"[GCE-GNN] {out_dir}: adj_{sample_num}.pkl, nodes={len(adj)}")
