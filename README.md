# NCS — Session-based Recommendation (GNN & Contrastive Learning)

Tổng hợp mã nguồn cho session-based recommendation trên **RetailRocket**.

**12 thư mục bài toán** (14 target huấn luyện khi tính 3 sub-model SCLRS).

## Cấu trúc thư mục

| Nhóm | Thư mục | Tên đầy đủ | GitHub |
|------|---------|------------|--------|
| **nhom1/** | `SR-GNN` | Session-based Recommendation with GNN (AAAI 2019) | [CRIPAC-DIG/SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN) |
| **nhom1/** | `GCE-GNN` | Global Context Enhanced GNN (SIGIR 2020) | [CCIIPLab/GCE-GNN](https://github.com/CCIIPLab/GCE-GNN) |
| **nhom2/** | `DHCN` | S²-DHCN (AAAI 2021) | [xiaxin1998/DHCN](https://github.com/xiaxin1998/DHCN) |
| **nhom2/** | `COTREC` | Self-Supervised Graph Co-Training (CIKM 2021) | [xiaxin1998/COTREC](https://github.com/xiaxin1998/COTREC) |
| **nhom2/** | `CSGNN` | Category-aware Self-supervised GNN (WWW 2024) | [HduDBSI/CSGNN](https://github.com/HduDBSI/CSGNN) |
| **nhom3/** | `DuoRec` | CL4SRec + DuoRec (ICDE/WWW 2022) | [RuihongQiu/DuoRec](https://github.com/RuihongQiu/DuoRec) |
| **nhom3/** | `CORE` | Consistent Representation Space (SIGIR 2022) | [RUCAIBox/CORE](https://github.com/RUCAIBox/CORE) |
| **nhom3/** | `SCLRS` | Self Contrastive Learning (ECIR 2024) | [ZhengxiangShi/SelfContrastiveLearningRecSys](https://github.com/ZhengxiangShi/SelfContrastiveLearningRecSys) |

### `papers_only/` (RetailRocket)

| Thư mục | Mô tả | GitHub |
|---------|-------|--------|
| `HGCAN` | Heterogeneous Graph + Category-aware Attention | [resistzzz/HGCAN](https://github.com/resistzzz/HGCAN) |
| `CM-HGNN` | Category-aware Multi-relation HGNN (KBS 2022) | [yangbo1973/CM-HGNN](https://github.com/yangbo1973/CM-HGNN) |
| `CCT-GNN` | Collaborative Category and Time-aware GNN (WWW 2025) | [Moosazadeh/CCT-GNN](https://github.com/Moosazadeh/CCT-GNN) |
| `FGNN` | Rethinking Item Order (CIKM 2019) | [RuihongQiu/FGNN](https://github.com/RuihongQiu/FGNN) |

## RetailRocket — quy trình nhanh

```bash
# 1. Tải data gốc
python3 scripts/download_retailrocket.py

# 2. Chuyển đổi tất cả
python3 scripts/preprocess_retailrocket.py -m all --link-papers

# 3. Huấn luyện (ví dụ)
cd nhom1/SR-GNN/pytorch_code && python main.py --dataset retailrocket --epoch 30
```

Chi tiết từng model: xem `chuyendoi.txt`.

## Data & log

- Data gốc: `Data/datagoc/Retailrocket/`
- Data huấn luyện: `Data/<TênBàiToán>/retailrocket/`
- Log: `Log/`, kết quả: `LogMins/` (xem `ncs_paths.py`, `ncs_logging.py`)
