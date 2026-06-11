# NCS — Session-based Recommendation (GNN & Contrastive Learning)

Tổng hợp mã nguồn GitHub và bài báo liên quan cho các thuật toán session-based recommendation.

## Cấu trúc thư mục

| Nhóm | Thư mục | Tên đầy đủ | GitHub | Bài báo |
|------|---------|------------|--------|---------|
| **nhom1/** | `SR-GNN` | Session-based Recommendation with Graph Neural Networks (AAAI 2019) | [CRIPAC-DIG/SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN) | `papers/SR-GNN_AAAI2019.pdf` |
| **nhom1/** | `GCE-GNN` | Global Context Enhanced GNN (SIGIR 2020) | [CCIIPLab/GCE-GNN](https://github.com/CCIIPLab/GCE-GNN) | `papers/GCE-GNN_SIGIR2020.pdf` |
| **nhom2/** | `DHCN` | S²-DHCN — Self-Supervised Hypergraph CNN (AAAI 2021) | [xiaxin1998/DHCN](https://github.com/xiaxin1998/DHCN) | `papers/S2-DHCN_AAAI2021.pdf` |
| **nhom2/** | `COTREC` | Self-Supervised Graph Co-Training (CIKM 2021) | [xiaxin1998/COTREC](https://github.com/xiaxin1998/COTREC) | `papers/COTREC_CIKM2021.pdf` |
| **nhom2/** | `CSGNN` | Category-aware Self-supervised GNN (WWW 2024) | [HduDBSI/CSGNN](https://github.com/HduDBSI/CSGNN) | `papers/PAPER_INFO.txt` (paywall) |
| **nhom3/** | `DuoRec` | CL4SRec (ICDE 2022) + DuoRec (WWW 2022) | [RuihongQiu/DuoRec](https://github.com/RuihongQiu/DuoRec) | `papers/CL4SRec_*.pdf`, `papers/DuoRec_*.pdf` |
| **nhom3/** | `SelfContrastiveLearningRecSys` | SCL — Self Contrastive Learning (ECIR 2024) | [ZhengxiangShi/SelfContrastiveLearningRecSys](https://github.com/ZhengxiangShi/SelfContrastiveLearningRecSys) | `papers/SCL_ECIR2024.pdf` |
| **nhom3/** | `CORE` | Consistent Representation Space (SIGIR 2022) | [RUCAIBox/CORE](https://github.com/RUCAIBox/CORE) | `papers/CORE_SIGIR2022.pdf` |
| **nhom3/** | `NCL` | Neighborhood-enriched CL (WWW 2022) | [RUCAIBox/NCL](https://github.com/RUCAIBox/NCL) | `papers/NCL_WWW2022.pdf` |

### Bài liên quan (`papers_only/`)

| Thư mục | Mô tả | GitHub | Ghi chú |
|---------|-------|--------|---------|
| `HGCAN` | Heterogeneous Graph + Category-aware Attention | [resistzzz/HGCAN](https://github.com/resistzzz/HGCAN) | PDF paywall (Elsevier) |
| `IAGNN` | Intention Adaptive GNN — Category-aware SBR (DASFAA 2022) | [strawhatboy/IAGNN](https://github.com/strawhatboy/IAGNN) | `papers/IAGNN_DASFAA2022.pdf` |
| `CM-HGNN` | Category-aware Multi-relation HGNN (KBS 2022) | [yangbo1973/CM-HGNN](https://github.com/yangbo1973/CM-HGNN) | `papers/` trong Cat-GNN |
| `CCT-GNN` | Collaborative Category and Time-aware GNN (WWW 2025) | [Moosazadeh/CCT-GNN](https://github.com/Moosazadeh/CCT-GNN) | — |
| `FGNN` | Rethinking Item Order (CIKM 2019) — baseline GNN | [RuihongQiu/FGNN](https://github.com/RuihongQiu/FGNN) | `papers/FGNN_CIKM2019.pdf` |
| `Cat-GNN` | Category message passing (không có repo chính thức) | N/A | Bài liên quan category đã tải |
| `CDT-GNN` | Category-integrated Dual-Task GNN (ESWA 2025) | N/A | `papers/PAPER_INFO.txt` (paywall) |
| `GNN-Survey-Index` | Survey & index GNN recommender | [tsinghua-fib-lab/GNN-Recommender-Systems](https://github.com/tsinghua-fib-lab/GNN-Recommender-Systems) | — |

## Bài báo đã tải (PDF)

Mỗi thư mục có subfolder `papers/` chứa PDF từ arXiv hoặc nguồn mở:

- **20 file PDF** (chủ yếu arXiv)
- Bài paywall (CSGNN, HGCAN, CDT-GNN): xem `papers/PAPER_INFO.txt` kèm DOI

### Bài liên quan bổ sung

- `DuoRec/papers/EC4SRec_CIKM2022.pdf` — Explanation Guided Contrastive Learning
- `COTREC/papers/SimCGNN_related_2023.pdf` — Simple Contrastive GNN
- `FGNN/papers/GNN_SBR_Benchmark_2023.pdf` — So sánh benchmark GNN-SBR
- `FGNN/papers/MGCT_related_2024.pdf` — Multi-Graph Co-Training

## Cách dùng

```bash
# Ví dụ chạy SR-GNN
cd nhom1/SR-GNN/pytorch_code
python main.py --dataset diginetica

# Ví dụ chạy DHCN
cd nhom2/DHCN
python main.py --dataset Tmall
```

## Lưu ý

- Một số repo (DHCN, COTREC) dùng dataset pickle từ [Dropbox](https://www.dropbox.com/sh/j12um64gsig5wqk/AAD4Vov6hUGwbLoVxh3wASg_a?dl=0).
- `Cat-GNN` trong danh sách gốc không có mã công khai; thư mục `papers_only/Cat-GNN/` chứa các bài category-GNN liên quan.
- `HGCAN` thực tế **có** mã tại GitHub (`resistzzz/HGCAN`), khác với ghi chú "N/A" ban đầu.
# test_all
