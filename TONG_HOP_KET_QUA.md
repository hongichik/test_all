# Tổng hợp kết quả thí nghiệm — RetailRocket

> Cập nhật: **12-06-2026**  
> Bộ dữ liệu: `retailrocket`  
> Nguồn log: `Log/<tên-bài-toán>/retailrocket/`

---

## 1. Mô tả dữ liệu

| Thuộc tính | Giá trị |
|------------|---------|
| **Tên bộ dữ liệu** | RetailRocket |
| **Loại** | Session-based recommendation (sự kiện *view*) |
| **Nguồn gốc** | `Data/datagoc/Retailrocket/events.csv` |
| **Tiền xử lý** | Gom session theo visitor + cùng ngày; lọc item freq ≥ 5; split theo thời gian |
| **Số session (sau lọc)** | 323,495 |
| **Train / Valid / Test** | 258,796 / 31,576 / 30,660 session |
| **Số item** | 51,428 |
| **Độ dài session TB** | 3.72 click/session |
| **Đường dẫn data** | `Data/<tên-bài-toán>/retailrocket/` |

**Ghi chú:** Các bài toán nhom1/nhom2 dùng format pickle (`train.txt`, `test.txt`, `all_train_seq.txt`). Metric đánh giá trên tập test; mỗi session dự đoán item tiếp theo (leave-one-out / sliding window theo code gốc).

---

## 2. Metric đánh giá

| Metric | Ý nghĩa |
|--------|---------|
| **Recall@K** | Tỉ lệ (%) session mà item đúng nằm trong top-K gợi ý |
| **MRR@K** / **MMR@K** | Mean Reciprocal Rank — thứ hạng trung bình nghịch đảo của item đúng trong top-K (%) |

Trong log, **GCE-GNN** và **SR-GNN** ghi `MMR@20`; **DHCN** ghi `MRR20` — cùng ý nghĩa MRR tại K=20.

---

## 3. Bảng kết quả chính (Recall@20 / MRR@20)

| Hạng | Bài toán | Nhóm | Recall@20 (%) | MRR@20 (%) | Best epoch | Epoch chạy | Thời gian | Trạng thái |
|:---:|----------|:----:|--------------:|-----------:|:----------:|:----------:|:---------:|:----------:|
| **1** | **DHCN** | nhom2 | **62.20** | **37.49** | 26 | 30/30 | ~6.1 giờ | Hoàn thành |
| 2 | GCE-GNN | nhom1 | 62.09 | 37.04 | 11 | 20/20 | ~12.3 giờ | Hoàn thành |
| 3 | SR-GNN | nhom1 | 59.63 | 35.86 | 4 | 15/30 (early stop) | ~5.1 giờ | Hoàn thành |

**Kết luận:** **DHCN** đạt kết quả cao nhất trên cả Recall@20 (+0.11 so với GCE-GNN) và MRR@20 (+0.45). GCE-GNN xếp thứ 2, sát DHCN. SR-GNN thấp hơn ~2.6 điểm Recall@20.

---

## 4. Chi tiết theo bài toán

### 4.1. DHCN (nhom2) — cao nhất

| Metric | Giá trị (%) | Epoch |
|--------|------------:|------:|
| Recall@5 | 47.75 | 24 |
| Recall@10 | 55.19 | 26 |
| **Recall@20** | **62.20** | **26** |
| MRR@5 | 36.00 | 24 |
| MRR@10 | 37.00 | 26 |
| **MRR@20** | **37.49** | **26** |

| Cấu hình | Giá trị |
|----------|---------|
| epoch | 30 |
| batchSize | 100 |
| embSize | 100 |
| lr | 0.001 |
| layer | 3 |
| beta | 0.01 |
| l2 | 1e-5 |

**Log:** `Log/DHCN/retailrocket/log-2026-06-12-13-02-23.log`

---

### 4.2. GCE-GNN (nhom1)

| Metric | Giá trị (%) | Epoch |
|--------|------------:|------:|
| **Recall@20** | **62.09** | **11** |
| **MMR@20** | **37.04** | **11** |

| Cấu hình | Giá trị |
|----------|---------|
| epoch | 20 |
| hiddenSize | 100 |
| batch_size | 100 |
| lr | 0.001 (decay 0.1 / 3 epoch) |
| n_sample / n_sample_all | 12 |
| dropout_gcn / global | 0.2 / 0.5 |
| alpha | 0.2 |
| patience | 3 |

**Log:** `Log/GCE-GNN/retailrocket/log-2026-06-12-03-55-15.log`  
**Run time:** 44,424 s (~12.3 giờ)

---

### 4.3. SR-GNN (nhom1)

| Metric | Giá trị (%) | Epoch |
|--------|------------:|------:|
| **Recall@20** | **59.63** | **4** |
| **MMR@20** | **35.86** | **4** |

| Cấu hình | Giá trị |
|----------|---------|
| epoch (cấu hình) | 30 |
| epoch (thực tế) | 15 (early stop, patience=10) |
| hiddenSize | 100 |
| batchSize | 100 |
| lr | 0.001 (decay 0.1 / 3 epoch) |
| l2 | 1e-5 |

**Log:** `Log/SR-GNN/retailrocket/log-2026-06-12-03-55-15.log`  
**Run time:** 18,454 s (~5.1 giờ)

**Diễn biến Recall@20 theo epoch:**

| Epoch | 0 | 1 | 2 | 3 | 4 |
|------:|--:|--:|--:|--:|--:|
| Recall@20 | 53.50 | 56.96 | 59.35 | 59.57 | **59.63** |

---

## 5. So sánh trực quan

```
Recall@20 (%)
DHCN     ██████████████████████████████████████████████████████████████ 62.20
GCE-GNN  █████████████████████████████████████████████████████████████▊ 62.09
SR-GNN   ███████████████████████████████████████████████████████████▎   59.63

MRR@20 (%)
DHCN     █████████████████████████████████████▌                         37.49
GCE-GNN  ████████████████████████████████████▊                          37.04
SR-GNN   ███████████████████████████████████▌                           35.86
```

| So sánh | Δ Recall@20 | Δ MRR@20 |
|---------|------------:|---------:|
| DHCN vs GCE-GNN | +0.11 | +0.45 |
| DHCN vs SR-GNN | +2.57 | +1.63 |
| GCE-GNN vs SR-GNN | +2.45 | +1.18 |

---

## 6. Tham chiếu log

| Bài toán | Log quá trình | Ghi chú |
|----------|---------------|---------|
| DHCN | `Log/DHCN/retailrocket/log-2026-06-12-13-02-23.log` | Run chính, 30 epoch |
| GCE-GNN | `Log/GCE-GNN/retailrocket/log-2026-06-12-03-55-15.log` | 20 epoch |
| SR-GNN | `Log/SR-GNN/retailrocket/log-2026-06-12-03-55-15.log` | Early stop epoch 14 |
| DHCN (lỗi/dừng sớm) | `Log/DHCN/retailrocket/log-2026-06-12-03-56-58.log` | Không có metric |
| DHCN (dừng sớm) | `Log/DHCN/retailrocket/log-2026-06-12-12-58-58.log` | Chỉ mới epoch 0 |

---

## 7. Ghi chú

- Chỉ gồm **3 bài toán đã chạy xong** trên `retailrocket` tính đến 12-06-2026.
- Các dự án khác (COTREC, CSGNN, DuoRec, CORE, …) chưa có log kết quả trong `Log/`.
- `LogMins/` chưa được ghi — có thể bổ sung sau khi chạy script tổng hợp hoặc thủ công.
- Metric là **tỉ lệ phần trăm** (0–100), không phải xác suất thập phân.
