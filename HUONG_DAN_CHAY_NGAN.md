# Hướng dẫn chạy ngầm RetailRocket — từng khối 2–3 dự án

Tài liệu này dành cho trường hợp **không chạy đủ 14 dự án cùng lúc**. Mỗi khối chạy **2 hoặc 3 job song song** (nền), xong khối này mới chuyển khối tiếp theo.

Mọi lệnh giả định bạn đang ở thư mục gốc repo:

```bash
cd ~/test_all   # hoặc đường dẫn tuyệt đối tới test_all
```

---

## 1. Chuẩn bị (chạy một lần)

### 1.1. Data & symlink

```bash
# Data gốc + convert sang 14 format
python3 scripts/download_retailrocket.py          # nếu chưa có Data/datagoc/
python3 scripts/preprocess_retailrocket.py -m all --link-papers

# Symlink RecBole (DuoRec, CORE)
ln -sfn ../../../Data/DuoRec/retailrocket  nhom3/DuoRec/dataset/retailrocket
ln -sfn ../../../Data/CORE/retailrocket  nhom3/CORE/dataset/retailrocket
```

### 1.2. Python dependencies

```bash
pip install -r requirements-smoke.txt
# Thêm torch, torch-geometric, recbole, … theo môi trường bạn đang dùng
```

### 1.3. Quy ước log (NCS)

| Loại | Đường dẫn |
|------|-----------|
| Log quá trình (stdout/stderr đầy đủ) | `Log/<tên-dự-án>/retailrocket/log-YYYY-MM-DD-HH-MM-SS.log` |
| Log kết quả ngắn | `LogMins/<tên-dự-án>/retailrocket/DD-MM-YYYY.log` |

Tên file sort A→Z = cũ→mới → **file log mới nhất nằm dưới cùng** trong thư mục.

### 1.4. Mẫu lệnh chạy ngầm

Dùng biến `LOG` + `nohup` + lưu PID để theo dõi:

```bash
LOG="Log/<PROJECT>/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup <lệnh-huấn-luyện> > "$LOG" 2>&1 &
echo $!   # ghi lại PID nếu cần kill sau
```

Chờ **cả khối** xong trước khi chạy khối kế:

```bash
wait    # chờ mọi job nền của shell hiện tại
```

Hoặc chờ từng PID cụ thể: `wait $PID1 $PID2 $PID3`

---

## 2. Chế độ huấn luyện

### Smoke (nhanh, kiểm tra pipeline)

- Đặt `NCS_SMOKE=1` (và tùy chọn `NCS_SMOKE_SAMPLES=2000`)
- DuoRec/CORE: thêm config `../../config/smoke_1epoch.yaml`
- Epoch = 1

### Full (thí nghiệm thật)

- Bỏ `NCS_SMOKE`
- Dùng epoch trong lệnh bên dưới (có thể chỉnh theo paper)

---

## 3. Chia 5 khối (14 dự án)

```text
Khối 1 │ SR-GNN, GCE-GNN, DHCN              │ 3 job │ nhom1 + nhom2
Khối 2 │ COTREC, CSGNN, DuoRec              │ 3 job │ nhom2 + RecBole
Khối 3 │ CORE, SCL-DHCN, SCL-COTREC         │ 3 job │ RecBole + SCL
Khối 4 │ SCL-GCE-GNN, FGNN, CM-HGNN         │ 3 job │ SCL + papers
Khối 5 │ CCT-GNN, HGCAN                     │ 2 job │ papers
```

> **Gợi ý GPU:** Nếu chỉ có 1 GPU, nên giảm xuống **2 job/khối** hoặc đặt `CUDA_VISIBLE_DEVICES=""` cho job chạy CPU (CORE smoke, CM-HGNN).

---

## 4. Lệnh từng khối (full training)

Copy cả block, dán vào terminal, Enter một lần. Sau `wait`, chạy khối tiếp theo.

### Khối 1 — SR-GNN, GCE-GNN, DHCN

```bash
cd ~/test_all

LOG="Log/SR-GNN/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup python3 nhom1/SR-GNN/pytorch_code/main.py --dataset retailrocket --epoch 30 \
  > "$LOG" 2>&1 &

LOG="Log/GCE-GNN/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup python3 nhom1/GCE-GNN/main.py --dataset retailrocket --epoch 20 \
  > "$LOG" 2>&1 &

LOG="Log/DHCN/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup python3 nhom2/DHCN/main.py --dataset retailrocket --epoch 30 \
  > "$LOG" 2>&1 &

wait
echo "[Khối 1] xong — kiểm tra Log/SR-GNN, Log/GCE-GNN, Log/DHCN"
```

### Khối 2 — COTREC, CSGNN, DuoRec

```bash
cd ~/test_all

LOG="Log/COTREC/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup python3 nhom2/COTREC/main.py --dataset retailrocket --epoch 30 \
  > "$LOG" 2>&1 &

LOG="Log/CSGNN/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup python3 nhom2/CSGNN/main.py --dataset retailrocket --epoch 1 \
  --embSize 100 --beta 0.005 \
  > "$LOG" 2>&1 &

LOG="Log/DuoRec/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup python3 nhom3/DuoRec/run_seq.py --dataset retailrocket --model DuoRec \
  --config_files "seq.yaml DuoRec.yaml" \
  > "$LOG" 2>&1 &

wait
echo "[Khối 2] xong — kiểm tra Log/COTREC, Log/CSGNN, Log/DuoRec"
```

### Khối 3 — CORE, SCL-DHCN, SCL-COTREC

```bash
cd ~/test_all

LOG="Log/CORE/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup python3 nhom3/CORE/main.py --model trm --dataset retailrocket \
  > "$LOG" 2>&1 &

LOG="Log/SCL-DHCN/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup python3 nhom3/SCLRS/DHCN/main.py --dataset retailrocket --epoch 30 \
  > "$LOG" 2>&1 &

LOG="Log/SCL-COTREC/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup python3 nhom3/SCLRS/COTREC/main.py --dataset retailrocket --epoch 30 \
  > "$LOG" 2>&1 &

wait
echo "[Khối 3] xong — kiểm tra Log/CORE, Log/SCL-DHCN, Log/SCL-COTREC"
```

### Khối 4 — SCL-GCE-GNN, FGNN, CM-HGNN

```bash
cd ~/test_all

LOG="Log/SCL-GCE-GNN/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup python3 nhom3/SCLRS/GCE-GNN/main.py --dataset retailrocket --epoch 20 \
  > "$LOG" 2>&1 &

LOG="Log/FGNN/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup python3 papers_only/FGNN/main.py --dataset retailrocket --epoch 30 \
  > "$LOG" 2>&1 &

LOG="Log/CM-HGNN/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
CUDA_VISIBLE_DEVICES="" nohup python3 papers_only/CM-HGNN/main.py \
  --dataset retailrocket --epoch 20 --batch_size 32 \
  > "$LOG" 2>&1 &

wait
echo "[Khối 4] xong — kiểm tra Log/SCL-GCE-GNN, Log/FGNN, Log/CM-HGNN"
```

### Khối 5 — CCT-GNN, HGCAN

```bash
cd ~/test_all

LOG="Log/CCT-GNN/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup python3 papers_only/CCT-GNN/main.py --dataset retailrocket --epoch 20 \
  > "$LOG" 2>&1 &

LOG="Log/HGCAN/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup python3 papers_only/HGCAN/main.py --dataset retailrocket --epochs 20 \
  > "$LOG" 2>&1 &

wait
echo "[Khối 5] xong — kiểm tra Log/CCT-GNN, Log/HGCAN"
```

---

## 5. Phiên bản smoke (1 epoch) — cùng 5 khối

Thay epoch/config như sau trong từng lệnh:

| Dự án | Ghi chú smoke |
|-------|----------------|
| SR-GNN, GCE-GNN, DHCN, COTREC, SCL-* | `--epoch 1` |
| CSGNN | `--epoch 1 --embSize 100 --beta 0.005` |
| DuoRec | `NCS_SMOKE=1` + `--config_files "seq.yaml DuoRec.yaml config/smoke_1epoch.yaml"` |
| CORE | `NCS_SMOKE=1 CUDA_VISIBLE_DEVICES=""` |
| FGNN, CCT-GNN | `--epoch 1` |
| CM-HGNN | `--epoch 1 --batch_size 32` + `CUDA_VISIBLE_DEVICES=""` |
| HGCAN | `--epochs 1` |

Ví dụ **Khối 2 smoke** (DuoRec):

```bash
cd ~/test_all
export NCS_SMOKE=1 NCS_SMOKE_SAMPLES=2000

LOG="Log/DuoRec/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup python3 nhom3/DuoRec/run_seq.py --dataset retailrocket --model DuoRec \
  --config_files "seq.yaml DuoRec.yaml config/smoke_1epoch.yaml" \
  > "$LOG" 2>&1 &
```

Hoặc chạy **tuần tự 14 job** (không song song) bằng script có sẵn:

```bash
nohup python3 scripts/run_all_retailrocket_smoke.py > /dev/null 2>&1 &
tail -f Log/_smoke_all/retailrocket/log-*.log   # file mới nhất ở dưới cùng
```

---

## 6. Theo dõi & xử lý sự cố

### Xem job đang chạy

```bash
ps aux | grep -E "main.py|run_seq.py" | grep -v grep
jobs -l    # trong cùng shell đã chạy nohup
```

### Xem log realtime

```bash
# Một dự án (thay tên file log mới nhất)
tail -f Log/DuoRec/retailrocket/log-2026-06-12-03-46-09.log

# Tất cả log hôm nay (LogMins)
grep -h '' LogMins/*/retailrocket/$(date +%d-%m-%Y).log
```

### Dừng một job

```bash
kill <PID>           # nhẹ
kill -9 <PID>        # ép dừng nếu treo
```

### Kiểm tra đã xong / lỗi

```bash
tail -3 Log/DuoRec/retailrocket/log-*.log    # dòng END ... OK hoặc traceback
grep -l "Traceback" Log/*/retailrocket/log-$(date +%Y-%m-%d)*.log
```

---

## 7. Bảng tra nhanh

| # | Dự án | Thư mục chạy | Lệnh chính |
|---|--------|--------------|------------|
| 1 | SR-GNN | `nhom1/SR-GNN/pytorch_code` | `python3 main.py --dataset retailrocket --epoch 30` |
| 2 | GCE-GNN | `nhom1/GCE-GNN` | `python3 main.py --dataset retailrocket --epoch 20` |
| 3 | DHCN | `nhom2/DHCN` | `python3 main.py --dataset retailrocket --epoch 30` |
| 4 | COTREC | `nhom2/COTREC` | `python3 main.py --dataset retailrocket --epoch 30` |
| 5 | CSGNN | `nhom2/CSGNN` | `python3 main.py --dataset retailrocket --epoch 1 --embSize 100 --beta 0.005` |
| 6 | DuoRec | `nhom3/DuoRec` | `python3 run_seq.py --dataset retailrocket --model DuoRec --config_files seq.yaml DuoRec.yaml` |
| 7 | CORE | `nhom3/CORE` | `python3 main.py --model trm --dataset retailrocket` |
| 8 | SCL-DHCN | `nhom3/SCLRS/DHCN` | `python3 main.py --dataset retailrocket --epoch 30` |
| 9 | SCL-COTREC | `nhom3/SCLRS/COTREC` | `python3 main.py --dataset retailrocket --epoch 30` |
| 10 | SCL-GCE-GNN | `nhom3/SCLRS/GCE-GNN` | `python3 main.py --dataset retailrocket --epoch 20` |
| 11 | FGNN | `papers_only/FGNN` | `python3 main.py --dataset retailrocket --epoch 30` |
| 12 | CM-HGNN | `papers_only/CM-HGNN` | `python3 main.py --dataset retailrocket --epoch 20 --batch_size 32` |
| 13 | CCT-GNN | `papers_only/CCT-GNN` | `python3 main.py --dataset retailrocket --epoch 20` |
| 14 | HGCAN | `papers_only/HGCAN` | `python3 main.py --dataset retailrocket --epochs 20` |

Data mỗi dự án: `Data/<TênDựÁn>/retailrocket/` (xem chi tiết convert trong `chuyendoi.txt`).

---

## 8. Quy trình gợi ý hàng ngày

1. `preprocess` (nếu data đổi) → symlink DuoRec/CORE
2. Chọn **full** hoặc **smoke**
3. Chạy **Khối 1** → `wait` → kiểm tra log
4. Lặp Khối 2 … 5
5. Cuối ngày: đọc `LogMins/*/retailrocket/<ngày>.log` để tổng hợp OK/FAIL

Nếu muốn chỉ chạy **một** dự án ngầm:

```bash
cd ~/test_all
LOG="Log/DHCN/retailrocket/log-$(date +%Y-%m-%d-%H-%M-%S).log"
mkdir -p "$(dirname "$LOG")"
nohup python3 nhom2/DHCN/main.py --dataset retailrocket --epoch 30 > "$LOG" 2>&1 &
tail -f "$LOG"
```
