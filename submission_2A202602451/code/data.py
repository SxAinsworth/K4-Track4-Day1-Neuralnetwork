"""data.py — nạp dữ liệu đã chia, tách validation, chuẩn hoá, đưa lên thiết bị, chia lô.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    tr = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")
    X_train_full, y_train_full = tr["X"], tr["y"]
    X_eval, y_eval, eval_row_id = ev["X"], ev["y"], ev["row_id"]
    for X, y in ((X_train_full, y_train_full), (X_eval, y_eval)):
        assert X.ndim == 2 and X.shape[1] == 54 and X.dtype == np.float32, (X.shape, X.dtype)
        assert y.shape == (len(X),) and y.dtype == np.int64, (y.shape, y.dtype)
        assert y.min() >= 0 and y.max() <= 6, "nhãn phải là 0..6"
    assert len(eval_row_id) == len(X_eval)
    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed)
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    Trả lời: mean/std là "tham số" của pipeline; tính trên val/eval là để thông tin của tập đánh giá
    rò rỉ vào mô hình, khiến điểm val/eval lạc quan hơn thực tế khi gặp dữ liệu mới.
    """
    num = X_tr[:, :N_NUMERIC].astype(np.float64)   # float64 để tổng của ~370k mẫu không mất chính xác
    mean = num.mean(axis=0)
    std = num.std(axis=0)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    Xs = X.copy()
    safe_std = np.where(std > 0, std, 1.0).astype(np.float32)   # cột hằng số: chỉ trừ mean
    Xs[:, :N_NUMERIC] = (Xs[:, :N_NUMERIC] - mean) / safe_std
    return Xs


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_full, y_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_full, y_full, val_fraction, seed)
    mean, std = fit_standardizer(X_tr)                # CHỈ từ X_tr: val/eval không tham gia
    X_tr, X_val, X_eval = (apply_standardizer(X, mean, std) for X in (X_tr, X_val, X_eval))

    majority = int(np.bincount(y_tr, minlength=7).argmax())
    print(f"train {X_tr.shape} | val {X_val.shape} | eval {X_eval.shape}")
    print(f"luôn đoán lớp đa số ({majority}): val acc = {(y_val == majority).mean():.4f}")

    def to_dev(a, dtype):
        return torch.tensor(a, dtype=dtype, device=device)

    return {
        "X_tr": to_dev(X_tr, torch.float32), "y_tr": to_dev(y_tr, torch.int64),
        "X_val": to_dev(X_val, torch.float32), "y_val": to_dev(y_val, torch.int64),
        "X_eval": to_dev(X_eval, torch.float32), "y_eval": to_dev(y_eval, torch.int64),
        "eval_row_id": eval_row_id,
        "mean": mean, "std": std, "majority_class": majority,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    Quyết định: GIỮ batch cuối nhỏ hơn (không bỏ), để mỗi epoch dùng đủ mọi mẫu train. Với 371 847 mẫu
    và batch 512 thì batch cuối có 135 mẫu, chỉ 1/727 số bước nên ảnh hưởng tới nhiễu gradient không đáng kể.
    """
    N = len(X)
    if shuffle:
        perm = torch.randperm(N, generator=generator, device=generator.device if generator is not None else "cpu")
        perm = perm.to(X.device)
    else:
        perm = torch.arange(N, device=X.device)
    for i in range(0, N, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
