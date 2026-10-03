"""train.py — đặt seed, đánh giá, vòng huấn luyện dùng chung, dự đoán và ghi file nộp.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, build_scheduler, clip_gradients

NUM_CLASSES = 7
TRAIN_EVAL_SUBSET = 50_000   # train loss đo trên 50 000 mẫu train CỐ ĐỊNH (cùng tập con cho mọi thí nghiệm)

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # chọn bằng val trong notebook (Part 2), không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
    scheduler=None,            # None | "cosine" (ghi vào notes nếu dùng)
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    pred_pos, true_pos = cm.sum(axis=0), cm.sum(axis=1)
    p = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    r = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return float(f1.mean())


def confusion_matrix(y, pred, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Ma trận nhầm lẫn (C, C) tính trên device bằng bincount: hàng = nhãn thật, cột = dự đoán."""
    idx = y.long() * num_classes + pred.long()
    return torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes).cpu().numpy()


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    return torch.cat([model(X[i:i + batch_size]).argmax(dim=1) for i in range(0, len(X), batch_size)])


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    Luôn tính ở FP32 (không autocast), kể cả với thí nghiệm mixed precision, để metric so sánh được.
    """
    model.eval()
    total_loss, preds = 0.0, []
    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size]).float()
        total_loss += compute_loss(logits, y[i:i + batch_size], loss_name, reduction="sum").item()
        preds.append(logits.argmax(dim=1))
    pred = torch.cat(preds)
    return {"loss": total_loss / len(X),
            "acc": float((pred == y).float().mean().item()),
            "macro_f1": macro_f1_from_confusion(confusion_matrix(y, pred))}


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).

    MSE: giống nn.MSELoss mặc định — KHÔNG có hệ số 1/2, lấy trung bình trên mọi phần tử của ma trận (B, 7),
    tức chia cho B·7. Với reduction="sum" (dùng khi đánh giá theo lô) chia thêm 7 để khi chia N ở evaluate
    ta được đúng cùng thang đo với lúc huấn luyện.
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        target = F.one_hot(y, NUM_CLASSES).to(logits.dtype)
        if reduction == "sum":
            return F.mse_loss(logits, target, reduction="sum") / NUM_CLASSES
        return F.mse_loss(logits, target, reduction=reduction)
    raise ValueError(f"loss không hợp lệ: {loss_name!r}")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG); khoá thiếu lấy từ DEFAULT_CFG
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1", "grad_norm",
                     "grad_norm_max", "clip_frac", "epoch_time_s", "lr"},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged", ...},
         "best_state": state_dict (trên CPU) của epoch có val_loss thấp nhất}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Cách đo:
      - train_loss: ở eval() trên TRAIN_EVAL_SUBSET mẫu train cố định (chọn bằng seed 0, giống nhau mọi thí nghiệm),
        để so được với val_loss kể cả khi có dropout.
      - grad_norm: chuẩn L2 toàn cục TRƯỚC khi clip, trung bình các bước trong epoch; grad_norm_max là bước lớn
        nhất trong epoch (thấy "gai"); clip_frac là tỉ lệ bước có ‖g‖ > clip_norm (clipping thực sự kích hoạt).
      - epoch_time_s: chỉ thời gian vòng cập nhật (không gồm phần đánh giá), synchronize trước/sau nếu GPU.
      - peak_mem_MB: torch.cuda.max_memory_allocated của lần chạy, GỒM dữ liệu đã nằm sẵn trên GPU;
        peak_mem_train_MB: phần tăng thêm so với lúc trước khi tạo model (model + optimizer + kích hoạt).
      - val_acc / val_macro_f1 trong summary lấy tại best_epoch (val_loss thấp nhất) — tương đương dừng sớm.
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    if cfg["lr"] is None:
        raise ValueError("cfg['lr'] chưa được đặt; hãy chọn bằng val")
    if cfg["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"precision không hợp lệ: {cfg['precision']!r}")
    X_tr, y_tr, X_val, y_val = data["X_tr"], data["y_tr"], data["X_val"], data["y_val"]
    device = X_tr.device
    use_cuda = device.type == "cuda"
    hidden = tuple(cfg["hidden"])

    # tập con train cố định để đo train loss (độc lập với seed huấn luyện)
    sub = torch.randperm(len(X_tr), generator=torch.Generator().manual_seed(0))[:TRAIN_EVAL_SUBSET].to(device)
    X_sub, y_sub = X_tr[sub], y_tr[sub]

    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_before = torch.cuda.memory_allocated()

    # 0. model, optimizer, (scheduler), scaler
    set_seed(cfg["seed"])
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"])
    assert count_params(model) == EXPECTED_PARAMS[hidden], (hidden, count_params(model))
    model = model.to(device)
    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    steps_per_epoch = math.ceil(len(X_tr) / cfg["batch"])
    scheduler = build_scheduler(optimizer, cfg.get("scheduler"), total_steps=steps_per_epoch * cfg["epochs"])
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(cfg["precision"])
    scaler = torch.amp.GradScaler(device.type) if cfg["precision"] == "fp16" else None
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])   # thứ tự xáo lô phụ thuộc seed

    # 1. loss bước 0, trước mọi cập nhật
    step0_loss = evaluate(model, X_val, y_val, cfg["loss"])["loss"]

    hist = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1",
                            "grad_norm", "grad_norm_max", "clip_frac", "epoch_time_s", "lr")}
    best_val_loss, best_epoch, best_state = float("inf"), None, None
    diverged, skipped_steps = False, 0

    # 2. vòng huấn luyện
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        norms = []
        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator=gen):
            # autocast chỉ bọc forward + loss; loss tính trên logit ép về FP32
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(xb)
            loss = compute_loss(logits.float(), yb, cfg["loss"])
            if not torch.isfinite(loss):
                diverged = True                     # dừng ngay, không để notebook chạy tiếp với NaN
                break
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)          # về thang thật TRƯỚC khi đo chuẩn / clip
            else:
                loss.backward()
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])
            if scaler is not None:
                scaler.step(optimizer)              # tự bỏ qua bước nếu gradient có inf/NaN (tràn FP16)
                scaler.update()
                if not math.isfinite(gn):
                    skipped_steps += 1
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if math.isfinite(gn):
                norms.append(gn)
        if use_cuda:
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - t0

        # cuối epoch: đo ở eval() trên tập con train cố định và val
        tr = evaluate(model, X_sub, y_sub, cfg["loss"])
        va = evaluate(model, X_val, y_val, cfg["loss"])
        norms_arr = np.array(norms) if norms else np.array([np.nan])
        clip = cfg["clip_norm"]
        hist["epoch"].append(epoch)
        hist["train_loss"].append(tr["loss"])
        hist["val_loss"].append(va["loss"])
        hist["val_acc"].append(va["acc"])
        hist["val_macro_f1"].append(va["macro_f1"])
        hist["grad_norm"].append(float(np.nanmean(norms_arr)))
        hist["grad_norm_max"].append(float(np.nanmax(norms_arr)))
        hist["clip_frac"].append(float((norms_arr > clip).mean()) if clip is not None and norms else 0.0)
        hist["epoch_time_s"].append(epoch_time)
        hist["lr"].append(optimizer.param_groups[0]["lr"])

        if not (math.isfinite(tr["loss"]) and math.isfinite(va["loss"])):
            diverged = True
        if va["loss"] < best_val_loss:              # NaN không bao giờ "tốt hơn"
            best_val_loss, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if diverged:
            break

    # 3. tóm tắt tại best_epoch (chọn bằng val loss)
    b = best_epoch - 1 if best_epoch is not None else None
    ran = len(hist["epoch"]) > 0
    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss if b is not None else None,
        "best_epoch": best_epoch,
        "final_train_loss": hist["train_loss"][-1] if ran else None,
        "final_val_loss": hist["val_loss"][-1] if ran else None,
        "val_acc": hist["val_acc"][b] if b is not None else None,
        "val_macro_f1": hist["val_macro_f1"][b] if b is not None else None,
        "time_per_epoch_s": float(np.mean(hist["epoch_time_s"])) if ran else None,
        "peak_mem_MB": torch.cuda.max_memory_allocated() / 2**20 if use_cuda else None,
        "peak_mem_train_MB": (torch.cuda.max_memory_allocated() - mem_before) / 2**20 if use_cuda else None,
        "diverged": diverged,
        "epochs_run": len(hist["epoch"]),
        "skipped_steps": skipped_steps,
        "device": torch.cuda.get_device_name(0) if use_cuda else str(device),
    }
    return {"cfg": {**cfg, "hidden": list(hidden)}, "history": hist, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    import pandas as pd
    row_id, preds = np.asarray(row_id), np.asarray(preds)
    assert len(row_id) == len(preds) and len(np.unique(row_id)) == len(row_id)
    assert preds.min() >= 0 and preds.max() <= NUM_CLASSES - 1
    pd.DataFrame({"row_id": row_id.astype(np.int64), "pred": preds.astype(np.int64)}).to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` (làm trong notebook) và ghi kết quả vào bảng/báo cáo
    """
    c = result["cfg"]
    model = MLP(hidden=tuple(c["hidden"]), dropout=c["dropout"], init=c["init"])
    model.load_state_dict(result["best_state"])
    model = model.to(data["X_eval"].device)
    preds = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print(f"đã ghi {len(preds)} dự đoán của {c['exp_id']} (best_epoch = {result['summary']['best_epoch']}) -> {pred_path}")
