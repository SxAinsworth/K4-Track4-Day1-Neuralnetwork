"""plots.py — ảnh cho từng thí nghiệm và ảnh chồng so sánh theo nhóm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt

METRIC_LABELS = {
    "train_loss": "train loss (eval mode)", "val_loss": "val loss", "val_acc": "val accuracy",
    "val_macro_f1": "val macro-F1", "grad_norm": "‖g‖ trung bình (trước clip)",
    "grad_norm_max": "‖g‖ lớn nhất (trước clip)", "epoch_time_s": "thời gian / epoch (s)",
}


def cfg_label(cfg: dict) -> str:
    """Tóm tắt cấu hình trên một dòng để đặt vào tiêu đề ảnh."""
    hidden = "-".join(str(h) for h in cfg["hidden"])
    parts = [f"{cfg['optimizer']} lr={cfg['lr']:g}", f"loss={cfg['loss']}", f"batch={cfg['batch']}",
             f"hidden={hidden}", f"init={cfg['init']}", f"dropout={cfg['dropout']:g}",
             f"clip={cfg['clip_norm']}", cfg["precision"], f"wd={cfg['weight_decay']:g}", f"seed={cfg['seed']}"]
    if cfg.get("scheduler"):
        parts.append(f"sched={cfg['scheduler']}")
    return " | ".join(parts)


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip): trung bình và lớn nhất trong epoch, kèm ngưỡng clip nếu có
    Tiêu đề ghi exp_id và cấu hình chính; đường thẳng đứng đánh dấu best_epoch (val loss thấp nhất).
    """
    cfg, h, s = result["cfg"], result["history"], result["summary"]
    ep = h["epoch"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    ax = axes[0]
    ax.plot(ep, h["train_loss"], "o-", ms=3, label="train (eval mode, 50k mẫu)")
    ax.plot(ep, h["val_loss"], "o-", ms=3, label="val")
    ax.set_title(f"Loss ({cfg['loss'].upper()})")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")

    ax = axes[1]
    ax.plot(ep, h["val_acc"], "o-", ms=3, c="C2", label="val accuracy")
    ax.plot(ep, h["val_macro_f1"], "s-", ms=3, c="C3", label="val macro-F1")
    ax.axhline(0.4876, ls=":", c="gray", lw=1, label="đoán đa số (acc 0,4876)")
    ax.set_title("Val accuracy và macro-F1")
    ax.set_xlabel("epoch"); ax.set_ylabel("điểm")

    ax = axes[2]
    ax.plot(ep, h["grad_norm"], "o-", ms=3, c="C4", label="‖g‖ trung bình")
    ax.plot(ep, h["grad_norm_max"], "^--", ms=3, c="C4", alpha=0.5, label="‖g‖ lớn nhất")
    if cfg["clip_norm"] is not None:
        ax.axhline(cfg["clip_norm"], ls="--", c="red", lw=1, label=f"clip c = {cfg['clip_norm']:g}")
    ax.set_yscale("log")
    ax.set_title("Chuẩn gradient toàn cục (trước clip)")
    ax.set_xlabel("epoch"); ax.set_ylabel("‖g‖₂ (thang log)")

    for ax in axes:
        if s["best_epoch"] is not None:
            ax.axvline(s["best_epoch"], c="k", ls="--", lw=0.8, alpha=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    best = (f"best epoch {s['best_epoch']}: val loss {s['best_val_loss']:.4f}, acc {s['val_acc']:.4f}, "
            f"macro-F1 {s['val_macro_f1']:.4f}" if s["best_epoch"] is not None else "không có epoch hợp lệ")
    status = "  [DIVERGED]" if s["diverged"] else ""
    fig.suptitle(f"{cfg['exp_id']}{status}  —  {best}\n{cfg_label(cfg)}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric, path: str, title: str = "") -> None:
    """Vẽ chồng một hoặc nhiều chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm,
    mỗi chỉ số một ô, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    metric: tên một chỉ số hoặc danh sách tên. Dùng cho ảnh figures/compare_<nhóm>.png.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.2), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for i, r in enumerate(results):
            ax.plot(r["history"]["epoch"], r["history"][m], "o-", ms=2.5, c=f"C{i % 10}",
                    label=r["cfg"]["exp_id"] + ("  [diverged]" if r["summary"]["diverged"] else ""))
        if m.startswith("grad_norm"):
            ax.set_yscale("log")
        ax.set_title(METRIC_LABELS.get(m, m))
        ax.set_xlabel("epoch"); ax.set_ylabel(m)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
