#!/usr/bin/env python3
"""
ModelNet40 + MinkowskiFCNN (minkfcnn), как в Seminar_12 / classification_modelnet40.py.
Логи: stdout + файл LOG_DIR/training.log
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import sklearn.metrics as metrics  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

import MinkowskiEngine as ME  # noqa: E402

# Оригинальный Stanford часто недоступен; см. https://huggingface.co/datasets/Msun/modelnet40
MODELNET40_ZIP_MIRRORS = [
    "https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip",
    "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
]


def setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "training.log")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    logging.info("Logging to %s", path)


def seed_all(random_seed: int) -> None:
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def minkowski_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"] for d in list_data],
        dtype=torch.float32,
    )
    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
        "labels": labels_batch,
    }


def _download_zip_to_path(url: str, dest_zip: str) -> None:
    if shutil.which("wget"):
        subprocess.run(
            [
                "wget",
                "-O",
                dest_zip,
                "--tries=3",
                "--timeout=120",
                "--no-check-certificate",
                url,
            ],
            check=True,
        )
    elif shutil.which("curl"):
        subprocess.run(
            ["curl", "-fL", "--retry", "3", "--connect-timeout", "30", "-o", dest_zip, url],
            check=True,
        )
    else:
        raise RuntimeError("Нужен wget или curl для скачивания ModelNet40")


def download_modelnet40_dataset(data_root: str, zip_urls: list[str] | None = None) -> None:
    """Скачивает и распаковывает датасет, если в data_root ещё нет *.h5 (в т.ч. пустой bind-mount)."""
    parent = os.path.dirname(os.path.abspath(data_root)) or "."
    os.makedirs(parent, exist_ok=True)
    zip_name = os.path.join(parent, "modelnet40_ply_hdf5_2048.zip")
    h5_ok = os.path.isdir(data_root) and len(glob.glob(os.path.join(data_root, "ply_data_*.h5"))) > 0
    if h5_ok:
        return
    env_url = (os.environ.get("MODELNET40_URL") or "").strip()
    if zip_urls is not None:
        urls = list(zip_urls)
    elif env_url:
        urls = [env_url] + [u for u in MODELNET40_ZIP_MIRRORS if u != env_url]
    else:
        urls = list(MODELNET40_ZIP_MIRRORS)
    if not os.path.exists(zip_name) or os.path.getsize(zip_name) < 1_000_000:
        if os.path.exists(zip_name):
            os.remove(zip_name)
        logging.info("Downloading ModelNet40 zip (пробуем зеркала)...")
        last: Exception | None = None
        for url in urls:
            try:
                logging.info("URL: %s", url)
                _download_zip_to_path(url, zip_name)
                if os.path.getsize(zip_name) > 1_000_000:
                    break
                os.remove(zip_name)
            except (OSError, subprocess.CalledProcessError) as e:
                last = e
                if os.path.exists(zip_name):
                    try:
                        os.remove(zip_name)
                    except OSError:
                        pass
                logging.warning("Не удалось: %s", e)
        else:
            raise RuntimeError(
                "Не удалось скачать modelnet40_ply_hdf5_2048.zip ни с одного URL. "
                "Задайте MODELNET40_URL или --modelnet-url, либо положите zip рядом с data_root. "
                f"Последняя ошибка: {last}"
            ) from (last if last else None)
    logging.info("Extracting ModelNet40 into %s", parent)
    subprocess.run(["unzip", "-o", zip_name, "-d", parent], check=True)


class ModelNet40H5(Dataset):
    def __init__(
        self,
        phase: str,
        data_root: str = "modelnet40_ply_hdf5_2048",
        transform=None,
        num_points: int = 2048,
        modelnet_zip_urls: list[str] | None = None,
    ):
        super().__init__()
        download_modelnet40_dataset(data_root, zip_urls=modelnet_zip_urls)
        phase = "test" if phase in ("val", "test") else "train"
        self.data, self.label = self._load_data(data_root, phase)
        self.transform = transform
        self.phase = phase
        self.num_points = num_points

    def _load_data(self, data_root, phase):
        data, labels = [], []
        assert os.path.exists(data_root), f"{data_root} does not exist"
        files = glob.glob(os.path.join(data_root, f"ply_data_{phase}*.h5"))
        assert len(files) > 0, "No h5 files found"
        for h5_name in files:
            with h5py.File(h5_name) as f:
                data.extend(f["data"][:].astype("float32"))
                labels.extend(f["label"][:].astype("int64"))
        return np.stack(data, axis=0), np.stack(labels, axis=0)

    def __getitem__(self, i: int) -> dict:
        xyz = self.data[i]
        if self.phase == "train":
            np.random.shuffle(xyz)
        if len(xyz) > self.num_points:
            xyz = xyz[: self.num_points]
        if self.transform is not None:
            xyz = self.transform(xyz)
        label = self.label[i]
        xyz = torch.from_numpy(xyz).to(torch.float32)
        label = torch.from_numpy(label)
        return {"coordinates": xyz, "features": xyz, "label": label}

    def __len__(self):
        return self.data.shape[0]


class CoordinateTransformation:
    def __init__(self, scale_range=(0.9, 1.1), trans=0.25, jitter=0.025, clip=0.05):
        self.scale_range = scale_range
        self.trans = trans
        self.jitter = jitter
        self.clip = clip

    def __call__(self, coords):
        if random.random() < 0.9:
            coords = coords * np.random.uniform(
                low=self.scale_range[0], high=self.scale_range[1], size=[1, 3]
            )
        if random.random() < 0.9:
            coords = coords + np.random.uniform(low=-self.trans, high=self.trans, size=[1, 3])
        if random.random() < 0.7:
            coords = coords + np.clip(
                self.jitter * (np.random.rand(len(coords), 3) - 0.5),
                -self.clip,
                self.clip,
            )
        return coords


class CoordinateTranslation:
    def __init__(self, translation: float):
        self.trans = translation

    def __call__(self, coords):
        if self.trans > 0:
            coords = coords + np.random.uniform(low=-self.trans, high=self.trans, size=[1, 3])
        return coords


def make_data_loader(phase, config):
    assert phase in ("train", "val", "test")
    is_train = phase == "train"
    dataset = ModelNet40H5(
        phase=phase,
        data_root=config.data_root,
        transform=CoordinateTransformation(trans=config.translation)
        if is_train
        else CoordinateTranslation(config.test_translation),
        modelnet_zip_urls=getattr(config, "modelnet_zip_urls", None),
    )
    return DataLoader(
        dataset,
        num_workers=config.num_workers,
        shuffle=is_train,
        collate_fn=minkowski_collate_fn,
        batch_size=config.batch_size,
    )


def create_input_batch(batch, device, quantization_size):
    batch["coordinates"][:, 1:] = batch["coordinates"][:, 1:] / quantization_size
    return ME.TensorField(
        coordinates=batch["coordinates"],
        features=batch["features"],
        device=device,
    )


def criterion(pred, labels, smoothing=True):
    labels = labels.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, labels.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        return -(one_hot * log_prb).sum(dim=1).mean()
    return F.cross_entropy(pred, labels, reduction="mean")


class MinkowskiFCNN(ME.MinkowskiNetwork):
    def __init__(
        self,
        in_channel,
        out_channel,
        embedding_channel=1024,
        channels=(32, 48, 64, 96, 128),
        D=3,
    ):
        ME.MinkowskiNetwork.__init__(self, D)
        self.network_initialization(
            in_channel,
            out_channel,
            channels=channels,
            embedding_channel=embedding_channel,
            kernel_size=3,
            D=D,
        )
        self.weight_initialization()

    def get_mlp_block(self, in_channel, out_channel):
        return nn.Sequential(
            ME.MinkowskiLinear(in_channel, out_channel, bias=False),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def get_conv_block(self, in_channel, out_channel, kernel_size, stride):
        return nn.Sequential(
            ME.MinkowskiConvolution(
                in_channel,
                out_channel,
                kernel_size=kernel_size,
                stride=stride,
                dimension=self.D,
            ),
            ME.MinkowskiBatchNorm(out_channel),
            ME.MinkowskiLeakyReLU(),
        )

    def network_initialization(
        self,
        in_channel,
        out_channel,
        channels,
        embedding_channel,
        kernel_size,
        D=3,
    ):
        self.mlp1 = self.get_mlp_block(in_channel, channels[0])
        self.conv1 = self.get_conv_block(channels[0], channels[1], kernel_size=kernel_size, stride=1)
        self.conv2 = self.get_conv_block(channels[1], channels[2], kernel_size=kernel_size, stride=2)
        self.conv3 = self.get_conv_block(channels[2], channels[3], kernel_size=kernel_size, stride=2)
        self.conv4 = self.get_conv_block(channels[3], channels[4], kernel_size=kernel_size, stride=2)
        self.conv5 = nn.Sequential(
            self.get_conv_block(
                channels[1] + channels[2] + channels[3] + channels[4],
                embedding_channel // 4,
                kernel_size=3,
                stride=2,
            ),
            self.get_conv_block(
                embedding_channel // 4,
                embedding_channel // 2,
                kernel_size=3,
                stride=2,
            ),
            self.get_conv_block(
                embedding_channel // 2,
                embedding_channel,
                kernel_size=3,
                stride=2,
            ),
        )
        self.pool = ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=D)
        self.global_max_pool = ME.MinkowskiGlobalMaxPooling()
        self.global_avg_pool = ME.MinkowskiGlobalAvgPooling()
        self.final = nn.Sequential(
            self.get_mlp_block(embedding_channel * 2, 512),
            ME.MinkowskiDropout(),
            self.get_mlp_block(512, 512),
            ME.MinkowskiLinear(512, out_channel, bias=True),
        )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x: ME.TensorField):
        x = self.mlp1(x)
        y = x.sparse()
        y = self.conv1(y)
        y1 = self.pool(y)
        y = self.conv2(y1)
        y2 = self.pool(y)
        y = self.conv3(y2)
        y3 = self.pool(y)
        y = self.conv4(y3)
        y4 = self.pool(y)
        x1 = y1.slice(x)
        x2 = y2.slice(x)
        x3 = y3.slice(x)
        x4 = y4.slice(x)
        x = ME.cat(x1, x2, x3, x4)
        y = self.conv5(x.sparse())
        x1 = self.global_max_pool(y)
        x2 = self.global_avg_pool(y)
        return self.final(ME.cat(x1, x2)).F


def evaluate(net, device, config):
    data_loader = make_data_loader("test", config)
    net.eval()
    labels, preds = [], []
    with torch.no_grad():
        for batch in data_loader:
            inp = create_input_batch(
                batch,
                device=device,
                quantization_size=config.voxel_size,
            )
            logit = net(inp)
            pred = torch.argmax(logit, 1)
            labels.append(batch["labels"].cpu().numpy())
            preds.append(pred.cpu().numpy())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return float(metrics.accuracy_score(np.concatenate(labels), np.concatenate(preds)))


def plot_and_save_metrics(history: dict, out_dir: str) -> str:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    if history["train_step"]:
        ax_loss.plot(history["train_step"], history["train_loss"], label="train loss")
    ax_loss.set_xlabel("iteration")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Training loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()
    if history["val_step"]:
        ax_acc.plot(history["val_step"], history["val_acc"], "o-", label="val acc")
    ax_acc.set_xlabel("iteration")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Validation accuracy (test split)")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def train_with_metrics(net, device, config, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    optimizer = optim.SGD(
        net.parameters(),
        lr=config.lr,
        momentum=0.9,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_steps)
    history = {
        "train_step": [],
        "train_loss": [],
        "val_step": [],
        "val_acc": [],
    }
    train_iter = iter(make_data_loader("train", config))
    best_metric = 0.0
    best_step = -1
    net.train()
    t0 = time.perf_counter()
    for i in range(config.max_steps):
        optimizer.zero_grad()
        try:
            data_dict = next(train_iter)
        except StopIteration:
            train_iter = iter(make_data_loader("train", config))
            data_dict = next(train_iter)
        inp = create_input_batch(
            data_dict,
            device=device,
            quantization_size=config.voxel_size,
        )
        logit = net(inp)
        loss = criterion(logit, data_dict["labels"].to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if i % config.stat_freq == 0:
            elapsed = time.perf_counter() - t0
            logging.info("iter=%s loss=%.4e elapsed_s=%.0f", i, loss.item(), elapsed)
            history["train_step"].append(i)
            history["train_loss"].append(float(loss.item()))

        if i % config.val_freq == 0 and i > 0:
            ckpt_path = os.path.join(out_dir, "checkpoint_last.pth")
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "curr_iter": i,
                    "history": dict(history),
                },
                ckpt_path,
            )
            acc = evaluate(net, device, config)
            history["val_step"].append(i)
            history["val_acc"].append(acc)
            logging.info(
                "iter=%s val_acc=%.4f best=%.4f @ %s",
                i,
                acc,
                best_metric,
                best_step,
            )
            if acc > best_metric:
                best_metric = acc
                best_step = i
                torch.save(
                    {
                        "state_dict": net.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "curr_iter": i,
                        "best_val_acc": best_metric,
                    },
                    os.path.join(out_dir, "best_model.pth"),
                )
            net.train()

    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump({**history, "config": dict(vars(config))}, f, indent=2)
    plot_path = plot_and_save_metrics(history, out_dir)
    logging.info("Saved history.json, %s, best_model.pth in %s", plot_path, out_dir)
    return history, best_metric, best_step


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default=os.environ.get("LOG_DIR", "/logs"))
    p.add_argument("--work-dir", default=os.environ.get("WORKDIR", "/workspace"))
    p.add_argument("--data-root", default="modelnet40_ply_hdf5_2048")
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "5000")))
    p.add_argument("--val-freq", type=int, default=int(os.environ.get("VAL_FREQ", "500")))
    p.add_argument("--stat-freq", type=int, default=int(os.environ.get("STAT_FREQ", "50")))
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "32")))
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--modelnet-url",
        default=None,
        help="Переопределить URL zip (иначе HF → Stanford; или env MODELNET40_URL)",
    )
    _skip_test = os.environ.get("SKIP_FINAL_TEST", "").strip().lower() in ("1", "true", "yes")
    p.add_argument(
        "--skip-final-test",
        action="store_true",
        default=_skip_test,
        help="Не гонять полный test set в конце (на CPU долго; или env SKIP_FINAL_TEST=1)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.chdir(args.work_dir)
    setup_logging(args.log_dir)
    out_dir = os.path.join(args.work_dir, "seminar_12_runs")
    logging.info("WORKDIR=%s LOG_DIR=%s OUT=%s", args.work_dir, args.log_dir, out_dir)
    logging.info("torch %s cuda=%s ME.cuda=%s", torch.__version__, torch.cuda.is_available(), ME.is_cuda_available())

    config = SimpleNamespace(
        voxel_size=0.05,
        max_steps=args.max_steps,
        val_freq=args.val_freq,
        stat_freq=args.stat_freq,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=1e-4,
        num_workers=args.num_workers,
        translation=0.2,
        test_translation=0.0,
        seed=args.seed,
        data_root=os.path.join(args.work_dir, args.data_root),
        modelnet_zip_urls=([args.modelnet_url] if args.modelnet_url else None),
    )
    seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MinkowskiFCNN(in_channel=3, out_channel=40, embedding_channel=1024).to(device)
    logging.info("Start training: max_steps=%s", config.max_steps)
    history, best_acc, best_step = train_with_metrics(net, device, config, out_dir)
    if args.skip_final_test:
        test_acc = None
        logging.info("Skipped full test evaluate (--skip-final-test / SKIP_FINAL_TEST=1); best_val=%.4f @ step %s", best_acc, best_step)
    else:
        test_acc = evaluate(net, device, config)
        logging.info("Final test_acc=%.4f best_val=%.4f @ step %s", test_acc, best_acc, best_step)
    with open(os.path.join(args.log_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_acc": test_acc,
                "best_val_acc": best_acc,
                "best_step": best_step,
                "out_dir": out_dir,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
