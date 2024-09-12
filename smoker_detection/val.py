import argparse

import timm
import torch
from dataset import BASIC_TRANSFORMS, ImageDataset
from opts import opts
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm
from utils import seed_everything


def main(opt: argparse.Namespace):
    seed_everything(opt.seed)  # 設置隨機種子
    val_dataset = ImageDataset(
        root="data/Validation",
        transforms=BASIC_TRANSFORMS(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
    )
    model = timm.create_model(opt.model_name, pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(opt.load_model_path))
    model = model.to(opt.device)

    acc_metric = BinaryAccuracy(threshold=0.5).to(opt.device)  # 準確率計算器

    model.eval()  # 切換到評估模式
    acc_metric.reset()  # 重置準確率計算器
    with torch.no_grad():  # 關閉梯度計算
        pbar = tqdm(val_loader, desc=f"[{opt.device}]")  # 進度條
        for source, targets in pbar:
            source, targets = source.to(opt.device), targets.to(opt.device)
            output = model(source).squeeze()
            acc_metric.update(output, targets)
            pbar.set_description(f"Accuracy {100 * acc_metric.compute().item():.2f}%")


if __name__ == "__main__":
    """必須啟動test模式並指定模型權重路徑"""
    opt = opts().parse(
        [
            "--test",
            "--load_model_path",
            "runs/epoch30_seed42/2024-03-25-15-08-37/best_model.pt",
        ]
    )
    main(opt)
