import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from logger import Logger
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        opt: argparse.Namespace,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: Logger,
    ):
        """初始化訓練器"""
        self.epochs_run = 0
        # 解析 opt 的必要參數
        self.lr = opt.lr
        self.device = opt.device
        self.eta_min = opt.eta_min
        self.max_epochs = opt.max_epochs
        self.log_dir = opt.log_dir

        self.model = model.to(self.device)  # 模型
        self.train_loader = train_loader  # 訓練資料
        self.val_loader = val_loader  # 驗證資料
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)  # 損失函數
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)  # 優化器
        # 學習率調整器
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.max_epochs, eta_min=self.eta_min, last_epoch=-1
        )
        # 訓練 & 驗證準確率計算器
        self.train_acc_metric = BinaryAccuracy(threshold=0.5).to(self.device)
        self.val_acc_metric = BinaryAccuracy(threshold=0.5).to(self.device)

        self.logger = logger  # 記錄器
        self.prof = logger.prof  # 設備瓶頸分析器

        if opt.resume:  # 是否從快照中恢復訓練
            print("Loading snapshot")
            # opt.snapshot_path = "runs/.../snapshot.pt"
            self._load_snapshot(opt.snapshot_path)

    def _load_snapshot(self, path: str):
        """
        讀取快照並恢復訓練器的狀態。

        Args:
            path (str): The path to the snapshot file.
        """
        loc = self.device
        snapshot = torch.load(path, map_location=loc)  # 載入快照
        self.epochs_run = snapshot["EPOCHS_RUN"]  # 載入已訓練週期
        self.model.load_state_dict(snapshot["MODEL_STATE"])  # 載入模型
        self.optimizer.load_state_dict(snapshot["OPTIMIZER"])  # 載入優化器狀態
        # 載入學習率調整器狀態
        self.lr_scheduler.load_state_dict(snapshot["LR_SCHEDULER"])
        print(
            f"[GPU{self.device}] Resuming training from snapshot at Epoch {snapshot['EPOCHS_RUN']}"
        )

    def _save_snapshot(self, epoch: int):
        """
        存儲模型快照

        Args:
            epoch (int): The current epoch number.
        """
        snapshot = dict(
            EPOCHS_RUN=epoch,  # 保存已訓練週期
            MODEL_STATE=self.model.state_dict(),  # 保存模型
            OPTIMIZER=self.optimizer.state_dict(),  # 保存優化器狀態
            LR_SCHEDULER=self.lr_scheduler.state_dict(),  # 保存學習率調整器狀態
        )
        model_path = os.path.join(self.log_dir, f"snapshot.pt")
        torch.save(snapshot, model_path)

    def train_one_epoch(self, epoch: int):
        """
        訓練一個epoch

        Args:
            epoch (int): 當前epoch數

        """
        self.model.train()  # 設置模型為訓練模式
        self.train_acc_metric.reset()  # 重置訓練準確率計算器
        loss = 0.0  # 紀錄損失
        # 設置進度條
        pbar = tqdm(self.train_loader, desc=f"[{self.device}] Train Epoch {epoch:2d}")
        # Pytorch 標準訓練迴圈
        for source, targets in pbar:
            source, targets = source.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(source).squeeze()
            loss_batch = self.criterion(output, targets.float())
            loss_batch.backward()
            self.optimizer.step()

            self.train_acc_metric.update(output, targets)  # 更新訓練準確率
            loss += loss_batch.item()  # 累加損失
            # 顯示當前損失和準確率
            pbar.set_postfix(
                Loss=f"{loss_batch.item():.4f}",
                Accuracy=f"{100 * self.train_acc_metric.compute().item():.2f}%",
            )

        self.lr_scheduler.step()  # 更新學習率調整器
        train_acc = 100 * self.train_acc_metric.compute().item()  # 計算訓練準確率
        # 記錄訓練數據
        self.logger.add_scalar("Train/Loss", loss / len(self.train_loader), epoch)
        self.logger.add_scalar("Train/Accuracy", train_acc, epoch)
        self.logger.add_scalar(
            "Learning Rate", self.optimizer.param_groups[0]["lr"], epoch
        )

    def validate(self, epoch: int):
        """
        驗證模型

        Args:
            epoch (int): 當前epoch數

        Returns:
            val_acc (float): 驗證準確率
        """
        self.model.eval()  # 設置模型為評估模式
        self.val_acc_metric.reset()  # 重置驗證準確率計算器
        with torch.no_grad():  # 不計算梯度
            # 設置進度條
            pbar = tqdm(
                self.val_loader, desc=f"[{self.device}] Validate Epoch {epoch:2d}"
            )
            # Pytorch 標準驗證迴圈，不計算梯度
            for source, targets in pbar:
                source, targets = source.to(self.device), targets.to(self.device)
                output = self.model(source).squeeze()
                loss = self.criterion(output, targets.float())

                self.val_acc_metric.update(output, targets)  # 更新驗證準確率
                # 顯示當前損失和準確率
                pbar.set_description(
                    f"Accuracy {100 * self.val_acc_metric.compute().item():.2f}%",
                )

        val_acc = 100 * self.val_acc_metric.compute().item()  # 計算驗證準確率
        # 記錄驗證數據
        self.logger.add_scalar("Validate/Loss", loss, epoch)
        self.logger.add_scalar("Validate/Accuracy", val_acc, epoch)
        return val_acc

    def run(self):
        """主要訓練迴圈流程"""
        best_val_acc = 0.0  # 最佳驗證準確率
        self.prof.start()  # 開始設備瓶頸分析
        for epoch in range(self.epochs_run, self.max_epochs):
            self.prof.step()
            self.train_one_epoch(epoch)  # 訓練一個epoch
            val_acc = self.validate(epoch)  # 驗證模型
            self._save_snapshot(epoch)  # 保存快照

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(self.log_dir, "best_model.pt")
                torch.save(self.model.state_dict(), model_path)  # 保存最佳模型

        self.prof.stop()  # 停止設備瓶頸分析
