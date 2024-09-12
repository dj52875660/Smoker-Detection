import argparse
import os
import time


class opts(object):
    def __init__(self):
        # 建立一個 ArgumentParser 物件，用於處理命令列參數
        self.parser = argparse.ArgumentParser(
            description="Binary Classification Training Script"
        )

        """基本實驗設定"""
        self.parser.add_argument("--exp_id", default="default")  # 實驗名稱
        self.parser.add_argument(
            "--data_dir",
            type=str,
            default="data/Training",
            help="dataset directory",  # 資料集目錄
        )
        self.parser.add_argument(
            "--seed", type=int, default=42, help="seed"
        )  # 隨機種子

        """系統設定"""
        self.parser.add_argument(
            "--gpu_id",
            type=int,
            default=0,  # 默認使用第一個 GPU
            help="-1 for CPU, use comma for single GPU, e.g. 0,1",  # 指定 GPU 編號，-1 為 CPU
        )

        """模型設定"""
        self.parser.add_argument(
            "--model_name",
            type=str,
            default="resnet18",
            help="model name from timm",  # timm模型名稱，模型列表：https://rwightman.github.io/pytorch-image-models/
        )
        self.parser.add_argument(
            "--pretrained",
            type=bool,
            default=True,
            help="use pretrained model weights",  # 是否使用預訓練模型權重
        )

        """訓練超參數設定"""
        self.parser.add_argument(
            "--max_epochs",
            type=int,
            default=30,
            help="max epochs to train",  # 最大訓練週期
        )
        self.parser.add_argument(
            "--batch_size", type=int, default=128, help="batch size"  # 批次大小
        )
        self.parser.add_argument(
            "--num_workers",
            type=int,
            default=8,
            help="number of workers",  # 資料加載器的工作數量(cpu)
        )
        self.parser.add_argument(
            "--lr", type=float, default=1e-4, help="learning rate"
        )  # 學習率
        self.parser.add_argument(
            "--eta_min",
            type=float,
            default=1e-5,
            help="minimum learning rate",  # 最小學習率
        )

        """接續訓練設定"""
        self.parser.add_argument(
            "--resume",
            action="store_true",
            help="resume an experiment.",  # 是否接續訓練
        )
        self.parser.add_argument(
            "--snapshot_path",
            type=str,
            default=None,
            help="path to the snapshot to resume",  # 接續訓練的快照路徑
        )

        """測試設定"""
        self.parser.add_argument("--test", action="store_true")  # 是否進行測試
        self.parser.add_argument(
            "--load_model_path",
            type=str,
            default=None,
            help="path to trained model to load for testing",  # 測試用的訓練模型路徑
        )
        self.parser.add_argument(
            "--image_path",
            type=str,
            default=None,
            help="path to image to test",  # 測試用的圖片路徑
        )

    def parse(self, args=""):
        # 解析命令列參數
        if args == "":
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        # 設定根目錄
        opt.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # 設定設備編號，如果 gpu_id 為 -1 則使用 CPU
        opt.device = f"cuda:{opt.gpu_id}" if opt.gpu_id >= 0 else "cpu"

        # 檢查資料集目錄是否存在，不存在則拋出異常
        opt.data_dir = os.path.join(opt.root_dir, opt.data_dir)
        if not os.path.exists(opt.data_dir):
            raise ValueError(f"Data directory {opt.data_dir} not found")

        # 設定日誌目錄
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
        opt.log_dir = os.path.join(opt.root_dir, "runs", opt.exp_id, f"{time_str}")

        # 建立日誌目錄
        if not opt.test and not opt.resume:  # 如果不是測試且不是接續訓練
            if not os.path.exists(opt.log_dir):  # 如果日誌目錄不存在, 則建立日誌目錄
                os.makedirs(opt.log_dir, exist_ok=True)
                print(f"Created directory {opt.log_dir}")

        # 處理接續訓練的情況
        if opt.resume:
            snapshot_path = os.path.join(opt.root_dir, opt.snapshot_path)
            if not os.path.isfile(snapshot_path) or not snapshot_path.endswith(
                "snapshot.pt"
            ):  # 如果快照文件不存在或名稱不是 snapshot.pt 文件
                raise ValueError(
                    f"Snapshot file {snapshot_path} is not a valid .pt file or does not exist"
                )
            opt.snapshot_path = snapshot_path
            # 使用舊的日誌目錄
            opt.log_dir = os.path.dirname(snapshot_path)

        # 處理測試的情況
        if opt.test:
            load_model_path = os.path.join(opt.root_dir, opt.load_model_path)
            if not os.path.isfile(load_model_path) or not load_model_path.endswith(
                ".pt"
            ):  # 如果模型文件不存在或名稱不是 .pt 文件
                raise FileNotFoundError(
                    f"Model not found at {opt.load_model_path} for testing"
                )
            opt.load_model_path = load_model_path

        return opt
