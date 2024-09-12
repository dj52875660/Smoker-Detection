import argparse

import timm
from dataset import AUGMENTATION_TRANSFORMS, BASIC_TRANSFORMS, ImageDataset
from logger import Logger
from opts import opts
from torch.utils.data import DataLoader
from trainer import Trainer
from utils import seed_everything


def main(opt: argparse.Namespace, logger: Logger):
    seed_everything(opt.seed)  # 設定隨機種子
    train_dataset = ImageDataset(
        root=opt.data_dir,
        transforms=AUGMENTATION_TRANSFORMS(),
    )
    val_dataset = ImageDataset(
        root="Smoker_Detection/data/Testing",
        transforms=BASIC_TRANSFORMS(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    model = timm.create_model(opt.model_name, pretrained=opt.pretrained, num_classes=1)
    trainer = Trainer(opt, model, train_loader, val_loader, logger)
    trainer.run()
    logger.close()


if __name__ == "__main__":
    opt = opts().parse(["--model_name", "convnext_tiny.fb_in22k", "--gpu_id", "0"])
    # opt = opts().parse(["--resume", "--snapshot_path", "runs/.../best_model.pt"])
    logger = Logger(opt)
    main(opt, logger)
