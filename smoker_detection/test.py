import argparse

import timm
import torch
from dataset import BASIC_TRANSFORMS
from opts import opts
from PIL import Image
import matplotlib.pyplot as plt
import os


def main(opt: argparse.Namespace, image: Image.Image) -> float:
    """測試模型並返回預測結果

    Args:
        opt (argparse.Namespace): 參數
        image (Image.Image): 輸入圖片

    Returns:
        logits (float): 預測結果 (0~1之間的機率值)
    """
    source = BASIC_TRANSFORMS()(image)

    model = timm.create_model(opt.model_name, pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(opt.load_model_path))
    model = model.to(opt.device)

    model.eval()
    with torch.no_grad():
        source = source.unsqueeze(0)
        source = source.to(opt.device)
        output = model(source).squeeze()
    print("Raw output:", output)
    logits = torch.sigmoid(output).item()
    return logits


def display_image_with_labels(
    image,
    true_label,
    predicted_label,
    save_path="Smoker_Detection/smoker_detection/output_plot/output_image.png",
):
    plt.imshow(image)
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.axis("off")  # Turn off the axis
    plt.savefig(save_path)  # Save the image to a file
    print(f"Image saved to {save_path}")


def extract_label_from_path(image_path):
    # Split the path and extract the label part
    filename = os.path.basename(image_path)  # Extracts 'smoking_0352.jpg'
    label = filename.split("_")[0]  # Extracts 'smoking' or 'notsmoking'
    return label


if __name__ == "__main__":
    """必須啟動test模式並指定模型權重路徑和輸入圖片路徑"""
    opt = opts().parse(
        [
            "--test",
            "--gpu_id",
            "1",
            "--load_model_path",
            "runs/default/2024-09-12-01-56-31/best_model.pt",
            "--image_path",
            "Smoker_Detection/data/Testing/smoking_0352.jpg",
            "--model_name",
            "convnext_tiny.fb_in22k",
        ]
    )
    # Load image
    with open(opt.image_path, "rb") as f:
        image = Image.open(opt.image_path).convert("RGB")
    logits = main(opt, image)  # 預測

    predicted_label = "smoking" if logits >= 0.5 else "notsmoking"

    # Assuming you know the true label (you can adjust this part based on your data)
    true_label = extract_label_from_path(opt.image_path)

    # Display the image with the true and predicted labels
    display_image_with_labels(image, true_label, predicted_label)

    print("Logits:", logits)
