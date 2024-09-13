import os
from pydiff.models.pydiff_model import PyDiffModel
import torch
# import torch.utils.data.dataloader
from basicsr.models import build_model
from basicsr.utils import get_root_logger, imfrombytes, img2tensor, tensor2img
from basicsr.utils.options import parse_options
from basicsr.utils.registry import MODEL_REGISTRY
from torchvision.transforms.functional import normalize
import pydiff.archs
import pydiff.data
import pydiff.models

from basicsr.utils import imfrombytes, imwrite
import imageio.v3 as iio
import argparse
import yaml  # 加载 YAML 文件

def load_model(opt):
    """加载 pydiff 模型."""
    model = build_model(opt)  # 使用 pydiff 的模型构建方法
    return model

def enhance_image(model : PyDiffModel, lr_img_path, save_path, gt_img_path = None):
    """利用模型对单张图片进行增强."""
    model.single_png_output(gt_img_path, lr_img_path, save_path)

def main(opt_path, lr_img_path, save_path, gt_img_path=None):
    # 读取模型配置，使用 yaml 加载器
    with open(opt_path, 'r') as f:
        opt = yaml.safe_load(f)
        opt['is_train'] = True  # 设置为推理模式
        opt['dist'] = False  # 设置为非分布式模式

    # 检查是否使用GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt['device'] = device

    # 加载模型
    model = load_model(opt)

    # 对图片进行增强
    enhance_image(model, lr_img_path, save_path, gt_img_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="单张图片暗光增强")
    parser.add_argument('opt', type=str, help="模型的配置文件路径")
    parser.add_argument('input', type=str, help="待增强的图片路径")
    parser.add_argument('output', type=str, help="增强后的图片保存路径")

    args = parser.parse_args()

    main(args.opt, args.input, args.output)