from PIL import Image
import os

def convert_and_crop_image(input_path, output_path):
    """
    将 JPG 图片转换为 PNG 并裁剪到 600x400 像素。
    
    :param input_path: 输入的 JPG 图片路径
    :param output_path: 输出的 PNG 图片路径
    """
    try:
        # 打开 JPG 图片
        img = Image.open(input_path)
        
        # 获取图片尺寸
        width, height = img.size
        
        # 如果图片大于 600x400，裁剪图片
        if width > 600 and height > 400:
            left = (width - 600) // 2
            top = (height - 400) // 2
            right = left + 600
            bottom = top + 400
            
            # 裁剪图片
            img = img.crop((left, top, right, bottom))
        
        # 将图片转换为 PNG 格式并保存
        img.save(output_path, 'PNG')
        print(f"成功转换并保存图片：{output_path}")
        # 同时在 ../high 中也保存一份
        filename = os.path.basename(output_path)
        print(os.path.dirname(output_path))
        high_folder = os.path.join(os.path.dirname(output_path), os.pardir, 'high')
        if not os.path.exists(high_folder):
            os.makedirs(high_folder)
        output_path_high = os.path.join(high_folder, filename)
        img.save(output_path_high, 'PNG')
        print(f"成功转换并保存图片：{output_path_high}")
    
    except Exception as e:
        print(f"处理图片时出错：{e}")

def process_folder(input_folder, output_folder):
    """
    遍历文件夹中的所有 JPG 文件，并将它们转换为 PNG 格式，裁剪为 600x400。
    
    :param input_folder: 输入 JPG 图片的文件夹路径
    :param output_folder: 输出 PNG 图片的文件夹路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg'):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_folder, output_filename)
            
            # 转换和裁剪图片
            convert_and_crop_image(input_path, output_path)

if __name__ == '__main__':
    input_folder = './my_lr_img'  # 输入 JPG 图片的文件夹路径
    output_folder = './dataset/LOLdataset/eval15/low'  # 输出 PNG 图片的文件夹路径
    
    process_folder(input_folder, output_folder)
