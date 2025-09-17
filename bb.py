import cv2
import numpy as np
import argparse
import sys
import os

def check_and_read_image(image_path):
    """
    检查并读取图像文件，返回图像对象和状态
    """
    if not os.path.exists(image_path):
        print(f"错误：文件 '{image_path}' 不存在")
        return None, False
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误：无法读取图像 '{image_path}'，可能不是有效的图像文件")
        return None, False
        
    return img, True

def get_black_border_rect(image):
    """
    获取图像中黑色边框的矩形区域
    """
    if image is None:
        return None
        
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 二值化处理，将非黑色像素设为白色
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # 获取最大轮廓的边界框
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    return (x, y, w, h)

def draw_black_borders(image):
    """
    在图像上绘制黑色边框的矩形
    """
    if image is None:
        return None
        
    # 获取黑色边框的矩形
    border_rect = get_black_border_rect(image)
    
    if border_rect is None:
        return image
    
    x, y, w, h = border_rect
    
    # 创建图像副本
    image_with_border = image.copy()
    
    # 绘制矩形
    cv2.rectangle(image_with_border, (x, y), (x+w, y+h), (0, 0, 255), 3)  # 红色边框，线宽为3
    
    # 添加文本说明
    text = f"Border: ({x}, {y}, {w}, {h})"
    cv2.putText(image_with_border, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return image_with_border

def crop_black_borders(image):
    """
    裁剪图像的黑色边框
    """
    if image is None:
        return None
        
    # 获取黑色边框的矩形
    border_rect = get_black_border_rect(image)
    
    if border_rect is None:
        return image
    
    x, y, w, h = border_rect
    
    # 裁剪图像
    cropped = image[y:y+h, x:x+w]
    
    return cropped

def main():
    # 设置参数解析
    parser = argparse.ArgumentParser(description='图像拼接')
    parser.add_argument('--images', nargs='+', help='输入图像路径列表', required=True)
    parser.add_argument('--output', type=str, default='panorama_result.jpg', help='输出图像路径')
    parser.add_argument('--no-crop', action='store_false', dest='crop', 
                       help='不裁剪黑色边框', default=False)
    parser.add_argument('--show-border', action='store_true', 
                       help='显示黑色边框的位置', default=False)
    args = parser.parse_args()
    
    # 读取图像
    images = []
    for image_path in args.images:
        img, success = check_and_read_image(image_path)
        if not success:
            print(f"请确保图像文件 '{image_path}' 存在且可访问")
            return
        images.append(img)
    
    if len(images) < 2:
        print("至少需要两张图像进行拼接")
        return
    
    print(f"成功读取 {len(images)} 张图像")
    
    # 初始化OpenCV的拼接器
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, panorama = stitcher.stitch(images)
    
    if status != cv2.Stitcher_OK:
        print(f"拼接失败，错误代码: {status}")
        return
    
    print("OpenCV拼接器成功")
    
    # 如果需要显示黑色边框
    if args.show_border:
        border_image = draw_black_borders(panorama)
        if border_image is not None:
            cv2.imshow('Black Border', border_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # 如果需要裁剪黑色边框
    if args.crop:
        panorama = crop_black_borders(panorama)
    
    # 保存结果
    cv2.imwrite(args.output, panorama)
    print(f"拼接完成，结果已保存至: {args.output}")
    
    # 显示最终结果
    cv2.imshow('Final Result', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 如果没有通过命令行参数运行，则使用默认图像路径
    if len(sys.argv) == 1:
        # 替换为您的图像路径
        image_paths = ["1.png", "3.png"]
        
        # 检查当前目录下是否有图像文件
        found_images = []
        for path in image_paths:
            if os.path.exists(path):
                found_images.append(path)
            else:
                print(f"警告：未找到文件 '{path}'")
        
        if len(found_images) >= 2:
            # 添加 --show-border 参数以显示黑色边框
            sys.argv = [sys.argv[0]] + ["--images"] + found_images + ["--output", "panorama_result.jpg", "--show-border"]
        else:
            print("错误：未找到足够的图像文件进行拼接")
            print("请使用命令行参数指定图像路径：")
            print("python script.py --images image1.jpg image2.jpg [image3.jpg ...] --output result.jpg [--show-border]")
            sys.exit(1)
    
    main()