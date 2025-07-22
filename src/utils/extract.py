import os
import shutil


def extract_images_to_root(folder_path):
    # 支持的图片扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path, topdown=False):
        if root == folder_path:
            continue  # 跳过根目录

        for file in files:
            if file.lower().endswith(image_extensions):
                src_path = os.path.join(root, file)

                # 处理可能存在的重名文件
                dest_path = os.path.join(folder_path, file)
                counter = 1
                while os.path.exists(dest_path):
                    name, ext = os.path.splitext(file)
                    dest_path = os.path.join(folder_path, f"{name}_{counter}{ext}")
                    counter += 1

                # 移动文件
                shutil.move(src_path, dest_path)
                print(f"Moved: {src_path} -> {dest_path}")

        # 尝试删除空文件夹
        try:
            if not os.listdir(root):
                os.rmdir(root)
                print(f"Deleted empty folder: {root}")
        except Exception as e:
            print(f"Error deleting folder {root}: {e}")


if __name__ == "__main__":
    folder_path = input("请输入文件夹路径: ").strip()

    if os.path.isdir(folder_path):
        extract_images_to_root(folder_path)
        print("操作完成！")
    else:
        print("错误: 提供的路径不是一个有效的文件夹。")