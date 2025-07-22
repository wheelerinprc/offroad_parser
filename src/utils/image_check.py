from PIL import Image
import os
from torchvision import transforms

if __name__=='__main__':
    output_folder = "../Rellis_3D_pylon_camera_node_label_color_480/train"
    output_image_list = os.listdir(output_folder)
    # output_first_image = output_image_list[0]
    output_first_image = "frame000916-1581623881_949.png"
    output_image = Image.open(os.path.join(output_folder, output_first_image)).convert("L")
    # pixel = output_image.getpixel((1, 1))
    # print(f"Pixel at ({1000}, {1000}): {pixel}")

    pixels = output_image.load()
    width, height = output_image.size

    # Find coordinates where pixel value > 20
    coordinates = [(x, y) for y in range(height) for x in range(width) if pixels[x, y] > 20]

    print(coordinates)
    transform = transforms.PILToTensor()
    output_tensor = transform(output_image)
    print(output_tensor)
    pixel = output_image.getpixel(coordinates[0])
    print(f"Pixel at ({1000}, {1000}): {pixel}")

    input_folder = "../Rellis_3D_pylon_camera_node_label_color_480/train"
    input_first_image = "frame000916-1581623881_949.jpg"
    # input_image_list = os.listdir(input_folder)
    # input_first_image = input_image_list[0]
    input_image = Image.open(os.path.join(input_folder, input_first_image)).convert("RGB")
    # input_pixel = input_image.getpixel(coordinates[0])
    input_pixel = input_image.getpixel((123,123))
    print(f"Pixel at ({1000}, {1000}): {input_pixel}")
    # pixel = input_image.getpixel((500, 500))
    # print(f"Pixel at ({1000}, {1000}): {pixel}")
    # print(input_image)