from PIL import Image

def crop_to_256(img):
    width, height = img.size
    new_height = 256
    left = 0
    bottom = height
    right = width
    top = bottom - new_height
    return img.crop((left, top, right, bottom))

def preprocess(img, target="image"):
    new_width = 480
    scale = img.width // 480
    new_height = img.height // scale
    new_size = (new_width, new_height)
    if target == "label":
        new_image = img.resize(new_size, Image.NEAREST)
    else:
        new_image = img.resize(new_size)
    new_image = crop_to_256(new_image)

    return new_image