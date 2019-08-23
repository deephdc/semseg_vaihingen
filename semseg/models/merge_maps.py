import os

from PIL import Image
import semseg.config as cfg

files = ['{}/Input_image_patch.png'.format(cfg.DATA_DIR),
         '{}/Classification_map.png'.format(cfg.DATA_DIR),
         '{}/Groundtruth.png'.format(cfg.DATA_DIR),
         '{}/Error_map.png'.format(cfg.DATA_DIR)]


def merge_images():
    result = Image.new("RGB", (800, 800))
    for index, path in enumerate(files):
        img = Image.open(path)
        img.thumbnail((400, 400), Image.ANTIALIAS)
        x = index // 2 * 400
        y = index % 2 * 400
        w, h = img.size
        print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
        result.paste(img, (x, y, x + w, y + h))
    
    file = '{}/merged_maps.jpg'.format(cfg.DATA_DIR)
    result.save(file)

    return file
