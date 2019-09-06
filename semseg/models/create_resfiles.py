"""
Methods to put together all interesting images and data
in order to be able to transfer all of it as a single file.
"""

import os
from PIL import Image
from fpdf import FPDF
import semseg.config as cfg

# Input and result images from the segmentation
files = ['{}/Input_image_patch.png'.format(cfg.DATA_DIR),
         '{}/Groundtruth.png'.format(cfg.DATA_DIR),
         '{}/Classification_map.png'.format(cfg.DATA_DIR),
         '{}/Error_map.png'.format(cfg.DATA_DIR)]


# Merge images and add a color legend
def merge_images():
    result = Image.new("RGB", (1280, 960))
    for index, path in enumerate(files):
        img = Image.open(path)
        img.thumbnail((640, 480), Image.ANTIALIAS)
        x = index % 2 * 640
        y = index // 2 * 480
        w, h = img.size
        #print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
        result.paste(img, (x, y, x + w, y + h))
    
    merged_file = '{}/merged_maps.png'.format(cfg.DATA_DIR)
    result.save(merged_file)

    return merged_file


# Put images and accuracy information together in one pdf file
def create_pdf(image, prediction):
    pdf = FPDF()
    pdf.add_page()
    pdf.image(image, 0, 0, w=210)

    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Labelwise Accuracy:', ln=1)
    pdf.set_font('Arial', size=14)
    for label, value in prediction["label_accuracy"].items():
        pdf.cell(0, 10, '{}: \t\t {}'.format(label, value), ln=1)

    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Overall Accuracy: {}%'.format(prediction["overall_accuracy"]), ln=1)

    results = '{}/prediction_results.pdf'.format(cfg.DATA_DIR)
    pdf.output(results,'F')

    return results
