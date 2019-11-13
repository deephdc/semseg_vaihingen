"""
Methods to put together all interesting images and data
in order to be able to transfer all of it as a single file.
"""

import os
from PIL import Image
from fpdf import FPDF
import semseg_vaihingen.config as cfg

# Input and result images from the segmentation
files_any = ['{}/Input_image_patch.png'.format(cfg.DATA_DIR),
             '{}/Classification_map.png'.format(cfg.DATA_DIR)]

files_vaihingen = ['{}/Input_image_patch.png'.format(cfg.DATA_DIR),
                   '{}/Groundtruth.png'.format(cfg.DATA_DIR),
                   '{}/Classification_map.png'.format(cfg.DATA_DIR),
                   '{}/Error_map.png'.format(cfg.DATA_DIR)]

# Merge images and add a color legend
def merge_images(data_type):
    result = Image.new("RGB", (1280, 960))
    if data_type == 'vaihingen':
        files = files_vaihingen
    else:
        files = files_any
        
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
def create_pdf(image, prediction, data_type):
    pdf = FPDF()
    pdf.add_page()
    pdf.image(image, 0, 0, w=210)

    pdf.add_page()

    summary_vaihingen = {'title' : 'Labelwise Accuracy:',
                         'header': ['Label', 'Accuracy'],
                         'info'  : ['label_accuracy'],
                         'summary': ['Overall Accuracy, %:', 'overall_accuracy']}
                         
    summary_any = {'title' : 'Labelwise amount of pixels:',
                   'header': ['Label', 'pixels', 'fraction'],
                   'info'  : ['label_pixels', 'label_pixels_fraction'],
                   'summary': ['Total pixels:', 'total_pixels']}
               
    cell_width = [55, 35, 35]
    
    
    if data_type == 'vaihingen':
        summary_print = summary_vaihingen
    else:
        summary_print = summary_any
        
    # print title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, summary_print['title'], ln=1)

    # print header
    pdf.set_font('Arial', size=16)
    for i in range(len(summary_print['header'])):
        pdf.cell(cell_width[i], 12, summary_print['header'][i], ln=0)
    pdf.ln(12)
        
    #print entries 
    pdf.set_font('Arial', size=14)
    for label, value in prediction[summary_print['info'][0]].items():
        pdf.cell(cell_width[0], 10, "{}".format(label), ln=0)
        pdf.cell(cell_width[1], 10, "{}".format(value), ln=0)
        if len(summary_print['info']) == 2:
            pdf.cell(cell_width[2], 10, "{}".format(
                       prediction[summary_print['info'][1]][label]), ln=0)
        pdf.ln(10)

    pdf.set_font('Arial', 'B', 14)
    pdf.cell(cell_width[0], 16, summary_print['summary'][0], ln=0)
    pdf.cell(cell_width[1], 16, "{}".format(
                                     prediction[summary_print['summary'][1]]),
                 ln=1)

    results = '{}/prediction_results.pdf'.format(cfg.DATA_DIR)
    pdf.output(results,'F')

    return results
