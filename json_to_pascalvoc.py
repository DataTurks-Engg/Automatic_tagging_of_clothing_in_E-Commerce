import argparse
import sys
import os
import json
import logging
import requests
from PIL import Image 

''' running python json_to_pascalvoc.py FILENAME.json PATH_TO_OUTPUT_FOLDER  '''

logging.getLogger().setLevel(logging.INFO)

def maybe_download(image_url, image_dir):
    """Download the image if not already exist, return the location path"""
    fileName = image_url.split("/")[-1]
    filePath = os.path.join(image_dir, fileName)
    if (os.path.exists(filePath)):
        return filePath

    #else download the image
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(filePath, 'wb') as f:
                f.write(response.content)
                return filePath
        else:
            raise (ValueError,"Not a 200 response")
    except Exception as e:
        logging.exception("Failed to download image at " + image_url + " \n" + str(e) + "\nignoring....")
        raise e


def get_xml_for_bbx(bbx_label, bbx_data, width, height):

    # We store the left top and right bottom as point '0' and point '1'
    xmin = int(bbx_data['points'][0]['x']*width)
    ymin = int(bbx_data['points'][0]['y']*height)
    xmax = int(bbx_data['points'][1]['x']*width)
    ymax = int(bbx_data['points'][1]['y']*height)

    xml = "<object>\n"
    xml = xml + "\t<name>" + bbx_label + "</name>\n"
    xml = xml + "\t<pose>Unspecified</pose>\n"
    xml = xml + "\t<truncated>Unspecified</truncated>\n"
    xml = xml + "\t<difficult>Unspecified</difficult>\n"
    xml = xml + "\t<occluded>Unspecified</occluded>\n"
    xml = xml + "\t<bndbox>\n"
    xml = xml +     "\t\t<xmin>" + str(xmin) + "</xmin>\n"
    xml = xml +     "\t\t<xmax>" + str(xmax) + "</xmax>\n"
    xml = xml +     "\t\t<ymin>" + str(ymin) + "</ymin>\n"
    xml = xml +     "\t\t<ymax>" + str(ymax) + "</ymax>\n"
    xml = xml + "\t</bndbox>\n"
    xml = xml + "</object>\n"
    return xml


def convert_to_PascalVOC(dataturks_labeled_item, image_dir, xml_out_dir):

    """Convert a dataturks labeled item to pascalVOCXML string.
      Args:
        dataturks_labeled_item: JSON of one labeled image from dataturks.
        image_dir: Path to  directory to downloaded images (or a directory already having the images downloaded).
        xml_out_dir: Path to the dir where the xml needs to be written.
      Returns:
        None.
      Raises:
        None.
      """
    try:
        data = json.loads(dataturks_labeled_item)
        width = data['annotation'][0]['imageWidth']
        height = data['annotation'][0]['imageHeight']
        image_url = data['content']

        filePath = maybe_download(image_url, image_dir)

        with Image.open(filePath) as img:
            width, height = img.size

        fileName = filePath.split("/")[-1]
        image_dir_folder_Name = image_dir.split("/")[-1]


        xml = "<annotation>\n<folder>" + image_dir_folder_Name + "</folder>\n"
        xml = xml + "<filename>" + fileName +"</filename>\n"
        xml = xml + "<path>" + filePath +"</path>\n"
        xml = xml + "<source>\n\t<database>Unknown</database>\n</source>\n"
        xml = xml + "<size>\n"
        xml = xml +     "\t<width>" + str(width) + "</width>\n"
        xml = xml +    "\t<height>" + str(height) + "</height>\n"
        xml = xml +    "\t<depth>Unspecified</depth>\n"
        xml = xml +  "</size>\n"
        xml = xml + "<segmented>Unspecified</segmented>\n"

        for bbx in data['annotation']:
            bbx_labels = bbx['label']
            #handle both list of labels or a single label.
            if not isinstance(bbx_labels, list):
                bbx_labels = [bbx_labels]

            for bbx_label in bbx_labels:
                xml = xml + get_xml_for_bbx(bbx_label, bbx, width, height)

        xml = xml + "</annotation>"

        #output to a file.
        xmlFilePath = os.path.join(xml_out_dir, fileName + ".xml")
        with open(xmlFilePath, 'w') as f:
            f.write(xml)
        return True
    except Exception as e:
        logging.exception("Unable to process item " + dataturks_labeled_item + "\n" + "error = "  + str(e))
        return False

def main():
    #make sure everything is setup.
    if (not os.path.isdir(image_download_dir)):
        logging.exception("Please specify a valid directory path to download images, " + image_download_dir + " doesn't exist")
        return
    if (not os.path.isdir(pascal_voc_xml_dir)):
        logging.exception("Please specify a valid directory path to write Pascal VOC xml files, " + pascal_voc_xml_dir + " doesn't exist")
        return
    if (not os.path.exists(dataturks_JSON_FilePath)):
        logging.exception(
            "Please specify a valid path to dataturks JSON output file, " + dataturks_JSON_FilePath + " doesn't exist")
        return

    lines = []
    with open(dataturks_JSON_FilePath, 'r') as f:
        lines = f.readlines()

    if (not lines or len(lines) == 0):
        logging.exception(
            "Please specify a valid path to dataturks JSON output file, " + dataturks_JSON_FilePath + " is empty")
        return

    count = 0;
    success = 0
    for line in lines:
        status = convert_to_PascalVOC(line, image_download_dir, pascal_voc_xml_dir)
        if (status):
            success = success + 1

        count+=1;
        if (count % 10 == 0):
            logging.info(str(count) + " items done ...")

    logging.info("Completed: " + str(success) + " items done, " + str(len(lines) - success)  + " items ignored due to errors")


def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description='Converts Dataturks output JSON file for Image bounding box to Pascal VOC format.')
    parser.add_argument('dataturks_JSON_FilePath',
                    help='Path to the JSON file downloaded from Dataturks.')
    parser.add_argument('image_download_dir',
                    help='Path to the directory where images will be dowloaded (if not already found in the directory).')
    parser.add_argument('pascal_voc_xml_dir',
                        help='Path to the directory where Pascal VOC XML files will be stored.')
    return parser

if __name__ == '__main__':
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    global dataturks_JSON_FilePath
    global image_download_dir
    global pascal_voc_xml_dir

    #setup global paths needed accross the script.
    dataturks_JSON_FilePath = parsed_args.dataturks_JSON_FilePath
    image_download_dir = parsed_args.image_download_dir
    pascal_voc_xml_dir = parsed_args.pascal_voc_xml_dir
    main()
