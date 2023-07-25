import requests
from urllib.parse import unquote
from urllib.parse import urlparse, parse_qs, unquote
import os 
import fnmatch
import cv2
import os
import shutil
import mne
import soundfile as sf
import xml.etree.ElementTree as ET
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import re
import noisereduce as nr
from scipy.signal import wiener


def get_num_of_file(string):
    pattern = r'\[(\d+)\]'  # Regular expression pattern to match a number inside square brackets
    # Search for the pattern in the string
    match = re.search(pattern, string)
    # Check if a match was found
    if match:
        number = match.group(1)  # Extract the number from the match
        return number
    else:
        return None  # Return None if no match was found

def read_rml_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    triples = []
    for line in lines:
        line = line.strip()
        if line.startswith('#') or not line:
            continue  # Skip comments and empty lines
        if line.endswith('.'):
            line = line[:-1]  # Remove trailing period
        triple = line.split(None, 2)
        if line.__contains__('Type="ObstructiveApnea"'):
            triples.append(tuple(triple))
    return triples

def extract_starting_point_apnea(triples, num_of_file):
    apnea_stating_point = []
    pattern = r'Start="(\d+(\.\d+)?)'
    duration_pattern = r'Duration="(\d+)'
    can_generate_negative_example = False
    for i, triple in enumerate(triples):
        match = re.search(pattern, str(triple))
        duration = re.search(duration_pattern, str(triple))
        if match:
            start_value = float(match.group(1))
            apnea_duration = float(duration.group(1))
            if (3600 * int(num_of_file)) >= start_value >= (3600 * (int(num_of_file) - 1)):
                start = float(start_value - (3600 * (int(num_of_file) - 1)))
                end = start + apnea_duration
                if i + 1 < len(triples):  # if it's not the last apnea in the annotation file
                    next_apnea = re.search(pattern, str(triples[i + 1]))
                    if duration and next_apnea:
                        next_apnea_start_value = float(next_apnea.group(1))
                        can_generate_negative_example = (end + apnea_duration) < next_apnea_start_value
                apnea_stating_point.append([start, end, can_generate_negative_example])
    print("apnea_stating_point size = ")
    print(len(apnea_stating_point))
    return apnea_stating_point

def create_xml_file(image_filename, image_path,starting_point, width, height, depth, label_name, xmin, ymin, xmax, ymax):
    # Create the root element
    root = ET.Element("annotation")

    # Create sub-elements and add them to the root
    folder = ET.SubElement(root, "folder")
    folder.text = "dataset"

    filename = ET.SubElement(root, "filename")
    filename.text = image_filename

    path = ET.SubElement(root, "path")
    path.text = image_path

    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    size = ET.SubElement(root, "size")
    image_width = ET.SubElement(size, "width")
    image_width.text = str(width)
    image_height = ET.SubElement(size, "height")
    image_height.text = str(height)
    image_depth = ET.SubElement(size, "depth")
    image_depth.text = str(depth)

    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"

    object_element = ET.SubElement(root, "object")
    name = ET.SubElement(object_element, "name")
    name.text = label_name
    pose = ET.SubElement(object_element, "pose")
    pose.text = "Unspecified"
    truncated = ET.SubElement(object_element, "truncated")
    truncated.text = "0"
    difficult = ET.SubElement(object_element, "difficult")
    difficult.text = "0"

    bndbox = ET.SubElement(object_element, "bndbox")
    xmin_element = ET.SubElement(bndbox, "xmin")
    xmin_element.text = str(xmin)
    ymin_element = ET.SubElement(bndbox, "ymin")
    ymin_element.text = str(ymin)
    xmax_element = ET.SubElement(bndbox, "xmax")
    xmax_element.text = str(xmax)
    ymax_element = ET.SubElement(bndbox, "ymax")
    ymax_element.text = str(ymax)

    # Create the XML tree and write it to a file
    tree = ET.ElementTree(root)
    xml_filename = image_filename.replace(".png", ".xml")
    tree.write("labelXml/"+str(starting_point)+xml_filename)

    print(f"XML file '{xml_filename}' created successfully.")

def display_image_with_label(image_path,image_filename,starting_point, xmin, ymin, xmax, ymax):
    print(image_path)
    image = cv2.imread(image_path)
    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("imageLabel/"+image_filename+str(starting_point)+"label"+".png", image)

def create_spectogram(channel_data, sr, starting_point, file_name, suffix, sub_folder):
    mel_spec = librosa.feature.melspectrogram(y=channel_data, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    # Display the Mel spectrogram
    plt.axis('off')
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])  # Remove the white edge
    plt.figure(figsize=(12, 4))
    plt.margins(0)
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis=None, y_axis=None)
    if sub_folder == negatives:
        suffix += 'negative'
    plt.savefig(sub_folder + '/' + file_name.removesuffix('.edf') + suffix + str(starting_point) + '.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()
    is_exist = os.path.exists(positives)
    if not is_exist:
        os.makedirs(positives)
    is_exist = os.path.exists(negatives)
    if not is_exist:
        os.makedirs(negatives)

def extract_file_name(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    file_name = query_params.get('fileName', [''])[0]
    decoded_file_name = unquote(file_name)
    return decoded_file_name

def searchEdfFromRml(fileRml):
    list_edf = []
    print("file Rml = "+fileRml)
    with open(fileUrl, 'r', encoding='UTF-8') as file:
        while line := file.readline():
            url = line.rstrip()
            if url.find(fileRml) != -1 and url.find(".rml") == -1:
                fileName = extract_file_name(url)
                list_edf.append(fileName)
    return list_edf

def getUrl(fileToFind):
     with open(fileUrl, 'r', encoding='UTF-8') as file:
        while line := file.readline():
            url= line.rstrip() 
            fileName = extract_file_name(url)
            if fileName == fileToFind:
                return url

def check_file_exists(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        return True
    else:
        return False
    

fileUrl = "url_list.txt"
folder_path = 'edf'

with open(fileUrl, 'r', encoding='UTF-8') as file:
    while line := file.readline():
        url= line.rstrip() 
        fileName = extract_file_name(url)
        if '.rml' in fileName:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_path = os.path.join(folder_path, fileName)
            r = requests.get(url+".rml", allow_redirects=True)
            print(f"Downloaded {fileName}. Server response: {r.status_code} {r.reason}")
            open(file_path, 'wb').write(r.content)
            list_edf = searchEdfFromRml(fileName.replace(".rml", ""))
            print(list_edf)
            for edf in list_edf:
                if not check_file_exists(folder_path,edf):
                    file_path = os.path.join(folder_path, edf)
                    print(f"Requesting {edf}")
                    r = requests.get(getUrl(edf), allow_redirects=True)
                    print(f"Downloaded {edf}. Server response: {r.status_code} {r.reason}")
                    open(file_path, 'wb').write(r.content)
            edf_name = fileName.replace(".rml","")
            rml_file = os.path.join(folder_path, edf_name + '.rml')
            positives = 'positive_examples'
            negatives = 'negative_examples'
            output_files = []
            print(list_edf)
            for filename in list_edf:
                output_file = filename.removesuffix('.edf')
                output_file += '.wav'
                output_files += output_file

            # output_file = file_base_name + '.wav'
            channel_label = 'Tracheal'
            sampling_rate = 48000
            triples = read_rml_file(rml_file)


            # Extract from rml file 'ObstructiveApnea'

            def create_folders():
                is_exist = os.path.exists(positives)
                if not is_exist:
                    os.makedirs(positives)
                is_exist = os.path.exists(negatives)
                if not is_exist:
                    os.makedirs(negatives)
 
            create_folders()

            # Read file like data raw
            for edf_file in list_edf:
                print("Reading " + edf_file)
                raw = mne.io.read_raw_edf(os.path.join(folder_path, edf_file))
                channel_index = raw.ch_names.index(channel_label)
                # Read the data for the selected channel
                channel_data, times = raw[channel_index, :]
                # Apply noise reduction to our data
                # data_noise_reduced = wiener(channel_data[0])
                # data_noise_reduced = nr.reduce_noise(y=channel_data[0], sr=sampling_rate)
                apnee_starting_point_list = []
                apnee_starting_point_list += extract_starting_point_apnea(triples, get_num_of_file(edf_file))
                print(apnee_starting_point_list)
                for apnea_starting_point in apnee_starting_point_list:
                    delta = 5
                    start_index = int(apnea_starting_point[0] - delta) * sampling_rate  # 5 secondi prima dell'apnea
                    end_index = int(apnea_starting_point[1] + delta) * sampling_rate  # 5 secondi dopo la fine dell'apnea
                    channel = channel_data[0]
                    duration_in_seconds = apnea_starting_point[1] - apnea_starting_point[0]
                    # esempio positivo
                    cut_audio = channel[start_index:end_index]
                    data_reduced = wiener(cut_audio)
                    data_noise_reduced = nr.reduce_noise(y=cut_audio, sr=sampling_rate)
                    #data_reduced = data_reduced[delta * sampling_rate: int(delta + duration_in_seconds) * sampling_rate]
                    #data_noise_reduced = data_noise_reduced[delta * sampling_rate: int(delta + duration_in_seconds) * sampling_rate]
                    # segment_data = channel_data[start_index:end_index]
                    create_spectogram(data_reduced, sampling_rate, apnea_starting_point, edf_file, '__', positives)
                    create_spectogram(data_noise_reduced, sampling_rate, apnea_starting_point, edf_file, '_', positives)
                    imageName = edf_file.removesuffix('.edf') + '_' + str(apnea_starting_point) + '.png'
                    imagePath = positives + '/' + imageName
                    width = 930
                    height = 308
                    depth = 3
                    label_name = "apnea"
                    all_time = duration_in_seconds + (delta*2)
                    x_delta = (width * duration_in_seconds) / all_time
                    xmin, ymin, xmax, ymax = ((width/2) - (x_delta / 2)), 0, ((width/2) + (x_delta / 2)), height
                    create_xml_file(imageName,imagePath,apnea_starting_point,width,height,depth,label_name,xmin, ymin, xmax, ymax)
                    display_image_with_label(imagePath,edf_file,apnea_starting_point, xmin, ymin, xmax, ymax)
        shutil.rmtree('edf')

