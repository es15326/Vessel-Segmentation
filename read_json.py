import json
import os

def load_data_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def extract_images_and_labels(data):
    images = []
    labels = []

    for entry in data:
        image_path = entry.get('image')
        label = entry.get('label')

        if image_path is not None and label is not None:
            images.append(os.path.basename(image_path))
            #labels.append(label)

    return images



