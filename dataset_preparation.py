import os
import xml.etree.ElementTree as ET
import shutil

def parse_cvat_annotations(file_path):
    """
    Parse CVAT annotations in XML format to extract line coordinates and image dimensions.

    Args:
        file_path (str): Path to the CVAT XML file.

    Returns:
        dict: A dictionary with keys as image names and values as line annotations.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    annotations = {}

    for image_tag in root.findall('image'):
        image_name = image_tag.get('name')
        width = int(image_tag.get('width'))
        height = int(image_tag.get('height'))
        subset = image_tag.get('subset')
        # width = 400
        # height = 400
        lines = []

        for polyline in image_tag.findall("polyline"):
            points = polyline.get('points')
            coords = [float(coord) for coord in points.replace(';', ',').split(',')]
            if len(coords) == 4:  # Ensure the polyline is a single line segment
                x1, y1, x2, y2 = coords
                lines.append((x1, x2, y1, y2, width, height))

        if lines:  # Only include images with at least one polyline
            annotations[image_name] = {'subset': subset, 'lines':lines}

    return annotations

def save_annotations(annotations, output_dir, split_mapping):
    """
    Save line annotations to .txt files in the specified format.

    Args:
        annotations (dict): A dictionary with image names as keys and line data as values.
        output_dir (str): Directory to save the label files.
        split_mapping (dict): A mapping of image names to their dataset split ('train' or 'test').
    """
    os.makedirs(output_dir, exist_ok=True)
    train_labels = {}
    test_labels = {}

    for idx, (image_name, annotation) in enumerate(annotations.items(), start=1):
        label_file = f"{idx:04d}.txt"
        if split_mapping[image_name] == "train":
            train_labels[image_name] = label_file
        elif split_mapping[image_name] == "test" or split_mapping[image_name] == "val":
            test_labels[image_name] = label_file

        with open(os.path.join(output_dir, label_file), "w") as f:
            for line in annotation['lines']:
                f.write(", ".join(map(str, line)) + "\n")

    return train_labels, test_labels

def organize_dataset(cvat_images_dir, annotations, output_root):
    """
    Organize the dataset into the desired format.

    Args:
        cvat_images_dir (str): Root directory containing 'default', 'Test', and 'Train' subfolders.
        annotations (dict): Parsed annotations.
        output_root (str): Root directory for the organized dataset.
    """
    labels_dir = os.path.join(output_root, "labels")
    images_output_dir = os.path.join(output_root, "images")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)

    train_idx_file = os.path.join(output_root, "train_idx.txt")
    test_idx_file = os.path.join(output_root, "test_idx.txt")

    # Mapping of images to dataset split
    split_mapping = {}
    for folder_name, split_name in [("Train", "train"), ("Test", "test"), ("Validation", "val")]:
        folder_path = os.path.join(cvat_images_dir, folder_name)
        if os.path.exists(folder_path):
            for image_name in os.listdir(folder_path):
                if image_name in annotations and annotations[image_name]['subset'] == folder_name:  # Only include images with annotations
                    split_mapping[image_name] = split_name

    # Save annotations and create train/test index files
    train_labels, test_labels = save_annotations(annotations, labels_dir, split_mapping)

    with open(train_idx_file, "w") as f:
        for image_name in train_labels.values():
            f.write(f"{image_name[:-4]}\n")  # Write file index without '.txt'

    with open(test_idx_file, "w") as f:
        for image_name in test_labels.values():
            f.write(f"{image_name[:-4]}\n")  # Write file index without '.txt'

    # Copy images to output directory
    for image_name, split in split_mapping.items():
        subset = "Train" if split == "train" else "Test" if split == "test" else "Validation"
        src_image_path = os.path.join(cvat_images_dir, subset, image_name)
        idx = list(annotations.keys()).index(image_name) + 1
        dest_image_path = os.path.join(images_output_dir, f"{idx:04d}.jpg")
        shutil.copy2(src_image_path, dest_image_path)

if __name__ == "__main__":
    # Input paths
    cvat_annotation_file = "../cvat_export_full/annotations.xml"  # Update with your file path
    cvat_images_dir = "../cvat_export_full/images"  # Root directory with 'default', 'Test', 'Train'

    # Output paths
    output_root = "test_dataset"

    # Parse annotations and organize dataset
    annotations = parse_cvat_annotations(cvat_annotation_file)
    organize_dataset(cvat_images_dir, annotations, output_root)
