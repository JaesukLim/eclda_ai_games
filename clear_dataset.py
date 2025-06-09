import os


def empty_train_data(image_dir="./dataset/images/train", label_dir="./dataset/labels/train"):
    """
    Deletes all image and label files in the train directories.

    Args:
        image_dir (str): Path to the train images directory.
        label_dir (str): Path to the train labels directory.
    """
    # Remove image files
    if os.path.isdir(image_dir):
        for filename in os.listdir(image_dir):
            file_path = os.path.join(image_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Emptied images in {image_dir}")
    else:
        print(f"Directory not found: {image_dir}")

    # Remove label files
    if os.path.isdir(label_dir):
        for filename in os.listdir(label_dir):
            file_path = os.path.join(label_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"Emptied labels in {label_dir}")
    else:
        print(f"Directory not found: {label_dir}")

empty_train_data(
    image_dir="./dataset/images/train",
    label_dir="./dataset/labels/train"
)
empty_train_data(
    image_dir="./dataset/images/val",
    label_dir="./dataset/labels/val"
)
