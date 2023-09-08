import os
import shutil


def copy_unique_images(source_folder, destination_folder):
    """
    Sample one image for each of the subjects and discard other similar frames.
    Each subject is identified by the first part of the image name.
    Save the results to the destination folder.

    """
    image_groups = {}

    # Iterate through all images in the source folder
    for filename in os.listdir(source_folder):
        filepath = os.path.join(source_folder, filename)

        # Check if it is a images
        if os.path.isfile(filepath):
            # Get the first part of the image name
            group_name = filename.split('_')[0]

            # Add the image to the corresponding group
            if group_name in image_groups:
                image_groups[group_name].append(filepath)
            else:
                image_groups[group_name] = [filepath]

    # Iterate through each image group and select one image to copy to the target folder
    for group_images in image_groups.values():
        if group_images:
            selected_image = group_images[0]
            shutil.copy(selected_image, destination_folder)


def process_folder(input_folder, output_folder):
    """
    given one dataset folder which contains several sub_folders for different categories,
    return one dataset with its sub_folders modified by some methods (in this case,copy_unique_images())
    :param input_folder:  the path of the original dataset
    :param output_folder:  the output path of the modified dataset
    """
    subfolders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]
    for subfolder in subfolders:
        input_path = os.path.join(input_folder, subfolder)
        output_path = os.path.join(output_folder, subfolder)
        # Create output folder, if it does not exist
        os.makedirs(output_path, exist_ok=True)
        # for each sub-folder, apply  copy_unique_images()
        copy_unique_images(input_path, output_path)

    print("complete!")


# remove the similar frames in the CK+ dataset.
source_folder = '../datasets/CK+_raw'
destination_folder = '../datasets/CK+'
process_folder(source_folder, destination_folder)
