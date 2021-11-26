import os
import shutil
import random

# Parameter
dataset_path = os.path.join("/tmp", "pair_space", "dataset")
dataset_path_abnormal = os.path.join(dataset_path, "abnormal")
dataset_path_normal = os.path.join(dataset_path, "normal")

source_path_abnormal = [
    os.path.join("/tmp", "pair_space", "abnormal-0.02"),
    os.path.join("/tmp", "pair_space", "abnormal-0.1"),
]
source_path_normal = os.path.join("/tmp", "pair_space", "normal")

# Create dataset path
for createPath in [dataset_path, dataset_path_abnormal, dataset_path_normal]:
    try:
        if not os.path.exists(createPath):
            os.makedirs(createPath)
    except OSError:
        print("Error: Creating directory. " + createPath)
        exit(1)


# Abnormal copy
fileCount = 0

for imgPath in source_path_abnormal:
    file_list = os.listdir(imgPath)
    file_list_img = [file for file in file_list if file.endswith(".jpg")]

    for imgFile in file_list_img:
        fileCount += 1
        filename1 = str(fileCount) + ".jpg"
        filename2 = str(fileCount) + "d.jpg"

        src_img_path = os.path.join(imgPath, imgFile)
        dest_img_path1 = os.path.join(dataset_path_abnormal, filename1)
        dest_img_path2 = os.path.join(dataset_path_abnormal, filename2)

        shutil.copyfile(src_img_path, dest_img_path1)
        shutil.copyfile(src_img_path, dest_img_path2)

        print("File copy", src_img_path, "success.")

# Normal copy
fileCount = 0

file_list = os.listdir(dataset_path_abnormal)
file_list_img = [file for file in file_list if file.endswith(".jpg")]

abnormal_img_count = len(file_list_img)

file_list = os.listdir(source_path_normal)
file_list_img = [file for file in file_list if file.endswith(".jpg")]

img_idx_list = random.sample(range(len(file_list_img)), abnormal_img_count)

for idx in img_idx_list:
    fileCount += 1
    filename = str(fileCount) + ".jpg"

    src_img_path = os.path.join(source_path_normal, file_list_img[idx])
    dest_img_path = os.path.join(dataset_path_normal, filename)

    shutil.copyfile(src_img_path, dest_img_path)

    print("File copy", src_img_path, "success.")
