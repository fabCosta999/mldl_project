import os
import shutil
import urllib.request
import zipfile

url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
zip_path = "dataset/tiny-imagenet-200.zip"

print("Downloading Tiny ImageNet...")
urllib.request.urlretrieve(url, zip_path)
print("Download complete.")

extract_dir = "tiny-imagenet"

print("Extracting zip file...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
print("Extraction complete.")

val_dir = os.path.join(extract_dir, "tiny-imagenet-200", "val")
images_dir = os.path.join(val_dir, "images")
annotations_file = os.path.join(val_dir, "val_annotations.txt")

print("Reorganizing validation folder...")

with open(annotations_file, "r") as f:
    for line in f:
        fname, cls, *_ = line.split('\t')
        
        class_dir = os.path.join(val_dir, cls)
        os.makedirs(class_dir, exist_ok=True)

        src = os.path.join(images_dir, fname)
        dst = os.path.join(class_dir, fname)

        if os.path.exists(src):
            shutil.copyfile(src, dst)

print("Cleaning up...")
shutil.rmtree(images_dir)

print("Done! TinyImageNet is ready.")
