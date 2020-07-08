# Author: Derek Wang
# Inputs
# img_path: the directory containing the JPG images
# save_path: the directoty that will contain the saved PNG images
# mode: "jpg2png" or "resize" or "filter"
# "jpg2png" converts JPG images to PNG to prepare the images for resizing
# "resize" takes in PNG images and resizes them to a specified width and height
# "filter" because the images are obtained through an extension that downloads all the images in the webpage, this filters the irrelevant images and deletes them

from PIL import Image
from PIL import ImageFile
from resizeimage import resizeimage
import os
import sys

def jpg2png(img_dir,save_dir):
    # Converts JPG images in the img_dir to PNG images
    # Resized images are saved into the save_dir
    filenames = os.listdir(img_dir)

    print("Converting JPG to PNG ...")
    for name in filenames:
        name_no_ext = name.split(".")[0]
        im = Image.open(os.path.join(img_dir,name))
        im.save(os.path.join(save_dir,name_no_ext)+".png")

    print("Successfully Converted!")

def resize_img(img_dir,save_dir,width,height):
    # Resize the images in the img_dir to the given width and height
    # Resized images are saved into the save_dir
    filenames = os.listdir(img_dir)

    print("Resizing images ...")
    for name in filenames:

        fd_img = open(os.path.join(img_dir,name),'r+b')
        img = Image.open(fd_img)
        img = resizeimage.resize_contain(img, [width, height])
        img.save(os.path.join(save_dir,name), img.format)
        fd_img.close()

    print("Resizing successfully completed!")

def filter_imgs(path,poke_digi):

    print("Filtering " + poke_digi + " images ...")

    names = os.listdir(path)
    not_relevant = []

    # Finds irrelevant files
    for file_name in names:
        if poke_digi == "Digi":
            # Filter Digimon images - if they contain "mon" in their names
            if "mon" not in file_name:
                not_relevant += [file_name]

        elif poke_digi == "Poke":
            # Filter Pokemon images - they are labelled in numerical order
            check = file_name[:-4]
            try:
                int(check)
            except ValueError:
                not_relevant += [file_name]

    # Delete these files
    for file_name in not_relevant:
        full_path = path + "\\" + file_name
        os.remove(full_path)

    print(poke_digi + " images filtered successfully!")

def main():
    """
    Inputs
    img_path: the directory containing the JPG images
    save_path: the directoty that will contain the saved PNG images
    mode: "jpg2png" or "resize" or "filter"
    """
    if len(sys.argv)!=4:
        print("Warning: Incorrect Usage. Example: python jpg2png.py img_path save_path [mode]")
        return

    # Accepting user inputs
    img_path = sys.argv[1]
    save_path = sys.argv[2]

    if mode == "jpg2png":
        # Converts JPG images to PNG images
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        jpg2png(img_path,save_path)

    elif mode == "resize":
        # Resizes the image to a certain width and height
        # Ask user for width and height inputs
        # Future work - sanitise inputs
        try:
            width = int(input("Enter width:"))
            height = int(input("Enter height:"))
        except ValueError:
            print("Input Error: Integers only!")
            return

        resize_img(img_dir,save_dir,width,height)

    elif mode == "filter":
        # Filters and removes irrelevant files
        filter_imgs(img_path,save_path)


if __name__ == "__main__":
    main()