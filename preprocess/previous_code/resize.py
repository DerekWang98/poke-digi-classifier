# Author: Derek Wang
# All the pokemon files are 215 by 215 pixels
# The digimon images are of varying sizes
# It will be consistent all the images are resized into 215 by 215 pixels

from PIL import Image
from resizeimage import resizeimage
import os

def resize_img(img_dir,save_dir,width,height):
    # Resize the image to the given width and height
    fd_img = open(img_dir,'r+b')
    img = Image.open(fd_img)
    img = resizeimage.resize_contain(img, [width, height])
    img.save(save_dir, img.format)
    fd_img.close()

def main():
    """
    Inputs
    img_path: the directory containing the PNG images
    output_path: the directoty that will contain the resized PNG images
    """
    width = 40
    height = 40

    poke_path = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\pokemon_png"
    poke_save_resized = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\pokemon_png_small"
    digi_path = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\digimon_png"
    digi_save_resized = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\digimon_png_small"

    poke_filenames = os.listdir(poke_path)
    digi_filenames = os.listdir(digi_path)

    print("Resizing images ...")

    for name in poke_filenames:
        resize_img(poke_path+"\\"+name, poke_save_resized+"\\"+name, width, height)

    for name in digi_filenames:
        resize_img(digi_path+"\\"+name, digi_save_resized+"\\"+name, width, height)

    print("Resizing successfully completed!")

if __name__ == "__main__":
	main()
