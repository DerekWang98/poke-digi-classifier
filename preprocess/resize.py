# All the pokemon files are 215 by 215 pixels
# The digimon images are of varying sizes
# It will be consistent all the images are resized into 215 by 215 pixels

from PIL import Image
from resizeimage import resizeimage
import os

def resize_img(img_dir,save_dir,width,height):

    fd_img = open(img_dir,'r+b')
    img = Image.open(fd_img)
    img = resizeimage.resize_contain(img, [215, 215])
    img.save(save_dir, img.format)
    fd_img.close()

def main():
    
    width = 215
    height = 215

    poke_path = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\pokemon_png"
    poke_save_resized = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\pokemon_png_resized"
    digi_path = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\digimon_png"
    digi_save_resized = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\digimon_png_resized"

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