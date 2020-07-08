# This script will convert jpg images into png images so that they are ready to be resized
from PIL import Image
from PIL import ImageFile
import os


def jpg2png(img_dir,save_dir):
    im = Image.open(img_dir)
    im.save(save_dir)

def main():
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    poke_path = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\pokemon_images"
    poke_save_png = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\pokemon_png"
    poke_filenames = os.listdir(poke_path)

    print("Converting Pokemon JPG to PNG ...")
    for name in poke_filenames:
        name_no_ext = name.split(".")[0]
        jpg2png(poke_path+"\\"+name, poke_save_png+"\\"+name_no_ext+".png")

    print("Successfully Converted")

    digi_path = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\digimon_images"
    digi_save_png = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\digimon_png"
    digi_filenames = os.listdir(digi_path)

    print("Converting Digimon JPG to PNG ...")
    for name in digi_filenames:
        name_no_ext = name.split(".")[0]
        jpg2png(digi_path+"\\"+name, digi_save_png+"\\"+name_no_ext+".png")

    print("Successfully Converted")

if __name__ == "__main__":
    main()
