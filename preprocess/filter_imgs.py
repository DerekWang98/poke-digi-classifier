# Author: Derek Wang
# To obtain imaging digimon files, I downloaded them from https://wikimon.net/Visual_List_of_Digimon using https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm?hl=en
# It downloads all images on a website and I need to filter out actual digimon images to random images such as logos
# Digimons or digital monsters have a 'mon' suffix in their name, that is used to filter out unwanted images

# I downloaded the pokemon images from https://www.pokemon.com/us/pokedex/ with the same chrome extension
# This does not include mega evolutions and only goes up to sun and moon (7th generation)
# The pokemon image files are labelled with numbers, that is used to filter out unwanted images
import os

def filter_poke_imgs(path):

    print("Filtering pokemon images ...")
    
    names = os.listdir(path)
    not_relevant = []

    for file_name in names:
        check = file_name[:-4]
        try:
            int(check)
        except ValueError:
            not_relevant += [file_name]

    # Delete these files
    for file_name in not_relevant:
        full_path = path + "\\" + file_name
        os.remove(full_path)

    print("Images filtered successfully ...")

def filter_digi_imgs(path):
    
    print("Filtering digimon images ...")

    names = os.listdir(path)
    not_relevant = []

    for file_name in names:
        if "mon" not in file_name:
            not_relevant += [file_name]

    # Delete these files
    for file_name in not_relevant:
        full_path = path + "\\" + file_name
        os.remove(full_path)

    print("Images filtered successfully!")

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
    poke_path = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\pokemon_images"
    filter_poke_imgs(poke_path)
    digi_path = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\image_data\\digimon_images"
    filter_digi_imgs(digi_path)

if __name__ == "__main__":
	main()