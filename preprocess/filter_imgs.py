# To obtain imaging digimon files, I downloaded them from https://wikimon.net/Visual_List_of_Digimon using https://chrome.google.com/webstore/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm?hl=en
# It downloads all images on a website and I need to filter out actual digimon images to random images such as logos
# Digimons or digital monsters have a 'mon' suffix in their name, that is used to filter out unwanted images

# I downloaded the pokemon images from https://www.pokemon.com/us/pokedex/ with the same chrome extension
# This does not include mega evolutions and only goes up to sun and moon (7th generation)
# The pokemon image files are labelled with numbers, that is used to filter out unwanted images
import os

def filter_irrelevant_img(path,poke_digi):
    
    names = os.listdir(path)
    not_relevant = []

    # If poke_digi == 0, it means the path is for pokemon images
    if poke_digi == 0:
    	for file_name in names:
            check = file_name[:-4]
            try:
                int(check)
            except ValueError:
                not_relevant += [file_name]
    # If poke_digi == 1, it means the path is for digimon images
    else:
    	for file_name in names:
            if "mon" not in file_name:
                not_relevant += [file_name]

    # Delete these files
    for file_name in not_relevant:
        full_path = path + "\\" + file_name
        os.remove(full_path)

def main():
    poke_path = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\pokemon_images"
    filter_irrelevant_img(poke_path,0)
    digi_path = "C:\\Users\\derek\\Desktop\\Coding stuff\\Projects\\pokemon_digimon\\digimon_images"
    filter_irrelevant_img(digi_path,1)

if __name__ == "__main__":
	main()