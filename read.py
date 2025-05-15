import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from PIL import Image


def read_tif_image(file_name):
    
    file_path_lcp = Path(f"data/data_01/lcp/{file_name}.TIF")
    file_path_rcp = Path(f"data/data_01/rcp/{file_name}.TIF")

    img_lcp = mpimg.imread(file_path_lcp)
    img_rcp = mpimg.imread(file_path_rcp)

    img = img_lcp - img_rcp
    print(img)

    fig, axs = plt.subplots(1,1, figsize = (10,10))

    axs.imshow(img, cmap='gray')
    plt.show()



def read_tif_image_v2(file_name):
    
    file_path_lcp = Path(f"data/data_01/lcp/{file_name}.TIF")
    file_path_rcp = Path(f"data/data_01/rcp/{file_name}.TIF")
    
    img_lcp = Image.open(file_path_lcp)
    img_rcp = Image.open(file_path_rcp)

    print(img_lcp)
    #img = img_lcp - img_rcp

    fig, axs = plt.subplots(1,1, figsize = (10,10))

    axs.imshow(img_lcp, cmap='gray')
    plt.show()



def read_tif_image_v3(file_name):
    
    file_path_lcp = Path(f"data/2020-08/lcp/{file_name}.TIF")
    file_path_rcp = Path(f"data/2020-08/rcp/{file_name}.TIF")

    img_lcp = mpimg.imread(file_path_lcp)
    img_rcp = mpimg.imread(file_path_rcp)

    img = img_lcp - img_rcp
    print(img)

    fig, axs = plt.subplots(1,1, figsize = (10,10))

    axs.imshow(img, cmap='gray')
    plt.show()


read_tif_image_v3("04")