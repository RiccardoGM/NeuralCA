## Import libraries
import numpy as np
import matplotlib.pyplot as plt


# ************************************************ #
#                                                  #
#                    Functions                     #
#                                                  #
# ************************************************ #

#-----------------\\  random_image

def random_image(size=(500, 500), periodic=False):

    random_image = np.round(np.random.uniform(low=0.0, high=1.0, size=size) * 255).astype(np.uint8)

    if periodic:
        size = min(50, int(random_image.shape[0]/10))
        random_image = np.tile(random_image[0:size, 0:size], (10, 10))

    return random_image


#-----------------\\  display_image 

def display_image(image):

    plt.imshow(image, cmap='gray')
    plt.axis('off')  
    plt.show()


#-----------------\\  apply_periodic_padding 

def apply_periodic_padding(image, block_size):
    """
    apply periodic padding to the image, where the image "wraps around" at the edges
    """        
    padded_image = np.pad(image, 
                          ((block_size, block_size), (block_size, block_size)), 
                          mode='wrap')
    return padded_image


#-----------------\\  apply_crop 

def apply_crop(image, block_size):
    """
    crop the image 
    """        
    cropped_image = image[block_size:-block_size, block_size:-block_size]
    return cropped_image


#-----------------\\  inv_gaussian 

def inv_gaussian(x):

    inv_g_x = -1/(2**(0.6*x**2))+1

    return np.clip(inv_g_x, 0, 1)


#-----------------\\  abs 

def abs(x):

    abs_x = np.abs(x)
    
    return np.clip(abs_x, 0, 1)