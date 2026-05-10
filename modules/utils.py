## Import libraries
import numpy as np
import matplotlib.pyplot as plt


# ************************************************ #
#                                                  #
#                    Functions                     #
#                                                  #
# ************************************************ #

#-----------------\\  image_to_state

def image_to_state(image, low=-1.0, high=1.0):
    w_range = high - low
    state = low + (image / 255) * w_range

    return state

#-----------------\\  state_to_image

def state_to_image(state, low=-1.0, high=1.0):
    w_range = high - low
    image = (state-low) * 255 / w_range

    return image

#-----------------\\  random_state

def random_state(size=(500, 500), periodic=False, low=-1.0, high=1.0):
    state = np.random.uniform(low=low, high=high, size=size)

    if periodic:
        size = min(50, int(state.shape[0]/10))
        state = np.tile(state[0:size, 0:size], (10, 10))

    return state

#-----------------\\  random_image

def random_image(size=(500, 500), periodic=False, low=-1.0, high=1.0):
    state = random_state(size=size, periodic=periodic, low=low, high=high)
    image = state_to_image(state, low=low, high=high)

    if periodic:
        size = min(50, int(image.shape[0]/10))
        image = np.tile(image[0:size, 0:size], (10, 10))

    return image


#-----------------\\  display_image 

def display_image(image, vmin=None, vmax=None):

    plt.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
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

def inv_gaussian_clip(x):

    inv_g_x = -1/(2**(0.6*x**2))+1

    return np.clip(inv_g_x, 0, 1)


#-----------------\\  abs 

def abs_clip(x):

    abs_x = np.abs(x)
    
    return np.clip(abs_x, 0, 1)


#-----------------\\  n_free_parameters
 
def n_free_parameters(f_symm, hv_symm, h_symm, v_symm, kernel_size, fixed_center=False):

    ks = kernel_size

    if f_symm:
        na = (ks//2)**2
        nb = (ks//2)
        nc = 0
        if not fixed_center:
            nc = 1
        n = na + nb + nc

    elif hv_symm:
        na = (ks//2)**2
        nb = (ks//2)
        nc = (ks//2)
        nd = 0
        if not fixed_center:
            nd = 1
        n = na + nb + nc + nd

    elif h_symm:
        if fixed_center:
            raise ValueError("Fixed center not allowed for horizontal symmetry only")
        na = (ks//2)*ks
        nb = ks
        n = na + nb

    elif v_symm:
        if fixed_center:
            raise ValueError("Fixed center not allowed for vertical symmetry only")
        na = (ks//2)*ks
        nb = ks
        n = na + nb

    else:
        if fixed_center:
            raise ValueError("Fixed center not allowed without symmetries")
        n = ks**2

    return n
