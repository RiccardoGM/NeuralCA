## Import libraries
import numpy as np
import gymnasium as gym
import cv2
import skimage as ski 
from scipy.stats import gmean

## Import custom modules
import sys
import os
# Append the parent directory to the sys.path
this_path = os.path.abspath('') 
parent_dir = os.path.dirname(this_path)  
sys.path.append(parent_dir)
from modules import utils


# ************************************************ #
#                                                  #
#                       Bot                        #
#                                                  #
# ************************************************ #

class Bot:
    def __init__(self, kernel=[], kernel_size=3, h_symm=True, v_symm=True, fixed_center=True, 
                 activation='identity', kernel_prec=1, scale_kernel=False):
        '''
        note: use odd numbers for kernel_size
        '''
        # Initialise attributes
        if len(kernel)==0:
            kernel = np.random.uniform(low=-1., high=1.0, size=(kernel_size, kernel_size)).astype(np.float32)
            if h_symm:
                kernel[:,kernel_size-kernel_size//2:] = kernel[:,:kernel_size//2]
            if v_symm:
                kernel[kernel_size-kernel_size//2:,:] = kernel[:kernel_size//2,:]
            if fixed_center:
                kernel[kernel_size//2,kernel_size//2] = 0.5
            if scale_kernel:
                kernel = kernel / kernel_size**2
            
            kernel = np.round(kernel, kernel_prec)

        self.kernel = kernel
        self.kernel_size = kernel_size
        self.parameters = kernel.flatten()
        self.activation = activation

    def step(self, image):
        image = image.copy() / 255
        """
        apply convolution
        args:
            image: image as 2D numpy array
        """
        if len(image.shape) != 2:
            raise ValueError("input must be a single-channel image")
        
        padded_image = utils.apply_periodic_padding(image, self.kernel_size)
        new_padded_image = cv2.filter2D(padded_image, -1, self.kernel)
        new_image = utils.apply_crop(new_padded_image, self.kernel_size)
        #new_image = cv2.filter2D(image, -1, self.kernel, borderType=cv2.BORDER_WRAP)
        
        if self.activation!='identity':
            if self.activation=='inv_gaussian':
                new_image = utils.inv_gaussian(new_image)
            elif self.activation=='abs':
                new_image = utils.abs(1.2*new_image)
            else:
                raise ValueError('activation function not recognised')
        return np.round(new_image * 255 / np.max(new_image))


# ************************************************ #
#                                                  #
#                     Rewarder                     #
#                                                  #
# ************************************************ #

class Rewarder:
    def __init__(self, desiderata, scales=[13, 7], sigma=4.):
        '''
        note: use odd numbers for scales
        '''
        # Initialise attributes
        self.desiderata = np.array(desiderata)
        self.reward = None
        self.observation = None
        self.is_end = None
        self.scales = scales
        self.sigma = sigma

    def update(self, image):
        """
        update the reward, observation, and is_end attributes based on input image
        args:
            image: image as 2D numpy array
        """
        if len(image.shape) != 2:
            raise ValueError("input must be a single-channel image")
        scale1 = min(image.shape) // self.scales[0]
        scale2 = min(image.shape) // self.scales[1]
        var1 = self.compute_variance(image, scale1)
        var2 = self.compute_variance(image, scale2)
        n_objects = self.count_objects(image)
        self.observation = np.array([var1, var2, n_objects])
        diff = self.desiderata - self.observation
        self.reward = 1/gmean(abs(diff))
        self.is_end = var1 + var2 == 0
        
    def compute_variance(self, image, scale=None):
        """
        compute the normalised variance of the image in local regions defined by the scale,
        using periodic boundary conditions
        """
        image = image.copy() / 255
        max_variance = 1/2.
        if scale==None:
            normalized_variance = np.var(image) / max_variance
        else:
            block_size = scale
            padded_image = utils.apply_periodic_padding(image, block_size)
            blurred_padded_image = cv2.blur(padded_image, (block_size, block_size))
            mean_image = utils.apply_crop(blurred_padded_image, block_size)
            #mean_image = cv2.blur(image, (block_size, block_size), borderType=cv2.BORDER_WRAP)
            squared_diff = (image - mean_image) ** 2
            local_variance = cv2.blur(squared_diff, (block_size, block_size))
            normalized_variance = np.mean(local_variance) / max_variance
        return np.clip(normalized_variance, 0, 1)
    
    def count_objects(self, image, connectivity=2):
        """
        compute number of distinct objects within the image
        CHECK: periodic boundary conditions?
        """
        edges = ski.feature.canny(image, sigma=self.sigma)
        _, n_objects = ski.measure.label(edges, return_num=True, connectivity=connectivity)
        return n_objects


# ************************************************ #
#                                                  #
#                   Environment                    #
#                                                  #
# ************************************************ #

class Environment(gym.Env):

    def __init__(self, image, bot, desiderata, rewarder, max_timesteps=1000, mid_timesteps=20):
        '''
        args:
            image:      2D array with a grayscale image
            bot:        object performing the convolution
            desiderata: 1D array with desired geom. properties
            rewarder:   object that evaluates the reward
        '''

        super(Environment, self).__init__()

        # Input
        self.image_t0 = image.copy()
        self.image = image
        self.bot = bot
        self.rewarder = rewarder
        self.parameters_t0 = bot.parameters.copy()
        self.parameters = bot.parameters
        self.desiderata = desiderata
        self.max_timesteps = max_timesteps
        self.mid_timesteps = mid_timesteps

        # Time
        self.timestep = 0
        
        # Action space
        self.action_space = gym.spaces.Box(low=-1, 
                                           high=1, 
                                           shape=(len(self.parameters),), 
                                           dtype=np.float32)
        
        # Observation space
        self.observation_space = gym.spaces.Box(low=-0, 
                                                high=1, 
                                                shape=(len(self.desiderata),), 
                                                dtype=np.float32)

    def reset(self, seed=0):

        self.timestep = 0
        self.image = self.image_t0.copy()
        self.parameters = self.parameters_t0.copy()
        return_obs = self.parameters.flatten()
        return_info = {'image': 'reset to t0',
                       'timestep': self.timestep}
        
        return return_obs, return_info

    def step(self):

        for _ in range(self.mid_timesteps):
            # Update
            self.image = self.bot.step(self.image)
            self.timestep += 1

        # Reward
        self.rewarder.update(self.image)
        reward = self.rewarder.reward

        # End episode flag
        terminated = bool(self.timestep > self.max_timesteps) 
        terminated = terminated | self.rewarder.is_end

        # Observation
        observation = self.rewarder.observation

        # Info
        info = {'desiderata': self.desiderata,
                'observation': observation,
                'reward': reward}

        return observation, reward, terminated, info