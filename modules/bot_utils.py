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
    def __init__(self, kernel=[], kernel_size=3, f_symm=True, h_symm=True, v_symm=True, fixed_center=True, 
                 kernel_center=0.5, activation='identity', kernel_prec=1, scale_kernel=False):
        '''
        note: use odd numbers for kernel_size
        kernel formats
            f symmetry:            h & v symmetry:        h symmetry:            v symmetry:
            +-----+-----+-----+    +-----+-----+-----+    +-----+-----+-----+    +-----------------+
            |  A  |  B  |  A  |    |  A  |  B  |  A  |    |     |     |     |    |        A        |
            +-----+-----+-----+    +-----+-----+-----+    |     |     |     |    |-----------------|
            |  B  |  C  |  B  |    |  C  |  D  |  C  |    |  A  |  B  |  A  |    |        B        |
            +-----+-----+-----+    +-----+-----+-----+    |     |     |     |    |-----------------|
            |  A  |  B  |  A  |    |  A  |  B  |  A  |    |     |     |     |    |        A        |
            +-----+-----+-----+    +-----+-----+-----+    +-----+-----+-----+    +-----------------+
        '''
        # Constraints (symmetries)
        self.f_symm = f_symm
        self.h_symm = h_symm
        self.v_symm = v_symm
        self.hv_symm = h_symm and v_symm
        self.fixed_center = fixed_center
        if fixed_center:
            self.kernel_center = kernel_center
        self.scale_kernel = scale_kernel
        self.kernel_prec = kernel_prec
        # Initialise sub-kernel dimensions
        if self.f_symm:
            self.kernel_na = (kernel_size//2)**2
            self.kernel_nb = kernel_size//2
            self.kernel_nc = 1            
        elif self.hv_sym:
            self.kernel_na = (kernel_size//2)**2
            self.kernel_nb = kernel_size//2
            self.kernel_nc = kernel_size//2
            self.kernel_nd = 1
        elif self.h_symm:
            self.kernel_na = (kernel_size//2)*kernel_size
            self.kernel_nb = kernel_size
        elif self.v_symm:
            self.kernel_na = (kernel_size//2)*kernel_size
            self.kernel_nb = kernel_size
        else:
            pass
        # Initialise kernel
        if len(kernel)==0:
            if kernel_size%2!=1:
                raise ValueError('kernel_size must be an odd number')
            self.kernel_size = kernel_size
            n_free_parameters = utils.n_free_parameters(f_symm=f_symm, 
                                                        hv_symm=self.hv_symm, 
                                                        h_symm=h_symm, 
                                                        v_symm=v_symm, 
                                                        kernel_size=kernel_size,
                                                        fixed_center=fixed_center)
            free_parameters = np.random.uniform(low=-1., high=1.0, size=n_free_parameters).astype(np.float32)
            self.update_kernel(free_parameters)
        else:
            self.kernel = kernel
            self.kernel_size = kernel.shape[0]
        self.parameters = self.kernel.flatten()
        self.activation_fn = self._get_activation_function(activation)

    def _get_activation_function(self, name):
        if name == 'identity':
            return lambda x: x
        elif name == 'inv_gaussian':
            return utils.inv_gaussian_clip
        elif name == 'abs':
            #return lambda x: utils.abs_clip(1.2 * x)
            return utils.abs_clip
        else:
            raise ValueError(f"Activation function '{name}' not recognized")

    def update_kernel(self, free_parameters):
        parameters = np.array(free_parameters).copy()
        if self.fixed_center:
            parameters = np.append(parameters, self.kernel_center)
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        ks = self.kernel_size
        if self.f_symm:
            na = self.kernel_na
            nb = self.kernel_nb
            nc = self.kernel_nc
            if len(parameters)!=na+nb+nc:
                raise ValueError("Parameters length not compatible with symmetry constraints")
            # A
            kernel_a = parameters[:na].reshape((ks//2, ks//2))
            kernel[:ks//2,:ks//2] = kernel_a
            kernel[1+ks//2:,:ks//2] = kernel_a
            kernel[:ks//2,1+ks//2:] = kernel_a
            kernel[1+ks//2:,1+ks//2:] = kernel_a
            # B
            kernel_b = parameters[na:na+nb]
            kernel[:ks//2,ks//2] = kernel_b
            kernel[1+ks//2:,ks//2] = kernel_b
            kernel[ks//2,:ks//2] = kernel_b
            kernel[ks//2,1+ks//2:] = kernel_b
            # C
            kernel_c = parameters[na+nb:na+nb+nc]
            kernel[ks//2,ks//2] = kernel_c         
        elif self.hv_sym:
            na = self.kernel_na
            nb = self.kernel_nb
            nc = self.kernel_nc
            nd = self.kernel_nd
            if len(parameters)!=na+nb+nc+nd:
                raise ValueError("Parameters length not compatible with symmetry constraints")
            # A
            kernel_a = parameters[:na].reshape((ks//2, ks//2))
            kernel[:ks//2,:ks//2] = kernel_a
            kernel[1+ks//2:,:ks//2] = kernel_a
            kernel[:ks//2,1+ks//2:] = kernel_a
            kernel[1+ks//2:,1+ks//2:] = kernel_a
            # B
            kernel_b = parameters[na:na+nb]
            kernel[:ks//2,ks//2] = kernel_b
            kernel[1+ks//2:,ks//2] = kernel_b
            # C
            kernel_c = parameters[na+nb:na+nb+nc]
            kernel[ks//2,:ks//2] = kernel_c
            kernel[ks//2,1+ks//2:] = kernel_c
            # D
            kernel_d = parameters[na+nb+nc:na+nb+nc+nd]
            kernel[ks//2,ks//2] = kernel_d
        elif self.h_symm:
            if self.fixed_center:
                raise ValueError("Fixed center not allowed for horizontal symmetry only")
            na = self.kernel_na
            nb = self.kernel_nb
            if len(parameters)!=na+nb:
                raise ValueError("Parameters length not compatible with symmetry constraints")
            # A
            kernel_a = parameters[:na].reshape((ks, ks//2))
            kernel[:,:ks//2] = kernel_a
            kernel[:,1+ks//2:] = kernel_a
            # B
            kernel_b = parameters[na:na+nb]
            kernel[:,ks//2] = kernel_b
        elif self.v_symm:
            if self.fixed_center:
                raise ValueError("Fixed center not allowed for vertical symmetry only")
            na = self.kernel_na
            nb = self.kernel_nb
            if len(parameters)!=na+nb:
                raise ValueError("Parameters length not compatible with symmetry constraints")
            # A
            kernel_a = parameters[:na].reshape((ks//2, ks))
            kernel[:ks//2,:] = kernel_a
            kernel[1+ks//2:,:] = kernel_a
            # B
            kernel_b = parameters[na:na+nb]
            kernel[ks//2,:] = kernel_b
        else:
            if self.fixed_center:
                raise ValueError("Fixed center not allowed without symmetries")
            if len(parameters)!=ks**2:
                raise ValueError("Parameters length not compatible with symmetry constraints")
            self.kernel = parameters.reshape((ks, ks))
        if self.scale_kernel:
            kernel = kernel / ks**2
        kernel = np.round(kernel, self.kernel_prec)
        self.kernel = kernel
        self.parameters = kernel.flatten()

    def step(self, state):
        """
        apply convolution
        args:
            state: 2D numpy array
        """
        if len(state.shape) != 2:
            raise ValueError("input must be a 2D array")
        # Coonvolution
        padded_state = utils.apply_periodic_padding(state, self.kernel_size)
        new_padded_state = cv2.filter2D(padded_state, -1, self.kernel)
        new_state = utils.apply_crop(new_padded_state, self.kernel_size)        
        # Activation
        new_state = self.activation_fn(new_state)
        # Clip
        new_state = np.clip(new_state, -1, 1)
        return new_state


# ************************************************ #
#                                                  #
#                     Rewarder                     #
#                                                  #
# ************************************************ #

class Rewarder:
    def __init__(self, desiderata, scales=[40, 10], sigma=4., pixel_lbound=-1, pixel_ubound=1):
        '''
        note: use odd numbers for scales
        '''
        # Initialise attributes
        self.desiderata = np.array(desiderata)
        self.reward = None
        self.observation = None
        self.variance = None
        self.scales = scales
        self.sigma = sigma
        self.max_variance = (1/4.) * (pixel_ubound-pixel_lbound)**2

    def update(self, state):
        """
        update the reward, observation, and is_end attributes based on input state
        args:
            state: 2D numpy array
        """
        if len(state.shape) != 2:
            raise ValueError("input must be a 2D array")
        self.variance = self.compute_variance(state)
        #self.observation = self.compute_obs_1(state)
        #self.observation = self.compute_obs_2(state)
        #self.observation = self.compute_obs_3(state)
        #
        #n_objects = self.count_objects(state)
        #n_objects_transformed = np.tanh(n_objects/1000)
        #self.observation = np.array([n_objects_transformed], dtype=np.float32)
        #
        variance = self.compute_variance(state)
        self.observation = np.array([variance], dtype=np.float32)
        diff = self.desiderata - self.observation + 1e-10
        rexp = 1./gmean(abs(diff))
        self.reward = rexp #1-np.exp(-rexp)

    def compute_obs_1(self, state):
        '''FIX: compute_variance has default scale=None'''
        scale1 = min(state.shape) // self.scales[0]
        var1 = self.compute_variance(state, scale1)
        observation = np.array([var1], dtype=np.float32)
        return observation

    def compute_obs_2(self, state):
        '''FIX: compute_variance has default scale=None'''
        scale1 = min(state.shape) // self.scales[0]
        scale2 = min(state.shape) // self.scales[1]
        var1 = self.compute_variance(state, scale1)
        var2 = self.compute_variance(state, scale2)
        observation = np.array([var1, var2], dtype=np.float32)
        return observation

    def compute_obs_3(self, state):
        '''FIX: compute_variance has default scale=None'''
        scale1 = min(state.shape) // self.scales[0]
        scale2 = min(state.shape) // self.scales[1]
        var1 = self.compute_variance(state, scale1)
        var2 = self.compute_variance(state, scale2)
        n_objects = self.count_objects(state)
        n_objects_transformed = np.tanh(n_objects/1000)
        observation = np.array([var1, var2, n_objects_transformed], dtype=np.float32)
        return observation
        
    def compute_variance(self, state, scale=None):
        """
        compute the normalised variance of the state in local regions defined by the scale,
        using periodic boundary conditions
        """
        if scale==None:
            normalized_variance = np.var(state) / self.max_variance
        else:
            block_size = scale
            padded_state = utils.apply_periodic_padding(state, block_size)
            blurred_padded_state = cv2.blur(padded_state, (block_size, block_size))
            mean_state = utils.apply_crop(blurred_padded_state, block_size)
            #mean_state = cv2.blur(state, (block_size, block_size), borderType=cv2.BORDER_WRAP)
            squared_diff = (state - mean_state) ** 2
            local_variance = cv2.blur(squared_diff, (block_size, block_size))
            normalized_variance = np.mean(local_variance) / self.max_variance
        return normalized_variance
    
    def count_objects(self, state, connectivity=2):
        """
        compute number of distinct objects within the state
        CHECK: periodic boundary conditions?
        """
        edges = ski.feature.canny(state, sigma=self.sigma)
        _, n_objects = ski.measure.label(edges, return_num=True, connectivity=connectivity)
        return n_objects


# ************************************************ #
#                                                  #
#                   Environment                    #
#                                                  #
# ************************************************ #

class Environment(gym.Env):

    def __init__(self, bot, n_free_parameters, observation_boundaries, rewarder, max_timesteps=1000, 
                 mid_timesteps=20, edge_len=400, var_th=1e-4, pixel_lbound=-1, pixel_ubound=1):
        '''
        args:
            n_free_parameters:  dimension action space (scalar)
            bot:                object performing the convolution
            desiderata:         1D array with desired geom. properties
            rewarder:           object that evaluates the reward
        '''
        super(Environment, self).__init__()
        # State bounds
        self.pixel_lbound = pixel_lbound
        self.pixel_ubound = pixel_ubound
        # Input
        self.var_th = var_th
        self.bot = bot
        self.n_free_parameters = n_free_parameters
        self.observation_boundaries = observation_boundaries
        self.rewarder = rewarder
        self.max_timesteps = max_timesteps
        self.mid_timesteps = mid_timesteps
        self.edge_len = edge_len
        self.image_t0 = utils.random_image(size=(edge_len,edge_len), 
                                           low=self.pixel_lbound, 
                                           high=self.pixel_ubound)
        self.image = self.image_t0.copy()
        # Initialize
        self.timestep = 0
        self.rewarder.update(utils.image_to_state(self.image, 
                                                  low=self.pixel_lbound,
                                                  high=self.pixel_ubound))
        # Action space
        self.action_space = gym.spaces.Box(low=-1, 
                                           high=1, 
                                           shape=(self.n_free_parameters,), 
                                           dtype=np.float32)
        # Observation space
        low = observation_boundaries[:, 0].astype(np.float32)
        high = observation_boundaries[:, 1].astype(np.float32)

        self.observation_space = gym.spaces.Box(low=low, 
                                                high=high, 
                                                dtype=np.float32)

    def reset(self, seed=0, desiderata=None):
        self.timestep = 0
        self.image_t0 = utils.random_image(size=(self.edge_len,self.edge_len), 
                                           low=self.pixel_lbound, 
                                           high=self.pixel_ubound)
        self.image = self.image_t0.copy()
        self.rewarder.update(utils.image_to_state(self.image, 
                                                  low=self.pixel_lbound, 
                                                  high=self.pixel_ubound))
        lows = self.observation_boundaries[:, 0]
        highs = self.observation_boundaries[:, 1]
        if not desiderata:
            self.desiderata = np.random.uniform(low=lows, high=highs).astype(np.float32)
        else:
            self.desiderata = desiderata
        self.rewarder.desiderata = self.desiderata
        return_obs = self.desiderata
        return_info = {'image': 'reset to t0',
                       'timestep': self.timestep}
        return return_obs, return_info

    def step(self, action):
        self.bot.update_kernel(action)
        for _ in range(self.mid_timesteps):
            # Update
            state = utils.image_to_state(self.image, 
                                         low=self.pixel_lbound, 
                                         high=self.pixel_ubound)
            state = self.bot.step(state)
            self.image = utils.state_to_image(state, 
                                              low=self.pixel_lbound, 
                                              high=self.pixel_ubound)
            self.timestep += 1
        # Reward
        self.rewarder.update(state)
        reward = self.rewarder.reward
        # End episode flag
        variance = self.rewarder.compute_variance(state)
        truncated = False
        if variance < self.var_th:
            truncated = True
            reward = 0.
        terminated = bool(self.timestep >= self.max_timesteps) 
        terminated = bool(terminated | truncated)
        # Observation
        '''The learner onserves the desiderata and acts accordingly'''
        observation = self.desiderata
        # Info
        info = {'desiderata': self.desiderata,
                'observation': self.rewarder.observation,
                'reward': reward}
        return observation, reward, terminated, truncated, info
    
    def step_eval(self):
        for _ in range(self.mid_timesteps):
            state = utils.image_to_state(self.image, 
                                         low=self.pixel_lbound, 
                                         high=self.pixel_ubound)
            state = self.bot.step(state)
            self.image = utils.state_to_image(state, 
                                              low=self.pixel_lbound, 
                                              high=self.pixel_ubound)
            self.timestep += 1
        self.rewarder.update(state)
        reward = self.rewarder.reward
        # End episode flag
        variance = self.rewarder.compute_variance(state)
        truncated = False
        if variance < self.var_th:
            truncated = True
            reward = 0.
        terminated = bool(self.timestep >= self.max_timesteps) 
        terminated = bool(terminated | truncated)
        observation = self.rewarder.observation
        info = {'desiderata': self.desiderata,
                'observation': observation,
                'reward': reward}
        return terminated, info