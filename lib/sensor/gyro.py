import numpy as np
import math
from scipy.stats import norm

class Gyro: ###noisesim_occlusion### 
    def __init__(self, init_data, direction_noise=math.pi/360*0.5, direction_bias_stddev=math.pi/360*0.1):
        
        self.direction_noise = direction_noise  
        self.direction_bias_phi = norm.rvs(scale=direction_bias_stddev) 
        self.direction_bias_theta = norm.rvs(scale=direction_bias_stddev) 
        self.direction_bias_psi = norm.rvs(scale=direction_bias_stddev)
        self.last_data = init_data
        
    def noise(self, euler):
        data = np.zeros(len(euler))
        for i in range(0, len(euler)):
          data[i] = norm.rvs(loc=euler[i], scale=self.direction_noise)
        return data
    
    def bias(self, euler): 
        data = np.zeros(len(euler))
        data[0] = euler[0] + self.direction_bias_phi
        data[1] = euler[1] + self.direction_bias_theta
        data[2] = euler[2] + self.direction_bias_psi
        return data
    
    def move_average(self, euler):
        data = np.zeros(len(euler))
        for i in range(0, len(euler)):
          data[i] = (self.last_data[i] + euler[i]) / 2
        return data
    
    def receive(self, euler):
        euler = self.bias(euler) # bias
        euler = self.noise(euler) # noise
        euler = self.move_average(euler) # move average
      
        self.last_data = euler
        return euler