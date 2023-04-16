import numpy as np
import math
from scipy.stats import norm

class StateSensor: ###noisesim_occlusion### 
    def __init__(self, init_data, distance_noise_rate=0.001, distance_bias_rate_stddev=0.001):
        
        self.distance_noise_rate = distance_noise_rate
        self.distance_bias_rate_std_0 = norm.rvs(scale=distance_bias_rate_stddev)
        self.distance_bias_rate_std_1 = norm.rvs(scale=distance_bias_rate_stddev)
        self.distance_bias_rate_std_2 = norm.rvs(scale=distance_bias_rate_stddev)
        self.distance_bias_rate_std_3 = norm.rvs(scale=distance_bias_rate_stddev)
        self.last_data = init_data
        
    def noise(self, q):
        data = np.zeros(len(q))
        for i in range(0, len(q)):
          if q[i] < 0:
            data[i] = -norm.rvs(loc=-q[i], scale=-q[i]*self.distance_noise_rate)
          else:
            data[i] = norm.rvs(loc=q[i], scale=q[i]*self.distance_noise_rate)
        return data
    
    def bias(self, q): 
        data = np.zeros(len(q))
        data[0] = q[0] + q[0]*self.distance_bias_rate_std_0
        data[1] = q[1] + q[1]*self.distance_bias_rate_std_1
        data[2] = q[2] + q[2]*self.distance_bias_rate_std_2
        data[3] = q[3] + q[3]*self.distance_bias_rate_std_3
        return data
    
    def move_average(self, euler):
        data = np.zeros(len(euler))
        for i in range(0, len(euler)):
          data[i] = (self.last_data[i] + euler[i]) / 2
        return data
    
    def receive(self, q):
        q = self.bias(q) # bias
        q = self.noise(q) # noise
        q = self.move_average(q) # move average
      
        self.last_data = q
        return q