import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None
        
    def draw(self):
        total_blocks = self.resolution // (2 * self.tile_size)
        
        black_block = np.zeros((self.tile_size, self.tile_size))
        white_block = np.ones((self.tile_size, self.tile_size))
        
        row1 = np.concatenate((black_block, white_block), axis=1)
        row2 = np.concatenate((white_block, black_block), axis=1)
        
        merged_base = np.concatenate((row1, row2), axis=0)
        
        self.output = np.tile(merged_base, (total_blocks, total_blocks))
        
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.title("Checker")
        plt.show()
        

class Circle:
    def __init__(self, resolution: int, radius: int, position: tuple):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None
    
    def draw(self):
        c_x,c_y = self.position
        x,y = np.arange(self.resolution), np.arange(self.resolution)
        X, Y = np.meshgrid(x, y)
        mask = (c_x - X)**2 + (c_y - Y)**2 <= self.radius**2
        self.output = mask.astype(np.uint8)
        return self.output.copy()
    
    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.axis('off')
        plt.title("Circle")
        plt.show()
        
class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        line = np.linspace(0, 1, self.resolution)
        self.output = np.zeros((self.resolution, self.resolution, 3))
        self.output[:, :, 0] = line      # Red
        
        self.output[:, :, 1] = line[:, None]     # Green 
        
        self.output[:, :, 2] = 1 - line   # Blue
        
        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.title("Spectrum")
        plt.show()
        
