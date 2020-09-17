import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
import time
from itertools import product
import utils
from abc import ABCMeta, abstractmethod


class Cell(object, metaclass=ABCMeta):
    
    def __init__(self, lx, ly, r_c, cell_index, neighbor_delta_coordinate, a=1, domain=1):
        """Constructor
        lx : float
            Lower x coordinate
        ly : float
            Lower y coordinate
        r_c : float
            Cut-off radius
        cell_index : int
            Index of cell in list_cells
        neighbor_delta_coordinate : list
            List of  List of numpy array(of size 2), where each numpy array contains the difference between the 
            2d coordinates of current cell and one of neighbor cells in upper right quadrant
        a : int (default value 1)
            Variable lined-cell parameter
        domain : float (default value 1.0)
            Size of domain
        """
        self.side_length = r_c / a
        self.a = a
        self.cell_center = np.array([lx + 0.5 * self.side_length, ly + 0.5 * self.side_length])
        self.cell_index = cell_index
        self.neighbor_cell_index = []
        self.create_neighbor_cell_index(neighbor_delta_coordinate, domain=domain)

    # HELPER FUNCTION
    def finish_relative_box_by_symmetry(self, neighbor_delta_coordinate):
        '''
            calculates the central top and right cells alligned with the center one

            Makes use of symmetry to create a final box of all the neighbours with relative 2d indeces

        '''
        relative_indeces = []  # list to hold all the relative neighbours calculated by symmetry
        
        sec_quad = np.copy(neighbor_delta_coordinate)  # create a copy of neighbor_delta_coordinate
        third_quad = np.copy(neighbor_delta_coordinate)  # create a copy of neighbor_delta_coordinate
        fourth_quad = np.copy(neighbor_delta_coordinate)  # create a copy of neighbor_delta_coordinate
 
        third_quad *= -1  # calculate indeces in third quadrant
        fourth_quad[:, 1] *= -1  # calculate indeces in fourth quadrant
        sec_quad[:, 0] *= -1  # calculate indeces in second quadrant
        
        cross_right = [np.array([0, i]) for i in range(1, self.a+1)]  # positive x direction
        cross_left = np.copy(cross_right)  # copy for negative x direction calculation
        cross_top = [np.array([i, 0]) for i in range(1, self.a+1)]  # positive y direction
        cross_bottom = np.copy(cross_top)  # copy for negative y direction calculation

        cross_left[:, 1] *= -1  # calc indeces of negative x direction
        cross_bottom[:, 0] *= -1  # calc indeces of negative y direction

        # join all cells
        relative_indeces = np.concatenate((third_quad, cross_bottom, fourth_quad, cross_left, \
            cross_right, sec_quad, cross_top, neighbor_delta_coordinate), axis=0)

        return relative_indeces

    # HELPER FUNCTION
    @staticmethod
    def check_column_bounds(center_idx, num_cells_row, cell_x, idx):
        '''
            Checks if calculated index is inside the top and bottom boundaries
            of the current column

            center_idx      :   index of center cell
            num_cells_row   :   number of cells in x direction
            cell_x          :   x index of cell (in reference to center index)
            idx             :   calculated index of current neighboring cell 

        '''
        cell_col = center_idx % num_cells_row + cell_x  # find cell's column
        col_low = cell_col  # find lower bound for this column
        col_up = cell_col + (num_cells_row-1)*num_cells_row  # find upper bound for this column
        if idx>=col_low and idx<=col_up:  # check if index is within
            return True
        return False

    # HELPER FUNCTION
    @staticmethod
    def check_line_bounds(center_idx, num_cells_row, cell_y, idx):
        '''
            Checks if calculated index is inside the most right and most left
            boundaries of the current line

            center_idx      :   index of center cell
            num_cells_row   :   number of cells in x direction
            cell_y          :   y index of cell (in reference to center index)
            idx             :   calculated index of current neighboring cell

        '''
        cell_line = center_idx // num_cells_row + cell_y  # find cell's line
        line_low = cell_line * num_cells_row  # find lower bound for this line
        line_up = line_low + num_cells_row - 1  # find upper bound for this line
        if idx>=line_low and idx<=line_up:  # check if index is within
            return True
        return False

    def create_neighbor_cell_index(self, neighbor_delta_coordinate, domain=1):
        """Creates neighbor cell index for the current cell
        Parameters
        ----------
        neighbor_delta_coordinate: list
            Relative 2d index of all neighbor interaction cells in first quadrant
        domain: float (Optional value 1.0)
            Size of domain
        """
        self.neighbor_cell_index = []
        ############## Task 1.2 begins ##################

        # use of symmetry to create a relative window around the center cell
        relative_indeces = self.finish_relative_box_by_symmetry(neighbor_delta_coordinate)

        center_idx = self.cell_index  # index of current cell
        num_cells_row = int(domain / self.side_length)  # number of cells in one row
        
        for cell in relative_indeces:  # loop for every relative neighbour
           
            # current calculated index of neighbor
            idx = int(center_idx + cell[1] * num_cells_row + cell[0])  # calc index
            
            # check if calculated index is inside acceptable line boundaries
            line_flag = Cell.check_line_bounds(center_idx, num_cells_row, cell[1], idx)  # bool: true if idx inside bounds

            # check if calculated index is inside acceptable column boundaries
            col_flag = Cell.check_column_bounds(center_idx, num_cells_row, cell[0], idx)  # bool: true if idx inside bounds

            # if calculated index is inside boundaries then append 
            # to list as neigbour
            if line_flag and col_flag:
                self.neighbor_cell_index.append(idx)

        self.neighbor_cell_index = sorted(self.neighbor_cell_index)

        ############## Task 1.2 ends ##################
        
    def __str__(self):
        return 'Object of type cell with center {}'.format(self.cell_center)
    
    @abstractmethod
    def add_particle(self, particle_index):
        return
        
    def add_neighbor_cell(self, cell_index):
        self.neighbor_cell_index.append(cell_index)
    
    @abstractmethod
    def delete_all_particles(self):
        return
          
    def plot_cell(self, ax, linewidth=1, edgecolor='r', facecolor='none'):
        lx = self.cell_center[0] - self.side_length/2
        ly = self.cell_center[1] - self.side_length/2
        rect = patches.Rectangle((lx, ly), self.side_length, self.side_length, linewidth=linewidth,
                                 edgecolor=edgecolor, facecolor=facecolor)
        ax.add_patch(rect)
   
    @abstractmethod
    def plot_particles(self, list_particles, marker='o', color='r', s=2):
        return
            
    def plot_neighbor_cells(self, ax, list_cells, linewidth=1, edgecolor='r', facecolor='none'):
        for idx in self.neighbor_cell_index:
            list_cells[idx].plot_cell(ax, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor)
            
    @abstractmethod
    def plot_neighbor_cell_particles(self, list_cells, list_particles, marker='o', color='r', s=2):
        return
    
    def distance(self, other):
        return np.linalg.norm(self.cell_center - other.cell_center, 2)
    
    def plot_rc(self, ax, rc):
        circle = patches.Circle((self.cell_center[0], self.cell_center[1]), rc)
