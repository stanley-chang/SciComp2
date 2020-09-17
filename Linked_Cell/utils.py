import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, patches
import time
from itertools import product


def lj_potential(distance, c1=1e-15, c2=1e-5):
    return   (c1 / distance**12) - (c2 / distance**6)

def get_successor_neighbor_delta_coordinate(a=1):
    """Returns neighbor_delta_coordinate
    
    Parameters
    ---------
    a: int
        Variable linked-cell parameter
    """
    
    neighbor_delta_coordinate = []
    ############# Task 1.1 begins ##################
     

    # create y indeces (e.g for a=2: [1 1 2 2])
    y = np.repeat(np.array([range(a)])+1, a, axis=1)
    
    # create x indeces (e.g. for a=2: [1 2 1 2])
    x = np.reshape(np.repeat(np.array([range(a)])+1, a, axis=0), (1, a*a))

    # make joint numpy array of x, y indeces: [[x0, y0], [x1, y1]...[xn, yn]]
    first_quad_cells = np.concatenate((np.transpose(x), np.transpose(y)), axis=1)

    # coordinates of bottom left corner of each cell in np.array
    target_corners_coords = first_quad_cells - 0.5  

    # calc the distances of bottom left corners of cells in 1st quadrant
    # to the top right corner of center cell
    # reference corner's coords of center cell are (0.5, 0.5)
    # subtracting 0.5 from np.array goes elementwise, like... 
    # ...doing --> sqrt( (target_corners_coords[i][0] - 0.5)**2 + (target_corners_coords[i][1] - 0.5)**2 )
    distances = np.sqrt(np.sum((target_corners_coords - 0.5)**2, axis=1))

    # get the cells whose distance is less than r_c ( r_c=a in this reference space )
    neighbor_delta_coordinate = first_quad_cells[np.where(distances<a)] 

    ############ Task 1.1 ends #####################
    return neighbor_delta_coordinate

def plot_all_cells(ax, list_cells, edgecolor='r',domain=1):
    for c in list_cells:
        c.plot_cell(ax, edgecolor=edgecolor)
    ax.tick_params(axis='both',labelsize=0, length = 0)
    plt.xlim(left=0, right=domain)
    plt.ylim(bottom=0, top=domain)
    ax.set_aspect('equal', adjustable='box')
    
def get_mean_relative_error(direct_potential, linked_cell_potential):
    return np.mean(np.abs((direct_potential - linked_cell_potential) / direct_potential))
