from placedb import PlaceDB
from canvas import Canvas
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import os
from ant_colony import AntColony  # Import AntColony class

def greedy_place(chip:Canvas, module_list):
    for i, module_name in enumerate(module_list):
        overlap = chip.get_overlap(module_name)
        wl_icre = chip.get_hpwl_increment(module_name)
        wl_icre[overlap > overlap.min()] = np.inf
        min_indices = np.where(wl_icre == np.min(wl_icre))
        idx = random.randint(0, len(min_indices[0])-1)
        
        # Ensure indices are integers
        x = int(min_indices[0][idx])
        y = int(min_indices[1][idx])
        chip.place_module(module_name, x, y)

    chip.overlap = np.maximum(0, chip.overlap - 1)
    chip.hpwl *= chip.ratio
    return chip

    
def aco_refinement(chip: Canvas, module_list):
    aco = AntColony(chip, module_list)  # Initialize the ACO with the given chip and modules
    aco.place_modules()  # This will run ACO to refine the placement
    return chip

def hybrid_place(chip: Canvas, module_list):
    # Step 1: Perform greedy placement
    chip = greedy_place(chip, module_list)

    # Step 2: Refine placement using ACO
    chip = aco_refinement(chip, module_list)

    return chip

# def rand_place(chip:Canvas, module_list):
#     for i, module_name in enumerate(module_list):
#         overlap = chip.get_overlap(module_name)
#         min_indices = np.where(overlap == np.min(overlap))
#         idx = random.randint(0, len(min_indices[0])-1)
#         chip.place_module(module_name, min_indices[0][idx], min_indices[1][idx])
#     chip.overlap = np.maximum(0, chip.overlap - 1)
#     chip.hpwl *= chip.ratio
#     return chip

def place(chip:Canvas, module_list, position):
    for module_name in module_list:
        x = position[module_name]['x']
        y = position[module_name]['y']
        chip.place_module(module_name, x, y)
    return chip