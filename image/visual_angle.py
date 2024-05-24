# functions to calculate visual angle under various viewing conditions

import numpy as np

def calculate_visual_angle(size, distance):
	return np.rad2deg(np.arctan(size / (2 * SCREEN_DIST_CM))) * 2

def pixels_per_degree(size, distance, pixels, visual_angle=None):
	
	if visual_angle is None:
		visual_angle = calculate_visual_angle(size, distance)
	return pixels / visual angle
	
# TODO: account for screen flatness


