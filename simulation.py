"""

Prototype simulation of different path planning algorithms
on the Kieler Foerde.

"""

import math
import time
import datetime
import copy
import random
from turtle import color
from statistics import mean

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from AStar.a_star import AStarPlanner
from Dijkstra.dijkstra import Dijkstra
from DepthFirstSearch.depth_first_search import DepthFirstSearchPlanner
from BreadthFirstSearch.breadth_first_search import BreadthFirstSearchPlanner
from RRT.rrt import RRT
import DynamicWindowApproach.dynamic_window_approach as DWA
from DStar.dstar import Dstar, Map
from RVO.RVO import RVO_update, compute_V_des
from RVO.vis import visualize_traj_dynamic
from RRT.rrt_with_pathsmoothing import path_smoothing
from StarSmoothing.star_path_smoothing import smoothing
from CPA.cpa_projection import cpa_projection
# from DStarLite.d_star_lite import DStarLite, Node


"""
Parameters and options
"""

# Choose the algorithm for the simulation
algorithm = 'A*'  # A*, Dijkstra, DFS, BFS, D*, RRT, DWA or RVO

# Activate hybrid mode (A* and RVO)
hybrid_mode = True

# Choose the scenario
scenario = 'difficult'  # test, overtaking, head-on, crossing, difficult

# Flag for animation option
show_animation = False  # the simulation is much faster without aninmations

# Default position of ASV-start
sx = 96.0
sy = 234.0

# Default position of ASV-goal
gx = 266.0
gy = 288.0

# Define the axises
axis_measure = [0, 300, 0, 300]  # x-axis (start, finish), y-axis (start, finish)

# Granularity of the calcuations (Best: A* 1.5, RRT 0.5, DWA 4.0, D* 1.5, RVO 3.0; fixed to those in hybrid mode)
grid_size = 3.0

# Approximate measurements of the ASV
asv_radius = 7.0  # about 100 meters safety zone
extend_security_distance = 14.0  # extended zone for switching between local and global in hybrid mode

# Number of runs for pathfinding
runs = 10  # for a single activated algorithm
hybrid_runs = 100  # only for hybrid mode; has to be a multiple of 10

# Projection of ships option
projection = True
# Factors for the probability cloud of the projected ships (not relevant for dynamic/local algorithms DWA and RVO)
probability_factor = 50.0  # add one layer for every x-th point
probability_distance = 2.0  # distance between these layers

# COLREG option (activate with overtaking and crossing scenarios)
colregs = False

# RRT Best-of option
rrt_bestof = True
rrt_bestof_runs = 10
# Maximal steering angle for RRT
steering_angle = 360.0

# Flag for preview plot of the other vessels courses in the chosen scenario
# (This is only for plotting them; no computations after that!)
course_preview = False

# Flag for path smoothing
smoothing_of_path = False  # right now only optimized for RRT, A* and D* (others might work though)
# Number of iterations smoothing is performed in case of the *-algorithms
smoothing_iterations = 5
# Granularity of *-algorithm smoothing (the lower the finer)
granular_size = 5  
# Granularity of RRT smoothing
maxIter = 1000  



"""
Functions
"""

# Build obstacles.
def build_obstacle_line(x_start, y_start, x_end, y_end):
    x_result, y_result = [], []
    slope = 0
    # Avoid division by zero
    if (x_start - x_end) != 0:
        slope = abs(y_start - y_end) / abs(x_start - x_end)
    
    # If x stays the same just add all y-values
    else:
        for i in range(abs(y_start - y_end) + 1):
            x_result.append(x_start)
            y_result.append(min(y_start, y_end) + i)

        return x_result, y_result

    # If y stays the same just add all x-values
    if (y_start - y_end) == 0:
        for i in range(abs(x_start - x_end) + 1):
            x_result.append(min(x_start, x_end) + i)
            y_result.append(y_start)

        return x_result, y_result

    # Calculate all the coordinates for the marker points otherwise
    for i in range(abs(x_start - x_end) + 1):
        if (x_start - x_end) >= 0 and (y_start - y_end) >= 0:
            x_temp = x_end + i
            for j in range(math.ceil(slope)):
                if round(y_end + i * slope + j) <= y_start:
                    x_result.append(x_temp)
                    y_result.append(round(y_end + i * slope + j))
        if (x_start - x_end) >= 0 and (y_start - y_end) < 0:
            x_temp = x_end + i
            for j in range(math.ceil(slope)):
                if (round(y_end - i * slope - j) >= y_start):
                    x_result.append(x_temp)
                    y_result.append(round(y_end - i * slope - j))
        if (x_start - x_end) < 0 and (y_start - y_end) >= 0:
            x_temp = x_start + i
            for j in range(math.ceil(slope)):
                if (round(y_start - i * slope - j) >= y_end):
                    x_result.append(x_temp)
                    y_result.append(round(y_start - i * slope - j))
        if (x_start - x_end) < 0 and (y_start - y_end) < 0:
            x_temp = x_start + i
            for j in range(math.ceil(slope)):
                if round(y_start + i * slope + j) <= y_end:
                    x_result.append(x_temp)
                    y_result.append(round(y_start + i * slope + j))

    return x_result, y_result

# Build boundaries out of individual obstacles.
def build_boundaries(xy_list):
    x_result, y_result = [], []
    x_start = y_start = x_end = y_end = None
    for i in range(len(xy_list) - 1):
        x_start, y_start = xy_list[i][0], xy_list[i][1]
        x_end, y_end = xy_list[i + 1][0], xy_list[i + 1][1]
        x_temp_result, y_temp_result = build_obstacle_line(x_start, y_start, x_end, y_end)
        x_result.extend(x_temp_result)
        y_result.extend(y_temp_result)

    return x_result, y_result

# Calculate the coordinates of a vessels hull, given it's centre point, size and direction.
def position_ship(centre, length, direction):
    # Calculate unit vector from direction
    x_temp = (1 / math.sqrt(direction[0] ** 2 + direction[1] ** 2)) * direction[0]
    y_temp = (1 / math.sqrt(direction[0] ** 2 + direction[1] ** 2)) * direction[1]
    unit_vector = (x_temp, y_temp)
    # Calculate the start and end-coordinates for the ship (as obstacle)
    x_bow = round(centre[0] + 0.5 * length * unit_vector[0])
    y_bow = round(centre[1] + 0.5 * length * unit_vector[1])
    x_stern = round(centre[0] - 0.5 * length * unit_vector[0])
    y_stern = round(centre[1] - 0.5 * length * unit_vector[1])

    return x_bow, y_bow, x_stern, y_stern

# Enrich the ship trajectory with more points (so, higher granularity).
def enrich_ship_trajectory(number_of_points, ship_trajectory):
    if number_of_points % 10 != 0:
        return []
    ship_traj_temp = []
    for i in range(len(ship_trajectory)):
        for j in range(int(number_of_points / 10)):
            ship_traj_temp.append([ship_trajectory[i][0] / (number_of_points / 10.0), ship_trajectory[i][1] / (number_of_points / 10.0)])
    return ship_traj_temp

# Adapt the ships and their trajectories to higher granularity of the hybrid model.
def adapt_ships_to_hybrid(number_of_points, ships, ship_trajectories):
    if number_of_points % 10 != 0:
        return []
    ships_temp = copy.deepcopy(ships)
    ship_trajectories_temp = []
    for i in range(len(ships_temp)):
        ship_trajectories_temp.append(enrich_ship_trajectory(number_of_points, ship_trajectories[i]))
        ships_temp[i][2] = copy.deepcopy(ship_trajectories_temp[-1][0])
    return ships_temp, ship_trajectories_temp



"""
Obstacle placement
"""

# Set obstacle positions
ox, oy = [], []
# Configure general borders
for i in range(axis_measure[0], axis_measure[1] + 1):
    ox.append(i)
    oy.append(0.0)
for i in range(axis_measure[2], axis_measure[3]):
    ox.append(0.0)
    oy.append(i)

# Limit navigable terrain to Kieler Foerde area (in foerde_cutout-image)
kieler_foerde_cutout = [(130, 300), (116, 288), (90, 234), (90, 100), (50, 63),
    (92, 35), (92, 10), (96, 0), (175, 0), (175, 50), (196, 76), (187, 83), 
    (185, 103), (205, 140), (197, 160), (200, 170), (275, 203), (283, 223),
    (273, 257), (269, 271), (275, 289), (273, 300), (130, 300)]
x_boundaries, y_boundaries = build_boundaries(xy_list=kieler_foerde_cutout)
# Add missing points (so that the boundaries are absolutely gapless)
x_boundaries + [198, 200, 202, 204, 276, 278, 280, 282, 270, 270, 272, 274, 275, 93, 95]
y_boundaries + [157, 151, 147, 141, 207, 211, 217, 221, 267, 265, 259, 293, 291, 7, 1]
ox.extend(x_boundaries)
oy.extend(y_boundaries)

# Build special boundaries for RRT (to limit the random possibilities)
eastern_shore, western_shore = [], []
foerde_eastern_shore = [(130, 300), (116, 288), (90, 234), (90, 100), (50, 63), 
    (92, 35), (92, 10), (96, 0)]
foerde_western_shore = [(175, 0), (175, 50), (196, 76), (187, 83), 
    (185, 103), (205, 140), (197, 160), (200, 170), (275, 203), (283, 223),
    (273, 257), (269, 271), (275, 289), (273, 300)]
x_eastern_shore, y_eastern_shore = build_boundaries(xy_list=foerde_eastern_shore)
# Add missing points (so that the eastern shore boundary is gapless)
x_eastern_shore = x_eastern_shore + [93, 95]
y_eastern_shore = y_eastern_shore + [7, 1]
x_western_shore, y_western_shore = build_boundaries(xy_list=foerde_western_shore)
# Add missing points (so that the western shore boundary is gapless)
x_western_shore = x_western_shore + [198, 200, 202, 204, 276, 278, 280, 282, 270, 270, 272, 274, 275]
y_western_shore = y_western_shore + [157, 151, 147, 141, 207, 211, 217, 221, 267, 265, 259, 293, 291]
for j in range(len(x_eastern_shore)):
    eastern_shore.append((x_eastern_shore[j], y_eastern_shore[j]))
for j in range(len(x_western_shore)):
    western_shore.append((x_western_shore[j], y_western_shore[j]))
foerde_shores = (eastern_shore, western_shore)

# Scenarios
ship1, ship2, ship3 = [], [], []
ship1_traj, ship2_traj, ship3_traj = [], [], []
# Test scenario
if scenario == 'test':
    # Position of ASV-start
    sx = 98.0
    sy = 234.0
    # Position of ASV-goal
    gx = 266.0
    gy = 288.0
    # Ships 
    ship1 = [[200, 270], 8, [-2, -4.5], 0.36]  # centre point, length, direction, speed
    ship2 = [[185, 244], 4, [-1.5, -3.375], 0.27]  # centre point, length, direction, speed
    ship3 = [[221, 281], 10, [-1, -2.25], 0.18]  # centre point, length, direction, speed
    # Ships trajectories
    ship1_traj = [[-2, -4.5], [-2, -4.5], [-2, -4.5], [-2, -4.5], [-2, -4.5], 
                [-2, -4.5], [-2, -4.5], [-2, -4.5], [-2, -4.5], [-2, -4.5]]
    ship2_traj = [[-1.5, -3.375], [-1.5, -3.375], [-1.5, -3.375], [-1.5, -3.375], [-1.5, -3.375], 
                [-1.5, -3.375], [-1.5, -3.375], [-1.5, -3.375], [-1.5, -3.375], [-1.5, -3.375]]
    ship3_traj = [[-1, -2.25], [-1, -2.25], [-1, -2.25], [-1, -2.25], [-1, -2.25], 
                [-1, -2.25], [-1, -2.25], [-1, -2.25], [-1, -2.25], [-1, -2.25]]
elif scenario == 'overtaking':
    # Position of ASV-start
    sx = 141.0
    sy = 50.0
    # Position of ASV-goal
    gx = 189.0
    gy = 249.0
    # Ships 
    ship1 = [[148, 90], 8, [2, 10.5], 0.2]  # centre point, length, direction, speed
    ship2 = [[145, 122], 4, [1, 6.8], 0.14]  # centre point, length, direction, speed
    ship3 = [[163, 75], 10, [1, 11], 0.2]  # centre point, length, direction, speed
    # Ships trajectories
    ship1_traj = [[2, 10.5], [2, 10.5], [2, 10.5], [2, 10.5], [2, 10.5], 
                    [2, 10.5], [2, 10.5], [2, 10.5], [2, 10.5], [2, 10.5]]
    ship2_traj = [[1, 6.8], [1, 6.8], [1, 6.8], [1, 6.8], [1, 6.8], 
                    [1, 6.8], [1, 6.8], [1, 6.8], [1, 6.8], [1, 6.8]]
    ship3_traj = [[1, 11], [1, 11], [1, 11], [1, 11], [1, 11], 
                    [1, 11], [1, 11], [1, 11], [1, 11], [1, 11]]
elif scenario == 'head-on':
    # Position of ASV-start
    sx = 159.0
    sy = 249.0
    # Position of ASV-goal
    gx = 141.0
    gy = 50.0
    # Ships 
    ship1 = [[148, 90], 8, [2, 10.5], 0.2]  # centre point, length, direction, speed
    ship2 = [[145, 122], 4, [1, 6.8], 0.14]  # centre point, length, direction, speed
    ship3 = [[163, 75], 10, [1, 11], 0.2]  # centre point, length, direction, speed
    # Ships trajectories
    ship1_traj = [[2, 10.5], [2, 10.5], [2, 10.5], [2, 10.5], [2, 10.5], 
                    [2, 10.5], [2, 10.5], [2, 10.5], [2, 10.5], [2, 10.5]]
    ship2_traj = [[1, 6.8], [1, 6.8], [1, 6.8], [1, 6.8], [1, 6.8], 
                    [1, 6.8], [1, 6.8], [1, 6.8], [1, 6.8], [1, 6.8]]
    ship3_traj = [[1, 11], [1, 11], [1, 11], [1, 11], [1, 11], 
                    [1, 11], [1, 11], [1, 11], [1, 11], [1, 11]]
elif scenario == 'crossing':
    # Position of ASV-start
    sx = 266.0
    sy = 210.0
    # Position of ASV-goal
    gx = 102.0
    gy = 234.0
    # Ships 
    ship1 = [[200, 264], 8, [-7, -15], 0.29]  # centre point, length, direction, speed
    ship2 = [[184, 238], 4, [-6, -10], 0.33]  # centre point, length, direction, speed
    ship3 = [[206, 250], 10, [-6, -12], 0.4]  # centre point, length, direction, speed
    # Ships trajectories
    ship1_traj = [[-7, -15], [-7, -15], [-7, -15], [-7, -15], [-2, -17], [-2, -17],
                     [-2, -17], [-2, -17], [-2, -17], [-2, -17]]
    ship2_traj = [[-6, -10], [-6, -10], [-6, -10], [-3, -10], [-3, -10], [-3, -10],
                     [-3, -10], [-3, -10], [-3, -10], [-3, -10]]
    ship3_traj = [[-6, -12], [-6, -12], [-6, -12], [-6, -12], [-2, -6], [-2, -6], 
                     [-2, -6], [-2, -6], [-2, -6], [-2, -6]]
elif scenario == 'difficult':
    # Position of ASV-start
    sx = 96.0
    sy = 176.0
    # Position of ASV-goal
    gx = 270.0
    gy = 224.0
    # Ships 
    ship1 = [[160, 232], 8, [-4, -7], 0.3]  # centre point, length, direction, speed
    ship2 = [[188, 154], 4, [-6, 15], 0.45]  # centre point, length, direction, speed
    ship3 = [[214, 258], 10, [-5, -11], 0.32]  # centre point, length, direction, speed
    # Ships trajectories
    ship1_traj = [[-4, -7], [-4, -7], [-4, -7], [-12, -5], [-12, -5],
                 [-12, -5], [-12, -5], [0, -17], [0, -17], [0, -17]]
    ship2_traj = [[-6, 15], [-6, 15], [-6, 15], [0, 16], [0, 16],
                 [0, 16], [10, 11], [10, 11], [10, 11], [10, 11]]
    # ship3_traj = [[-4.5, -10], [-4.5, -10], [-4.5, -10], [-4.5, -10], [-4.5, -10],
    #              [-7, -9.5], [-7, -9.5], [-7, -9.5], [-7, -9.5], [-7, -9.5]]
    ship3_traj = [[-5, -11], [-5, -11], [-5, -11], [-5, -11], [-5, -11],
                 [-7, -9.5], [-7, -9.5], [-7, -9.5], [-7, -9.5], [-7, -9.5]]
else:
    print('No valid test scenario chosen!')
    quit()

# For calculation purposes
ships = []
ships.extend([ship1, ship2, ship3])
ships_traj = []
ships_traj.extend([ship1_traj, ship2_traj, ship3_traj])

# For plotting ships different
ship_x, ship_y = [], []

# Put ships in the obstacle-list
for i in range(len(ships)):
    x_bow, y_bow, x_stern, y_stern = position_ship(ships[i][0], ships[i][1], ships[i][2])
    x_obstacle, y_obstacle = build_obstacle_line(x_bow, y_bow, x_stern, y_stern)
    ox.extend(x_obstacle)
    oy.extend(y_obstacle)
    ship_x.extend(x_obstacle)
    ship_y.extend(y_obstacle)

# Plot the other vessels courses in the chosen scenario (optional)
# (Only for plotting reasons; no computations after that!)
if course_preview:
    plt.axis(axis_measure)
    plt.grid(True)
    img = plt.imread("foerde_cutout.png")
    plt.imshow(img, extent=axis_measure)
    plt.plot(sx, sy, "og")
    plt.plot(gx, gy, "xb", markersize=10, markeredgewidth=2)
    plt.annotate('ASV', xy=(sx, sy), xytext=(sx - 40, sy - 40), arrowprops=dict(facecolor='black', shrink=0.05),)
    ship1_traj_x = [i[0] for i in ship1_traj]
    ship1_traj_y = [i[1] for i in ship1_traj]
    ship2_traj_x = [i[0] for i in ship2_traj]
    ship2_traj_y = [i[1] for i in ship2_traj]
    ship3_traj_x = [i[0] for i in ship3_traj]
    ship3_traj_y = [i[1] for i in ship3_traj]
    # Move the ships according to the activated scenario
    ship1_traj_x = [copy.deepcopy(ships[0][0][0])]
    ship1_traj_y = [copy.deepcopy(ships[0][0][1])]
    ship2_traj_x = [copy.deepcopy(ships[1][0][0])]
    ship2_traj_y = [copy.deepcopy(ships[1][0][1])]
    ship3_traj_x = [copy.deepcopy(ships[2][0][0])]
    ship3_traj_y = [copy.deepcopy(ships[2][0][1])]
    for i in range(10):
        for j in range(len(ships)):
            ships[j][0][0] += ships_traj[j][i][0]  # move in x direction
            ships[j][0][1] += ships_traj[j][i][1]  # move in y direction
            if j == 0:
                ship1_traj_x.append(ships[j][0][0])
                ship1_traj_y.append(ships[j][0][1])
            if j == 1:
                ship2_traj_x.append(ships[j][0][0])
                ship2_traj_y.append(ships[j][0][1])
            if j == 2:
                ship3_traj_x.append(ships[j][0][0])
                ship3_traj_y.append(ships[j][0][1])
            ships[j][2][0] = ships_traj[j][i][0]  # update ships movement vector (x)
            ships[j][2][1] = ships_traj[j][i][1]  # update ships movement vector (y)
    plt.plot(ship1_traj_x, ship1_traj_y, "--", color='aqua')
    plt.plot(ship2_traj_x, ship2_traj_y, "--", color='aqua')
    plt.plot(ship3_traj_x, ship3_traj_y, "--", color='aqua')
    plt.plot(ship_x, ship_y, "dw")
    ship_x, ship_y = [], []
    for i in range(len(ships)):
        x_bow, y_bow, x_stern, y_stern = position_ship(ships[i][0], ships[i][1], ships[i][2])
        x_obstacle, y_obstacle = build_obstacle_line(x_bow, y_bow, x_stern, y_stern)
        ox.extend(x_obstacle)
        oy.extend(y_obstacle)
        ship_x.extend(x_obstacle)
        ship_y.extend(y_obstacle)
    plt.plot(ship_x, ship_y, "dk")
    plt.pause(1000)
    plt.close()
    quit()



"""
Path planning and plotting
"""
# Results
x_results, y_results = [], []  # A*, Dijkstra, DFS, BFS
paths = []  # RRT
trajectories = []  # DWA
times = []  # for time measurement
times_norm = []  # time measurement normalised with euclidean start - goal distance
lengths = []  # for measurement of path lengths
lengths_norm = []  # path length measurement normalised with euclidean start - goal distance
differences = [0]  # for measuring the differences between two consecutive paths (obviously 0 for first path)
areas_under_paths = []  # used for difference measurement
suboptimality = []  # for measuring the mean deviation from the optimal path (direct line between start and goal)
closest = []  # for measuring the closest encounters
course_changes = []  # for measuring the amount of course changes
path_result = []  # the overall resulting path of all runs
V_new = [[0, 0]]  # velocity for RVO
global_path_planned = False  # flag for hybrid mode (indicates if there is a global path planned)
global_rx = []  # store global path x coordinates
global_ry = []  # store global path y coordinates
local_mode = False  # flag for hybrid mode (indicates local path planning)

# Euclidean distance from ASV to goal
goal_dist = math.sqrt((sx - gx) ** 2 + (sy - gy) ** 2)

# DWA start parameters depending on scenario [x, y, yaw(rad), v(points/s), omega(rad/s)]
if scenario == 'test' or scenario == 'overtaking':
    dwa_start_paras = np.array([sx, sy, math.pi / 8.0, 0.0, 0.0])
else:
    dwa_start_paras = np.array([sx, sy, math.atan2(gy - sy, gx - sx), 0.0, 0.0])

# If hybrid mode is activated, adapt the ships and their trajectories as well as number of runs
if hybrid_mode:
    ships, ships_traj = adapt_ships_to_hybrid(hybrid_runs, ships, ships_traj)
    runs = hybrid_runs

# Loop of the simulations timesteps (with ship movement)
for i in range(runs):

    # Timestamps (for normal path finding and for extra smoothing)
    start = 0
    end = 0
    start_smoothing = 0
    end_smoothing = 0

    # Set obstacle positions
    ox, oy = [], []

    # Check if hybrid mode is active and set variables accordingly
    if hybrid_mode and not global_path_planned:
        algorithm = "A*"
        grid_size = 1.5
        smoothing_of_path = True
    if hybrid_mode and local_mode:
        algorithm = "RVO"
        grid_size = 3.0

    # Following steps only necessary for A*, Dijkstra, DFS, BFS, D*
    if algorithm == 'A*' or algorithm == 'Dijkstra' or algorithm == 'DFS' or algorithm == 'BFS'\
            or algorithm == 'D*': 
    # Configure general borders (nescessary for calculations)
        for j in range(axis_measure[0], axis_measure[1] + 1):
            ox.append(j)
            oy.append(0.0)
        for j in range(axis_measure[2], axis_measure[3]):
            ox.append(0.0)
            oy.append(j)

        # Limit navigable terrain to Kieler Foerde area (in foerde_cutout-image)
        x_boundaries, y_boundaries = build_boundaries(xy_list=kieler_foerde_cutout)
        ox.extend(x_boundaries)
        oy.extend(y_boundaries)

    # For plotting ships differently
    ship_x, ship_y = [], []
    # For ship projection
    ships_proj = []
    # For plotting ship projections differently
    ship_proj_x, ship_proj_y = [], []
    # For plotting virtual COLREG obstacles differently
    colreg_x, colreg_y = [], []
    # Individual obstacles (mainly other ships)
    # Project other ships further along their actual course if projection option is activated
    if projection:
        # (Deep) Copy ships to ship projections
        ships_proj = copy.deepcopy(ships)
        # Save actual ship positions for plotting
        for j in range(len(ships)):
            x_bow, y_bow, x_stern, y_stern = position_ship(ships[j][0], ships[j][1], ships[j][2])
            x_obstacle, y_obstacle = build_obstacle_line(x_bow, y_bow, x_stern, y_stern)
            ship_x.extend(x_obstacle)
            ship_y.extend(y_obstacle)
        # Use CPA-based projection for dynamic/local path planning algorithms ...
        ships_proj_indices = []  # for storing the relevant ship indices
        if algorithm == "DWA" or algorithm == "RVO":
            if algorithm == "DWA":
                if dwa_start_paras[3] == 0:
                    # Parameters for DWA: position, goal, DWA speed, vessels
                    ships_proj_position = cpa_projection([sx, sy], [gx, gy], 0.26, asv_radius, ships)
                else:
                    # Parameters for DWA: position, goal, DWA speed (with DWA adjustment), vessels
                    ships_proj_position = cpa_projection([sx, sy], [sx + math.cos(dwa_start_paras[2]) * dwa_start_paras[3], 
                                                                    sy + math.sin(dwa_start_paras[2]) * dwa_start_paras[3]], 
                                                                    dwa_start_paras[3] / 20, asv_radius, ships)                                   
            else:
                # Determine speed from the RVO velocity vector
                rvo_speed = math.sqrt(V_new[0][0] ** 2 + V_new[0][1] ** 2)
                if V_new == [[0, 0]]:
                    # Parameters for VO: position, goal, RVO speed, vessels
                    ships_proj_position = cpa_projection([sx, sy], [gx, gy], 0.26, asv_radius, ships)
                else:
                    # Parameters for VO: position, goal, RVO speed, vessels
                    ships_proj_position = cpa_projection([sx, sy], [sx + V_new[0][0], sy + V_new[0][1]], rvo_speed, 
                                                         asv_radius, ships)
            ships_proj_temp = []
            for j in range(len(ships_proj_position)):
                if ships_proj_position[j][0] != 999 and ships_proj_position[j][1] != 999:
                    ships_proj_temp.append(copy.deepcopy(ships[j]))
                    ships_proj_temp[-1][0][0] = ships_proj_position[j][0]
                    ships_proj_temp[-1][0][1] = ships_proj_position[j][1]
                    ships_proj_indices.append(j)  # store indices of projected ships
            ships_proj = copy.deepcopy(ships_proj_temp)
        # ... or the standard probability projection for the global path planning algorithms
        else:
            # Euclidean distance to other ships
            ships_dist = []
            for j in range(len(ships)):
                ships_dist.append(math.sqrt((sx - ships[j][0][0]) ** 2 + (sy - ships[j][0][1]) ** 2))
            # Build a 'probability cloud' around the projected ships, based on the distance
            # First lengthen the projection
            for j in range(len(ships)):
                ships_proj[j][1] += int(ships_dist[j] / probability_factor) * 2 
            # Second widen the projection based on the orthogonal vectors of the ships direction
            for j in range(len(ships)):
                # Calculate the two orthogonal vectors
                ortho_1 = [-ships[j][2][1], ships[j][2][0]]
                ortho_2 = [ships[j][2][1], -ships[j][2][0]]
                # Calculate the unit vectors of the orthogonal vectors
                x1_temp = (1 / math.sqrt(ortho_1[0] ** 2 + ortho_1[1] ** 2)) * ortho_1[0]
                y1_temp = (1 / math.sqrt(ortho_1[0] ** 2 + ortho_1[1] ** 2)) * ortho_1[1]
                x2_temp = (1 / math.sqrt(ortho_2[0] ** 2 + ortho_2[1] ** 2)) * ortho_2[0]
                y2_temp = (1 / math.sqrt(ortho_2[0] ** 2 + ortho_2[1] ** 2)) * ortho_2[1]
                # Add 'new' projected ships next to the 'real' projection in order to widen it
                # Number of new layers is based on the distance
                for k in range(math.ceil(ships_dist[j] / probability_factor)):
                    ship_proj_temp_1 = copy.deepcopy(ships[j])
                    ship_proj_temp_2 = copy.deepcopy(ships[j])
                    ship_proj_temp_1[0][0] += x1_temp * k * probability_distance 
                    ship_proj_temp_1[0][1] += y1_temp * k * probability_distance
                    ship_proj_temp_2[0][0] += x2_temp * k * probability_distance
                    ship_proj_temp_2[0][1] += y2_temp * k * probability_distance
                    # Adapt the length of the projection layer (down to original length in the outermost layer)
                    ship_proj_temp_1[1] += (int(ships_dist[j] / probability_factor) - k) * 2
                    ship_proj_temp_2[1] += (int(ships_dist[j] / probability_factor) - k) * 2
                    # Add the layers
                    ships_proj.extend([copy.deepcopy(ship_proj_temp_1), copy.deepcopy(ship_proj_temp_2)])
                    # Extend the ships distance list, so that it corresponds to the ship projections list
                    ships_dist.extend([ships_dist[j], ships_dist[j]])
                # # TEST possibility to add only the outermost layer of the projection
                # k = math.ceil(ships_dist[j] / probability_factor)
                # ship_proj_temp_1 = copy.deepcopy(ships[j])
                # ship_proj_temp_2 = copy.deepcopy(ships[j])
                # ship_proj_temp_1[0][0] += x1_temp * k * probability_distance 
                # ship_proj_temp_1[0][1] += y1_temp * k * probability_distance
                # ship_proj_temp_2[0][0] += x2_temp * k * probability_distance
                # ship_proj_temp_2[0][1] += y2_temp * k * probability_distance
                # Adapt the length of the projection layer (down to original length in the outermost layer)
                ship_proj_temp_1[1] += (int(ships_dist[j] / probability_factor) - k) * 2
                ship_proj_temp_2[1] += (int(ships_dist[j] / probability_factor) - k) * 2
                # Add the layers
                ships_proj.extend([copy.deepcopy(ship_proj_temp_1), copy.deepcopy(ship_proj_temp_2)])
                # Extend the ships distance list, so that it corresponds to the ship projections list
                ships_dist.extend([ships_dist[j], ships_dist[j]])
            # Project ships a certain number of timesteps based on the comparison of the calculated distances
            # with the original start goal distance
            for j in range(len(ships_proj)):
                timesteps = int(ships_dist[j] / (goal_dist / runs))
                ships_proj[j][0][0] += timesteps * ships_proj[j][2][0]
                ships_proj[j][0][1] += timesteps * ships_proj[j][2][1]
        # Build the projected ship obstacles in the simulation
        for j in range(len(ships_proj)):
            x_bow, y_bow, x_stern, y_stern = position_ship(ships_proj[j][0], ships_proj[j][1], ships_proj[j][2])
            x_obstacle, y_obstacle = build_obstacle_line(x_bow, y_bow, x_stern, y_stern)
            ox.extend(x_obstacle)
            oy.extend(y_obstacle)
            ship_proj_x.extend(x_obstacle)
            ship_proj_y.extend(y_obstacle)
        if colregs:
            # Build COLREG virtual obstacles in the simulation
            for j in range(len(ships_proj)):
                # Determine the orthogonal vector on port side
                if ships_proj[j][2][0] >= 0 and ships_proj[j][2][1] >= 0:  # (x >= 0, y >= 0)
                    orthogonal = [-ships_proj[j][2][1], ships_proj[j][2][0]]
                elif ships_proj[j][2][0] < 0 and ships_proj[j][2][1] < 0:  # (x < 0, y < 0)
                    orthogonal = [-ships_proj[j][2][1], ships_proj[j][2][0]]
                elif ships_proj[j][2][0] >= 0 and ships_proj[j][2][1] < 0:  # (x >= 0, y < 0)
                    orthogonal = [ships_proj[j][2][1], -ships_proj[j][2][0]]
                elif ships_proj[j][2][0] < 0 and ships_proj[j][2][1] >= 0:  # (x < 0, y >= 0)
                    orthogonal = [ships_proj[j][2][1], -ships_proj[j][2][0]]
                # Calculate the unit vector of the orthogonal vector
                x_temp = (1 / math.sqrt(orthogonal[0] ** 2 + orthogonal[1] ** 2)) * orthogonal[0]
                y_temp = (1 / math.sqrt(orthogonal[0] ** 2 + orthogonal[1] ** 2)) * orthogonal[1]
                # Add the orthogonal virtual COLREG obstacle on port side
                colreg_temp = copy.deepcopy(ships_proj[j])
                colreg_temp[1]  # /= 2  # halve the length
                colreg_temp[2][0] = orthogonal[0]  # change x direction
                colreg_temp[2][1] = orthogonal[1]  # change y direction
                colreg_temp[0][0] += x_temp * colreg_temp[1]  # shift center x
                colreg_temp[0][1] += y_temp * colreg_temp[1]  # shift center y
                # Add the whole obstacle
                x_bow, y_bow, x_stern, y_stern = position_ship(colreg_temp[0], colreg_temp[1], colreg_temp[2])
                x_obstacle, y_obstacle = build_obstacle_line(x_bow, y_bow, x_stern, y_stern)
                ox.extend(x_obstacle)
                oy.extend(y_obstacle)
                colreg_x.extend(x_obstacle)
                colreg_y.extend(y_obstacle)
                # Save bow coordinates for the third colreg obstacle line
                x_1, y_1 = x_bow, y_bow
                # Add the forward virtual COLREG obstacle
                colreg_temp = copy.deepcopy(ships_proj[j])
                # Calculate the unit vector
                x_temp = (1 / math.sqrt(colreg_temp[2][0] ** 2 + colreg_temp[2][1] ** 2)) * colreg_temp[2][0]
                y_temp = (1 / math.sqrt(colreg_temp[2][0] ** 2 + colreg_temp[2][1] ** 2)) * colreg_temp[2][1]
                # Add the orthogonal virtual COLREG obstacle on the bow
                colreg_temp[1] /= 2  # halve the length
                colreg_temp[0][0] += x_temp * 1.5 * colreg_temp[1]  # shift center x
                colreg_temp[0][1] += y_temp * 1.5 * colreg_temp[1]  # shift center y
                # Add the whole obstacle
                x_bow, y_bow, x_stern, y_stern = position_ship(colreg_temp[0], colreg_temp[1], colreg_temp[2])
                x_obstacle, y_obstacle = build_obstacle_line(x_bow, y_bow, x_stern, y_stern)
                ox.extend(x_obstacle)
                oy.extend(y_obstacle)
                colreg_x.extend(x_obstacle)
                colreg_y.extend(y_obstacle)
                # Save bow coordinates for the third colreg obstacle line
                x_2, y_2 = x_bow, y_bow
                # Add the third virtual COLREG obstacle line
                x_obstacle, y_obstacle = build_obstacle_line(x_1, y_1, x_2, y_2)
                ox.extend(x_obstacle)
                oy.extend(y_obstacle)
                colreg_x.extend(x_obstacle)
                colreg_y.extend(y_obstacle)
    else:
        # Build the ship obstacles in the simulation
        for j in range(len(ships)):
            x_bow, y_bow, x_stern, y_stern = position_ship(ships[j][0], ships[j][1], ships[j][2])
            x_obstacle, y_obstacle = build_obstacle_line(x_bow, y_bow, x_stern, y_stern)
            ox.extend(x_obstacle)
            oy.extend(y_obstacle)
            ship_x.extend(x_obstacle)
            ship_y.extend(y_obstacle)
        if colregs:
            # Build COLREG virtual obstacles in the simulation
            for j in range(len(ships)):
                # Determine the orthogonal vector on port side
                if ships[j][2][0] >= 0 and ships[j][2][1] >= 0:  # (x >= 0, y >= 0)
                    orthogonal = [-ships[j][2][1], ships[j][2][0]]
                elif ships[j][2][0] < 0 and ships[j][2][1] < 0:  # (x < 0, y < 0)
                    orthogonal = [-ships[j][2][1], ships[j][2][0]]
                elif ships[j][2][0] >= 0 and ships[j][2][1] < 0:  # (x >= 0, y < 0)
                    orthogonal = [ships[j][2][1], -ships[j][2][0]]
                elif ships[j][2][0] < 0 and ships[j][2][1] >= 0:  # (x < 0, y >= 0)
                    orthogonal = [ships[j][2][1], -ships[j][2][0]]
                # Calculate the unit vector of the orthogonal vector
                x_temp = (1 / math.sqrt(orthogonal[0] ** 2 + orthogonal[1] ** 2)) * orthogonal[0]
                y_temp = (1 / math.sqrt(orthogonal[0] ** 2 + orthogonal[1] ** 2)) * orthogonal[1]
                # Add the orthogonal virtual COLREG obstacle on port side
                colreg_temp = copy.deepcopy(ships[j])
                colreg_temp[1] /= 2  # halve the length
                colreg_temp[2][0] = orthogonal[0]  # change x direction
                colreg_temp[2][1] = orthogonal[1]  # change y direction
                colreg_temp[0][0] += x_temp * colreg_temp[1]  # shift center x
                colreg_temp[0][1] += y_temp * colreg_temp[1]  # shift center y
                # Add the whole obstacle
                x_bow, y_bow, x_stern, y_stern = position_ship(colreg_temp[0], colreg_temp[1], colreg_temp[2])
                x_obstacle, y_obstacle = build_obstacle_line(x_bow, y_bow, x_stern, y_stern)
                ox.extend(x_obstacle)
                oy.extend(y_obstacle)
                colreg_x.extend(x_obstacle)
                colreg_y.extend(y_obstacle)
                # Save bow coordinates for the third colreg obstacle line
                x_1, y_1 = x_bow, y_bow
                # Add the forward virtual COLREG obstacle
                colreg_temp = copy.deepcopy(ships[j])
                # Calculate the unit vector
                x_temp = (1 / math.sqrt(colreg_temp[2][0] ** 2 + colreg_temp[2][1] ** 2)) * colreg_temp[2][0]
                y_temp = (1 / math.sqrt(colreg_temp[2][0] ** 2 + colreg_temp[2][1] ** 2)) * colreg_temp[2][1]
                # Add the orthogonal virtual COLREG obstacle on the bow
                colreg_temp[1] /= 2  # halve the length
                colreg_temp[0][0] += x_temp * 1.5 * colreg_temp[1]  # shift center x
                colreg_temp[0][1] += y_temp * 1.5 * colreg_temp[1]  # shift center y
                # Add the whole obstacle
                x_bow, y_bow, x_stern, y_stern = position_ship(colreg_temp[0], colreg_temp[1], colreg_temp[2])
                x_obstacle, y_obstacle = build_obstacle_line(x_bow, y_bow, x_stern, y_stern)
                ox.extend(x_obstacle)
                oy.extend(y_obstacle)
                colreg_x.extend(x_obstacle)
                colreg_y.extend(y_obstacle)
                # Save bow coordinates for the third colreg obstacle line
                x_2, y_2 = x_bow, y_bow
                # Add the third virtual COLREG obstacle line
                x_obstacle, y_obstacle = build_obstacle_line(x_1, y_1, x_2, y_2)
                ox.extend(x_obstacle)
                oy.extend(y_obstacle)
                colreg_x.extend(x_obstacle)
                colreg_y.extend(y_obstacle)

    if show_animation:
    # Prepare the plot
        if colregs:
            plt.plot(colreg_x, colreg_y, "o", color="orange")
        if projection and (not global_path_planned or local_mode):
            plt.plot(ship_x, ship_y, "dw")
            x_proj, y_proj = [], []
            for j in range(len(ships)):
                if algorithm != "DWA" and algorithm != "RVO":
                    x_proj = [ships[j][0][0], ships_proj[j][0][0]]
                    y_proj = [ships[j][0][1], ships_proj[j][0][1]]
                    plt.plot(x_proj, y_proj, ":w")
                else:
                    if j in ships_proj_indices:
                        x_proj = [ships[j][0][0], ships_proj[ships_proj_indices.index(j)][0][0]]
                        y_proj = [ships[j][0][1], ships_proj[ships_proj_indices.index(j)][0][1]]
                        plt.plot(x_proj, y_proj, ":w")
                    else:
                        plt.plot(ships[j][0][0], ships[j][0][1], "x", color="darkslategrey")
            plt.plot(ship_proj_x, ship_proj_y, "d", color="darkslategrey")
        else:
            plt.plot(ship_x, ship_y, "dk")
        plt.plot(ox, oy, ",w")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb", markersize=10, markeredgewidth=2)
        plt.grid(True)
        plt.annotate('ASV', xy=(sx, sy), xytext=(sx - 40, sy - 40), arrowprops=dict(facecolor='black', shrink=0.05),)
        plt.axis(axis_measure)
        img = plt.imread("foerde_cutout.png")
        plt.imshow(img, extent=axis_measure)
        plt.rcParams["figure.figsize"] = (8, 6)  # Smaller windowsize -> higher speed

    # Activate the chosen algorithm
    if not hybrid_mode or not global_path_planned or (hybrid_mode and local_mode):
        if algorithm == 'A*':
            a_star = AStarPlanner(ox, oy, grid_size, asv_radius)
            a_star.show_animation = show_animation
            # The actual algorithm (with time measurement)
            start = time.perf_counter()
            rx, ry = a_star.planning(sx, sy, gx, gy)
            end = time.perf_counter()
        elif algorithm == 'Dijkstra':
            dijkstra = Dijkstra(ox, oy, grid_size, asv_radius)
            dijkstra.show_animation = show_animation
            # The actual algorithm (with time measurement)
            start = time.perf_counter()
            rx, ry = dijkstra.planning(sx, sy, gx, gy)
            end = time.perf_counter()
        elif algorithm == 'DFS':
            dfs = DepthFirstSearchPlanner(ox, oy, grid_size, asv_radius)
            dfs.show_animation = show_animation
            # The actual algorithm (with time measurement)
            start = time.perf_counter()
            rx, ry = dfs.planning(sx, sy, gx, gy)
            end = time.perf_counter()
        elif algorithm == 'BFS':
            bfs = BreadthFirstSearchPlanner(ox, oy, grid_size, asv_radius)
            bfs.show_animation = show_animation
            # The actual algorithm (with time measurement)
            start = time.perf_counter()
            rx, ry = bfs.planning(sx, sy, gx, gy)
            end = time.perf_counter()
        elif algorithm == 'D*':
            # First we need to build a safety area around the ship obstacles as a projection
            # (Deep) Copy ships to ship projections
            ships_proj_d = copy.deepcopy(ships)
            # Widen the ship obstacles based on the orthogonal vectors of the ships directions
            for j in range(len(ships)):
                # Calculate the two orthogonal vectors
                ortho_1 = [-ships[j][2][1], ships[j][2][0]]
                ortho_2 = [ships[j][2][1], -ships[j][2][0]]
                # Calculate the unit vectors of the orthogonal vectors
                x1_temp = (1 / math.sqrt(ortho_1[0] ** 2 + ortho_1[1] ** 2)) * ortho_1[0]
                y1_temp = (1 / math.sqrt(ortho_1[0] ** 2 + ortho_1[1] ** 2)) * ortho_1[1]
                x2_temp = (1 / math.sqrt(ortho_2[0] ** 2 + ortho_2[1] ** 2)) * ortho_2[0]
                y2_temp = (1 / math.sqrt(ortho_2[0] ** 2 + ortho_2[1] ** 2)) * ortho_2[1]
                # Add 'new' projected ships next to the 'real' projection in order to widen it
                ship_proj_temp_1 = copy.deepcopy(ships[j])
                ship_proj_temp_2 = copy.deepcopy(ships[j])
                ship_proj_temp_1[0][0] += x1_temp * (asv_radius - 1) 
                ship_proj_temp_1[0][1] += y1_temp * (asv_radius - 1)
                ship_proj_temp_2[0][0] += x2_temp * (asv_radius - 1)
                ship_proj_temp_2[0][1] += y2_temp * (asv_radius - 1)
                # Add the safety area
                ships_proj_d.extend([copy.deepcopy(ship_proj_temp_1), copy.deepcopy(ship_proj_temp_2)])
            # Build the projected safety area obstacles in the simulation
            for j in range(len(ships_proj_d)):
                x_bow, y_bow, x_stern, y_stern = position_ship(ships_proj_d[j][0], ships_proj_d[j][1], ships_proj_d[j][2])
                x_obstacle, y_obstacle = build_obstacle_line(x_bow, y_bow, x_stern, y_stern)
            # Create a circular safety space around the ships bows and sterns
            # Theta goes from 0 to 2pi
            theta = np.linspace(0, 2 * np.pi, 120)  # every 3° a new point
            # The radius of the circle
            radius = asv_radius - 1
            # Compute xs and ys of the circle
            x_circle = radius * np.cos(theta)
            y_circle = radius * np.sin(theta)
            # Build the circular safety area obstacles in the simulation
            for j in range(len(ships_proj)):
                x_bow, y_bow, x_stern, y_stern = position_ship(ships_proj[j][0], ships_proj[j][1], ships_proj[j][2])
                ox.extend(x_circle + x_bow)
                ox.extend(x_circle + x_stern)
                oy.extend(y_circle + y_bow)
                oy.extend(y_circle + y_stern)
                # Do the same thing additionally for COLREGs if activated
                if colregs:
                    # Determine the orthogonal vector on port side
                    if ships_proj[j][2][0] >= 0 and ships_proj[j][2][1] >= 0:  # (x >= 0, y >= 0)
                        orthogonal = [-ships_proj[j][2][1], ships_proj[j][2][0]]
                    elif ships_proj[j][2][0] < 0 and ships_proj[j][2][1] < 0:  # (x < 0, y < 0)
                        orthogonal = [-ships_proj[j][2][1], ships_proj[j][2][0]]
                    elif ships_proj[j][2][0] >= 0 and ships_proj[j][2][1] < 0:  # (x >= 0, y < 0)
                        orthogonal = [ships_proj[j][2][1], -ships_proj[j][2][0]]
                    elif ships_proj[j][2][0] < 0 and ships_proj[j][2][1] >= 0:  # (x < 0, y >= 0)
                        orthogonal = [ships_proj[j][2][1], -ships_proj[j][2][0]]
                    # Calculate the unit vector of the orthogonal vector
                    x_temp = (1 / math.sqrt(orthogonal[0] ** 2 + orthogonal[1] ** 2)) * orthogonal[0]
                    y_temp = (1 / math.sqrt(orthogonal[0] ** 2 + orthogonal[1] ** 2)) * orthogonal[1]
                    # Add the orthogonal virtual COLREG obstacle on port side
                    colreg_temp = copy.deepcopy(ships_proj[j])
                    colreg_temp[1] /= 2  # halve the length
                    colreg_temp[2][0] = orthogonal[0]  # change x direction
                    colreg_temp[2][1] = orthogonal[1]  # change y direction
                    colreg_temp[0][0] += x_temp * colreg_temp[1]  # shift center x
                    colreg_temp[0][1] += y_temp * colreg_temp[1]  # shift center y
                    # Add the whole obstacle
                    x_bow, y_bow, x_stern, y_stern = position_ship(colreg_temp[0], colreg_temp[1], colreg_temp[2])
                    # x_obstacle, y_obstacle = build_obstacle_line(x_bow, y_bow, x_stern, y_stern)
                    ox.extend(x_circle + x_bow)
                    ox.extend(x_circle + x_stern)
                    oy.extend(y_circle + y_bow)
                    oy.extend(y_circle + y_stern)
                    # Add the forward virtual COLREG obstacle
                    colreg_temp = copy.deepcopy(ships_proj[j])
                    # Calculate the unit vector
                    x_temp = (1 / math.sqrt(colreg_temp[2][0] ** 2 + colreg_temp[2][1] ** 2)) * colreg_temp[2][0]
                    y_temp = (1 / math.sqrt(colreg_temp[2][0] ** 2 + colreg_temp[2][1] ** 2)) * colreg_temp[2][1]
                    # Add the orthogonal virtual COLREG obstacle on the bow
                    colreg_temp[1] /= 2  # halve the length
                    colreg_temp[0][0] += x_temp * 1.5 * colreg_temp[1]  # shift center x
                    colreg_temp[0][1] += y_temp * 1.5 * colreg_temp[1]  # shift center y
                    # Add the whole obstacle
                    x_bow, y_bow, x_stern, y_stern = position_ship(colreg_temp[0], colreg_temp[1], colreg_temp[2])
                    ox.extend(x_circle + x_bow)
                    ox.extend(x_circle + x_stern)
                    oy.extend(y_circle + y_bow)
                    oy.extend(y_circle + y_stern)
            # # Plot the safety area for test purposes
            # plt.plot(ox, oy, ",r")
            m = Map(axis_measure[1], axis_measure[3])
            m.set_obstacle([(int(i), int(j)) for i, j in zip(ox, oy)])
            map_start = m.map[int(sx)][int(sy)]
            map_end = m.map[int(gx)][int(gy)]
            dstar = Dstar(m)
            dstar.show_animation = show_animation
            # The actual algorithm (with time measurement)
            start = time.perf_counter()
            rx, ry = dstar.run(map_start, map_end)
            end = time.perf_counter()
        # elif algorithm == 'DStarLite':
        #     dstarlite = DStarLite(ox, oy)
        #     dstarlite.show_animation = show_animation
        #     # # DEBUGGING!!!
        #     # spoofed_ox = [[], [], [],
        #     #           [i for i in range(0, 21)] + [0 for _ in range(0, 20)]]
        #     # spoofed_oy = [[], [], [],
        #     #           [20 for _ in range(0, 21)] + [i for i in range(0, 20)]]
        #     # The actual algorithm (with time measurement)
        #     start = time.perf_counter()
        #     dstarlite_bool, rx, ry = dstarlite.main(Node(x=int(sx), y=int(sy)), Node(x=int(gx), y=int(gy)), 
        #         spoofed_ox=[], spoofed_oy=[])
        #     end = time.perf_counter()
        elif algorithm == 'RRT':
            if not rrt_bestof:
                # One normal RRT run (if best-of option is not activated)
                # Format the obstacles for RRT
                obstacleList = []
                for j in range(len(ox)):
                    obstacleList.append((ox[j], oy[j], 0.5))  # (x, y, radius)
                # Determine a plausible corridor to limit the random sampling in the foerde simulation
                difference = max(max(sy, gy) - min (sy, gy), 25)  # 25 * 2 = 50 is a minimum for the corridor
                sampling_corridor = [max(min(sy, gy) - difference, axis_measure[2]), 
                    min(max(sy, gy) + difference, axis_measure[3])]  # corridor boundaries limited by coordinate system
                # Initialize RRT    
                rrt = RRT(
                start=[sx, sy],
                goal=[gx, gy],
                rand_area=[axis_measure[0], axis_measure[1]],
                extra_rand_boundaries=foerde_shores,
                obstacle_list=obstacleList,
                # Limit to plausible sample space [xmin, xmax, ymin, ymax]
                play_area=[axis_measure[0], axis_measure[1], sampling_corridor[0], sampling_corridor[1]],  
                path_resolution=grid_size,  # 0.1 possible
                expand_dis=grid_size * 4.0,  # special factor for RRT; 0.4 possible
                robot_radius=asv_radius,
                steering_angle=steering_angle,
                ships=ships,
                ships_proj=ships_proj,
                ship_x=ship_x,
                ship_y=ship_y,
                ship_proj_x=ship_proj_x,
                ship_proj_y=ship_proj_y,
                colreg_x=colreg_x,
                colreg_y=colreg_y
                )
                # The actual algorithm (with time measurement)
                start = time.perf_counter()
                path = rrt.planning(animation=show_animation)
                end = time.perf_counter()
            else:
                # Use the best RRT path of several runs
                paths_temp = []
                time_temp = 0.0
                lengths_temp = []
                for j in range(rrt_bestof_runs):
                    # Format the obstacles for RRT
                    obstacleList = []
                    for k in range(len(ox)):
                        obstacleList.append((ox[k], oy[k], 0.5))  # (x, y, radius)
                    # Determine a plausible corridor to limit the random sampling in the foerde simulation
                    difference = max(max(sy, gy) - min (sy, gy), 25)  # 25 * 2 = 50 is a minimum for the corridor
                    sampling_corridor = [max(min(sy, gy) - difference, axis_measure[2]), 
                        min(max(sy, gy) + difference, axis_measure[3])]  # corridor boundaries limited by coordinate system
                    # Initialize RRT    
                    rrt = RRT(
                    start=[sx, sy],
                    goal=[gx, gy],
                    rand_area=[axis_measure[0], axis_measure[1]],
                    extra_rand_boundaries=foerde_shores,
                    obstacle_list=obstacleList,
                    # Limit to plausible sample space
                    play_area=[axis_measure[0], axis_measure[1], sampling_corridor[0], sampling_corridor[1]],  
                    path_resolution=grid_size,  # 0.1 possible
                    expand_dis=grid_size * 4.0,  # special factor for RRT; 0.4 possible
                    robot_radius=asv_radius,
                    steering_angle=steering_angle,
                    ships=ships,
                    ships_proj=ships_proj,
                    ship_x=ship_x,
                    ship_y=ship_y,
                    ship_proj_x=ship_proj_x,
                    ship_proj_y=ship_proj_y,
                    colreg_x=colreg_x,
                    colreg_y=colreg_y
                    )
                    # The actual algorithm (with time measurement)
                    start = time.perf_counter()
                    path = rrt.planning(animation=show_animation)
                    end = time.perf_counter()
                    # Save path to compare
                    paths_temp.append(path)
                    # Save time to sum runtime for bestof
                    time_temp += end - start
                    # Calculate length for comparison in order to find the shortest/best path
                    length = 0.0
                    for k in range(len(path) - 1):
                        # Calculate Euclidean distance of one path segment
                        temp = math.sqrt((path[k + 1][0] - path[k][0]) ** 2 + (path[k + 1][1] - path[k][1]) ** 2)
                        length += temp
                    lengths_temp.append(length)
                    # Plot the path
                    if show_animation:
                        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
                        plt.grid(True)
                        plt.pause(0.01)  # Enough time for path generation
                        plt.show(block=False)
                        plt.pause(1)
                        plt.close()
                # Show all sampled paths
                if show_animation:
                    plt.axis(axis_measure)
                    plt.grid(True)
                    img = plt.imread("foerde_cutout.png")
                    plt.imshow(img, extent=axis_measure)
                    plt.plot(sx, sy, "og")
                    plt.plot(gx, gy, "xb", markersize=10, markeredgewidth=2)
                    if colregs:
                        plt.plot(colreg_x, colreg_y, "o", color="orange")
                    if projection:
                        x_proj, y_proj = [], []
                        for j in range(len(ships)):
                            x_proj = [ships[j][0][0], ships_proj[j][0][0]]
                            y_proj = [ships[j][0][1], ships_proj[j][0][1]]
                            plt.plot(x_proj, y_proj, ":w")
                        plt.plot(ship_x, ship_y, "dw")
                        plt.plot(ship_proj_x, ship_proj_y, "d", color="darkslategrey")
                    else:
                        plt.plot(ship_x, ship_y, "dk")
                    plt.title('Sampled Paths')
                # Find the shortest (=best) path
                best_path_index = 0
                best_path_length = 99999
                for j in range(len(lengths_temp)):
                    if lengths_temp[j] < best_path_length:
                        best_path_length = lengths_temp[j]
                        best_path_index = j
                # Add all found paths in distinguishable colors and highlight the best
                if show_animation:
                    for j in range(len(paths_temp)):
                        if j == best_path_index:
                            plt.plot([x for (x, y) in paths_temp[j]], [y for (x, y) in paths_temp[j]], "-r")
                        else:
                            plt.plot([x for (x, y) in paths_temp[j]], [y for (x, y) in paths_temp[j]], "-", 
                                    color='mistyrose')
                plt.pause(1)
                plt.close()    
                # Forward the best path and the runtime
                path = paths_temp[best_path_index]
                start = 0.0
                end = time_temp
        elif algorithm == 'DWA':
            # Necessary variables for DWA
            robot_type = DWA.RobotType.circle
            config = DWA.Config()
            config.robot_radius = asv_radius
            config.dt *= grid_size  # time tick for motion predicition
            config.max_speed = 0.38  # about 10 knots
            config.min_speed = -0.19  # max speed halved
            config.max_yaw_rate = 30.0 * math.pi / 180.0  # about 30° steering angle over 14 meters (rad) 
            config.max_accel = 0.1  # so ASV reaches 10 knots in about 50 m
            config.max_delta_yaw_rate = 30 * math.pi / 180.0  # how fast can the ruder change its angle over 14 m (rad)        # config.yaw_rate_resolution *= granularity
            config.to_goal_cost_gain = 0.15
            config.speed_cost_gain = 1.0
            config.obstacle_cost_gain = 0.2
            # Apply the DWA start parameters [x, y, yaw(rad), v(points/s), omega(rad/s)]
            x = dwa_start_paras
            trajectory = np.array(x)
            # goal position
            goal = np.array([gx, gy])
            # Format the obstacles for DWA
            ox.extend(x_boundaries)  # extend with foerde cutout 
            oy.extend(y_boundaries)  # extend with foerde cutout
            obstacleList = []
            for j in range(len(ox)):
                obstacleList.append([(ox[j] / 20.0), (oy[j] / 20.0)])
            config.ob = np.array(obstacleList)
            ob = config.ob
            config.robot_type = robot_type
            # The actual algorithm (with time measurement)
            start = time.perf_counter()
            while True:
                # Adjust the parameters for the algorithm
                x[0] /= 20.0  # sx
                x[1] /= 20.0  # sy
                x[3] /= 20.0  # v (points/s)
                config.robot_radius /= 20.0  # ASV radius
                goal[0] = gx / 20.0  # gx
                goal[1] = gy / 20.0  # gy
                u, predicted_trajectory = DWA.dwa_control(x, config, goal, ob)
                x = DWA.motion(x, u, config.dt)  # simulate robot
                # Re-adjust the parameters
                x[0] *= 20.0  # sx
                x[1] *= 20.0  # sy
                x[3] *= 20.0  # v/s
                config.robot_radius *= 20.0  # ASV radius
                trajectory = np.vstack((trajectory, x))  # store state history
                # Show animation if True
                if show_animation:
                    plt.cla()
                    # For stopping simulation with the esc key
                    plt.gcf().canvas.mpl_connect(
                        'key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                    if colregs:
                        plt.plot(colreg_x, colreg_y, "o", color="orange")
                    if projection:
                        plt.plot(ship_x, ship_y, "dw")
                        x_proj, y_proj = [], []
                        for j in range(len(ships)):
                            if j in ships_proj_indices:
                                x_proj = [ships[j][0][0], ships_proj[ships_proj_indices.index(j)][0][0]]
                                y_proj = [ships[j][0][1], ships_proj[ships_proj_indices.index(j)][0][1]]
                                plt.plot(x_proj, y_proj, ":w")
                            else:
                                plt.plot(ships[j][0][0], ships[j][0][1], "x", color="darkslategrey")
                        plt.plot(ship_proj_x, ship_proj_y, "d", color="darkslategrey")
                    else:
                        plt.plot(ship_x, ship_y, "dk")               
                    plt.plot((predicted_trajectory[:, 0] * 20.0), (predicted_trajectory[:, 1] * 20), "-g")
                    plt.plot(x[0], x[1], "og")
                    plt.plot(gx, gy, "xb", markersize=10, markeredgewidth=2)
                    plt.annotate('ASV', xy=(x[0], x[1]), xytext=(x[0] - 40, x[1] - 40), 
                        arrowprops=dict(facecolor='black', shrink=0.05),)
                    DWA.plot_robot(x[0], x[1], x[2], config)
                    DWA.plot_arrow(x[0], x[1], x[2], length=10.0, width=2.0)
                    plt.axis(axis_measure)
                    plt.grid(True)
                    img = plt.imread("foerde_cutout.png")
                    plt.imshow(img, extent=axis_measure)
                    plt.pause(0.0001)
                # Check if the goal is reached
                dist_to_goal = math.hypot(x[0] - gx, x[1] - gy)
                if dist_to_goal <= config.robot_radius:
                    # Add goal position
                    trajectory = np.append(trajectory, [trajectory[-1]], axis=0)
                    trajectory[-1][0] = gx
                    trajectory[-1][1] = gy
                    # Stop timer for computation
                    end = time.perf_counter()
                    break
        elif algorithm == 'RVO':
            # Format the obstacles for RVO
            obstacleList = []
            for j in range(len(ox)):
                obstacleList.append([ox[j], oy[j], 0.5])  # (x, y, radius)
            # Define workspace model
            ws_model = dict()
            ws_model['robot_radius'] = asv_radius - 1
            ws_model['circular_obstacles'] = obstacleList
            ws_model['boundary'] = []
            # Initialize the agent
            X = [[sx, sy]]  # position of the agent
            V = copy.deepcopy(V_new)  # velocity in x and y-direction
            V_max = [0.31, 0.31]  # maximal velocity (compromis to have approximatly 10 knots in the mean)
            goal = [[gx, gy]]
            # The actual algorithm (with time measurement)
            # Simulation setup
            total_time = 10000  # maximal time-steps (s)
            step = grid_size  # simulation step
            t = 0  # time initialization
            rx, ry = [sx], [sy]  # initialize path saving
            V_list = copy.deepcopy(V)
            start = time.perf_counter()
            # Simulation starts
            while math.sqrt((X[0][0] - gx) ** 2 + (X[0][1] - gy) ** 2) > asv_radius:
                # Compute desired velocity to goal
                V_des = compute_V_des(X, goal, V_max)
                # Compute the optimal velocity to avoid collision
                V = RVO_update(X, V_des, V, ws_model)
                # Update position
                X[0][0] += V[0][0] * step
                X[0][1] += V[0][1] * step
                # Save path points
                rx.append(X[0][0])
                ry.append(X[0][1])
                V_list.append(V)
                # # Visualization (#TODO has to be adopted to the simulation animation)
                # if show_animation and t%30 == 0:
                #     visualize_traj_dynamic(ws_model, X, V, goal, time=t*step, name='RVO/data/snap%s.png'%str(t/10))
                t += 1
            # Add goal position
            rx.append(gx)
            ry.append(gy)
            # Stop timer for computation
            end = time.perf_counter()
        else: 
            print('Algorithm unknown.')
            quit()

        # Result handling (saving and plotting)
        # Format the obstacles for optional path smoothing
        obstacle_list_for_smoothing = []
        if projection:
            for j in range(len(ship_proj_x)):
                obstacle_list_for_smoothing.append((ship_proj_x[j], ship_proj_y[j], asv_radius))  # (x, y, radius)
        else:
            for j in range(len(ship_x)):
                obstacle_list_for_smoothing.append((ship_x[j], ship_y[j], asv_radius))  # (x, y, radius)
        # Save and plot the results
        if algorithm == 'A*' or algorithm == 'Dijkstra' or algorithm == 'DFS' or algorithm == 'BFS' or algorithm == 'D*'\
            or algorithm == 'RVO':
            # Reverse the resulting path for the applicable algorithms (so it's' from start to goal)
            if algorithm != 'D*' and algorithm != 'RVO':
                rx.reverse()
                ry.reverse()
            # Save resulting path
            x_results.append(rx)
            y_results.append(ry)
            # Optional smoothing of path
            path = []
            if smoothing_of_path and not local_mode:
                # Format path for smoothing
                for j in range(len(rx)):
                    path.append([rx[j], ry[j]])
                # Compute actual distance from goal in order to have enough points in the smooth path
                number_of_points = int(math.sqrt((sx - gx) ** 2 + (sy - gy) ** 2)) * 3
                # Use custom smoothing and measure the time
                start_smoothing = time.perf_counter()
                smoothed_path = smoothing(path, obstacle_list_for_smoothing, smoothing_iterations, granular_size, number_of_points)
                end_smoothing = time.perf_counter()
                # If hybrid mode then use the smoothed path for the rest of the computation
                if hybrid_mode:
                    rx = []
                    ry = []
                    for j in range(len(smoothed_path)):
                        rx.append(smoothed_path[j][0])
                        ry.append(smoothed_path[j][1])
                    x_results[-1] = copy.deepcopy(rx)
                    y_results[-1] = copy.deepcopy(ry)
                    global_rx = copy.deepcopy(rx)
                    global_ry = copy.deepcopy(ry)
        # Plot the path
        if show_animation:
            plt.title('Resulting Path')
            plt.plot(rx, ry, "-r")
            if smoothing_of_path and not local_mode:
                    # For custom smoothing
                    plt.plot([x for (x, y) in smoothed_path], [y for (x, y) in smoothed_path], '-g')
            plt.pause(0.001)  # enough time for the path
            plt.show(block=False)
            plt.pause(1)
            plt.close()
        if algorithm == 'RRT':
            # Reverse the resulting path (so it's' from start to goal)
            path.reverse()
            # Save resulting path
            paths.append(path)
            # Optional smoothing of path
            if smoothing_of_path:
                # Enrich the number of points in the path, if necessary, in order to gain a nice smoothness
                # (not sure if this works as intendend right now ...)
                enriched_path = copy.deepcopy(path)
                while len(enriched_path) < 100:
                    temp_len = len(enriched_path)
                    temp_enriched_path = []
                    # Enrich the path by adding a new point in the midst of every two consecutive points
                    for j in range(temp_len - 1):
                        temp_enriched_path.append(enriched_path[j])
                        temp_enriched_path.append([(enriched_path[j][0] + enriched_path[j + 1][0]) / 2,\
                                                    (enriched_path[j][1] + enriched_path[j + 1][1]) / 2])
                        temp_enriched_path.append(enriched_path[j + 1])
                    enriched_path = copy.deepcopy(temp_enriched_path)
                # Smooth the (enriched) path and measure the time
                start_smoothing = time.perf_counter()
                smoothed_path = path_smoothing(enriched_path, maxIter, obstacle_list_for_smoothing)
                end_smoothing = time.perf_counter()
            if not rrt_bestof:
                # Plot the path
                if show_animation:
                    plt.title('Resulting Path')
                    plt.axis(axis_measure)
                    img = plt.imread("foerde_cutout.png")
                    plt.imshow(img, extent=axis_measure)
                    plt.plot(sx, sy, "og")
                    plt.plot(gx, gy, "xb", markersize=10, markeredgewidth=2)
                    if colregs:
                        plt.plot(colreg_x, colreg_y, "o", color="orange")
                    if projection:
                        plt.plot(ship_x, ship_y, "dw")
                        plt.plot(ship_proj_x, ship_proj_y, "d", color="darkslategrey")
                    else:
                        plt.plot(ship_x, ship_y, "dk")
                    plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
                    if smoothing_of_path:
                            plt.plot([x for (x, y) in smoothed_path], [
                            y for (x, y) in smoothed_path], '-c')
                    plt.grid(True)
                    plt.pause(0.01)  # enough time for path generation
                    plt.show(block=False)
                    plt.pause(1)
                    plt.close()
        if algorithm == 'DWA':
            # Save resulting path
            trajectories.append(trajectory)
            # Plot the path
            if show_animation:
                plt.title('Resulting Path')
                plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
                plt.pause(0.0001)
                plt.show(block=False)
                plt.pause(1)
                plt.close()
    else:
        # Plot the path in the case of hybrid mode and global path already there
        if show_animation:
            plt.title('Resulting Path')
            plt.plot(global_rx, global_ry, "-r")
            plt.pause(0.001)  # enough time for the path
            plt.show(block=False)
            plt.pause(1)
            plt.close()



    """
    Measurements for evaluation
    """
    # Adapt to hybrid mode eventually
    if not hybrid_mode or not global_path_planned or (hybrid_mode and local_mode):
        # Compute and save measurements for evaluation
        # Time measurement
        runtime = end - start
        # If smoothing is active add the time for that
        if smoothing_of_path:
            runtime += end_smoothing - start_smoothing
        # Save time measurement
        times.append(runtime)
        # Save time measurement, normalised with euclidean distance from start to goal 
        times_norm.append(runtime / math.sqrt((sx - gx) ** 2 + (sy - gy) ** 2))

        # Length measurement
        length = 0.0
        if algorithm == 'A*' or algorithm == 'Dijkstra' or algorithm == 'DFS' or algorithm == 'BFS' or algorithm == 'D*'\
            or algorithm == 'RVO':
            for j in range(len(rx) - 1):
                # Calculate Euclidean distance of one path segment
                temp = math.sqrt((rx[j + 1] - rx[j]) ** 2 + (ry[j + 1] - ry[j]) ** 2)
                length += temp
        if algorithm == 'RRT':
            for j in range(len(path) - 1):
                # Calculate Euclidean distance of one path segment
                temp = math.sqrt((path[j + 1][0] - path[j][0]) ** 2 + (path[j + 1][1] - path[j][1]) ** 2)
                length += temp
        if algorithm == 'DWA':
            for j in range(len(trajectory) - 1):
                # Calculate Euclidean distance of one path segment
                temp = math.sqrt((trajectory[j + 1][0] - trajectory[j][0]) ** 2 
                    + (trajectory[j + 1][1] - trajectory[j][1]) ** 2)
                length += temp
        # Save length measurement
        lengths.append(length)
        # Save length measurement, normalised with euclidean distance from start to goal 
        lengths_norm.append(length / math.sqrt((sx - gx) ** 2 + (sy - gy) ** 2))

        # Difference measurement between two consecutive paths
        area_under_path = 0.0
        difference = None
        if algorithm == 'A*' or algorithm == 'Dijkstra' or algorithm == 'DFS' or algorithm == 'BFS' or algorithm == 'D*'\
            or algorithm == 'RVO':
            for j in range(len(rx) - 1):
                # Calculate area under one path segment
                triangle = 0.5 * abs(rx[j + 1] - rx[j]) * abs(ry[j + 1] - ry[j])
                rectangle = abs(rx[j + 1] - rx[j]) * min(ry[j +1], ry[j])
                area_under_path += (triangle + rectangle)
        if algorithm == 'RRT':
            for j in range(len(path) - 1):
                # Calculate area under one path segment
                triangle = 0.5 * abs(path[j + 1][0] - path[j][0]) * abs(path[j + 1][1] - path[j][1])
                rectangle = abs(path[j + 1][0] - path[j][0]) * min(path[j +1][1], path[j][1])
                area_under_path += (triangle + rectangle)
        if algorithm == 'DWA':
            for j in range(len(trajectory) - 1):
                # Calculate area under one path segment
                triangle = 0.5 * abs(trajectory[j + 1][0] - trajectory[j][0]) * abs(trajectory[j + 1][1] - trajectory[j][1])
                rectangle = abs(trajectory[j + 1][0] - trajectory[j][0]) * min(trajectory[j +1][1], trajectory[j][1])
                area_under_path += (triangle + rectangle)
        # Calculate the mean of the area segments
        if algorithm == 'A*' or algorithm == 'Dijkstra' or algorithm == 'DFS' or algorithm == 'BFS' or algorithm == 'D*'\
            or algorithm == 'RVO':
            area_under_path /= (len(rx) - 1)
        if algorithm == 'RRT':
            area_under_path /= (len(path) - 1)
        if algorithm == 'DWA':
            area_under_path /= (len(trajectory) - 1)
        # Save area mean and possibly difference measurement
        areas_under_paths.append(area_under_path)
        if len(areas_under_paths) > 1:
            difference = abs(areas_under_paths[-2] - areas_under_paths[-1])
            differences.append(difference)

        # Suboptimality measurement (mean deviation of waysteps from the optimal path)
        suboptimal = []
        # Compute slope of optimal path between start and goal
        slope = None
        if (gx - sx) == 0:
            slope = axis_measure[1]
        else:
            slope = (gy - sy) / (gx - sx)   
        # Save x and y-values in different variables for calculation clarity
        path_x_values = []
        path_y_values = []
        if algorithm == 'A*' or algorithm == 'Dijkstra' or algorithm == 'DFS' or algorithm == 'BFS' or algorithm == 'D*'\
            or algorithm == 'RVO':
            path_x_values = copy.deepcopy(rx)
            path_y_values = copy.deepcopy(ry)
        if algorithm == 'RRT':
            path_x_values = [i[0] for i in path]
            path_y_values = [i[1] for i in path]
        if algorithm == 'DWA':
            path_x_values = [i[0] for i in trajectory]
            path_y_values = [i[1] for i in trajectory]
        # Save start and target point in a different format for calculation clarity
        x_start = sx
        y_start = sy
        x_goal = gx
        y_goal = gy
        # Adjust calculations to direction from start to target
        if gx < sx:
            x_start = gx
            y_start = gy
            x_goal = sx
            y_goal = sy
            path_x_values.reverse()
            path_y_values.reverse()
        for j in range(1, len(path_x_values) - 1):
            # Rename x and y-value of the actual path point for clarity in the following calculations
            x_path = path_x_values[j]
            y_path = path_y_values[j]
            # Shift the origin of the coordinate system to the current start point
            x_path -= x_start
            y_path -= y_start
            # If the ideal line is parallel to the x-axis just add the shifted y-value of the path
            if slope == 0:
                suboptimal.append(y_path)
            else:
                # First calculate the y-value of the ideal path with the x-value from the planners path with the linear equation
                y_ideal = x_path * slope
                # Second calculate the x-value of the ideal path with the y-value from the planners path with the transformed
                # linear equation
                x_ideal = y_path / slope
                # Third calculate the hypotenuse of the intended right triangle
                hypotenuse = math.sqrt((x_path - x_ideal) ** 2 + (y_path - y_ideal) ** 2)
                if hypotenuse > 0:
                # Fourth calculate the catheti of the intended right triangle
                    cathetus_a = abs(x_path - x_ideal)
                    cathetus_b = abs(y_path - y_ideal)
                    # Fifth calculate p and q of the hypotenuse using Euclids cathetus theorem (transformed)
                    p = (cathetus_a ** 2) / hypotenuse
                    q = (cathetus_b ** 2) / hypotenuse
                    # Six calculate the height of the right triangle by the geometric mean theorem
                    heigth = math.sqrt(p * q)
                    # Then save the result
                    suboptimal.append(heigth)
        # Save suboptimality measurement
        suboptimality.append(mean(suboptimal))

        # Closest encounters measurement
        close = []
        if not hybrid_mode:  # hybrid mode has its own measurement down below
            if algorithm == 'A*' or algorithm == 'Dijkstra' or algorithm == 'DFS' or algorithm == 'BFS' or algorithm == 'D*'\
                or algorithm == 'RVO':  
                # Look for the closeness of encounters in the given path
                for j in range(len(rx)):
                    if projection and len(ship_proj_x) > 0:
                        for k in range(len(ship_proj_x)):
                            close.append(math.sqrt((rx[j] - ship_proj_x[k]) ** 2 + (ry[j] - ship_proj_y[k]) ** 2))
                    else:
                        for k in range(len(ship_x)):
                            close.append(math.sqrt((rx[j] - ship_x[k]) ** 2 + (ry[j] - ship_y[k]) ** 2))
            if algorithm == 'RRT':
                # Look for the closeness of encounters in the given path
                for j in range(len(path)):
                    if projection and len(ship_proj_x) > 0:
                        for k in range(len(ship_proj_x)):
                            close.append(math.sqrt((path[j][0] - ship_proj_x[k]) ** 2 + (path[j][1] - ship_proj_y[k]) ** 2))
                    else:
                        for k in range(len(ship_x)):
                            close.append(math.sqrt((path[j][0] - ship_x[k]) ** 2 + (path[j][1] - ship_y[k]) ** 2))
            if algorithm == 'DWA':
                # Look for the closeness of encounters in the given path
                for j in range(len(trajectory)):
                    if projection and len(ship_proj_x) > 0:
                        for k in range(len(ship_proj_x)):
                            close.append(math.sqrt((trajectory[j][0] - ship_proj_x[k]) ** 2 + 
                                (trajectory[j][1] - ship_proj_y[k]) ** 2))
                    else:
                        for k in range(len(ship_x)):
                            close.append(math.sqrt((trajectory[j][0] - ship_x[k]) ** 2 + (trajectory[j][1] - ship_y[k]) ** 2))
            # Save the closest 10% of encounters
            close.sort()
            target = None
            if algorithm == 'A*' or algorithm == 'Dijkstra' or algorithm == 'DFS' or algorithm == 'BFS' or algorithm == 'D*'\
                or algorithm == 'RVO':
                target = len(rx) / 10
            if algorithm == 'RRT':
                target = len(path) / 10
            if algorithm == 'DWA':
                target = len(trajectory) / 10
            closest.append(close[:int(target)])

        # Course changes measurement
        temp = []
        templist = []
        if algorithm == 'A*' or algorithm == 'Dijkstra' or algorithm == 'DFS' or algorithm == 'BFS' or algorithm == 'D*'\
            or algorithm == 'RVO':
            for j in range(len(rx) - 1):
                # Calculate the angle
                angle = math.degrees(math.atan2((ry[j + 1] - ry[j]), (rx[j + 1] - rx[j])))
                # Save course change measurement
                temp.append(angle)
                if len(temp) > 1:
                    course_change = abs(temp[-1] - temp[-2])
                    # To acommodate math.atan2(); besides, realistic angle would never be greater than 180°, 
                    # otherwise ASV could just turn the other way around
                    if course_change > 180:
                        course_change = 360 - course_change
                    templist.append(course_change)
        if algorithm == 'RRT':
            for j in range(len(path) - 1):
                # Calculate the angle
                angle = math.degrees(math.atan2((path[j + 1][1] - path[j][1]), (path[j + 1][0] - path[j][0])))
                # Save course change measurement
                temp.append(angle)
                if len(temp) > 1:
                    course_change = abs(temp[-1] - temp[-2])
                    # To acommodate math.atan2(), besides, realistic angle would never be greater than 180°, 
                    # otherwise ASV could just turn the other way around
                    if course_change > 180:
                        course_change = 360 - course_change
                    templist.append(course_change)
        if algorithm == 'DWA':
            for j in range(len(trajectory) - 1):
                # Calculate the angle
                angle = math.degrees(math.atan2((trajectory[j + 1][1] - trajectory[j][1]), 
                                (trajectory[j + 1][0] - trajectory[j][0])))
                # Save course change measurement
                temp.append(angle)
                if len(temp) > 1:
                    course_change = abs(temp[-1] - temp[-2])
                    # To acommodate math.atan2(); besides, realistic angle would never be greater than 180°, 
                    # otherwise ASV could just turn the other way around
                    if course_change > 180:
                        course_change = 360 - course_change
                    templist.append(course_change)
        # Save the biggest 10% of course changes
        templist.sort(reverse=True)
        target = None
        if algorithm == 'A*' or algorithm == 'Dijkstra' or algorithm == 'DFS' or algorithm == 'BFS' or algorithm == 'D*'\
            or algorithm == 'RVO':
            target = len(rx) / 10
        if algorithm == 'RRT':
            target = len(path) / 10
        if algorithm == 'DWA':
            target = len(trajectory) / 10
        course_changes.append(templist[:int(target)])



    """
    Result of a timestep-run an preparation for the next one
    """
    if not hybrid_mode or not global_path_planned or (hybrid_mode and local_mode):
        # Print results of the run
        print('Found goal - Run ' + str(i + 1) + ' with ' + algorithm + ' in ' + str(round(runtime, 5)) + 
                ' of length ' + str(round(length)) + '.')
        # Print the number of path points
        if algorithm == 'A*' or algorithm == 'Dijkstra' or algorithm == 'DFS' or algorithm == 'BFS' or algorithm == 'D*'\
            or algorithm == 'RVO':
            print('Path points: ' + str(len(rx)))
        if algorithm == 'RRT':
            print('Path points: ' + str(len(path)))
        if algorithm == 'DWA':
            print('Path points: ' + str(len(trajectory)))
    
    # Adapt path information and compute its length if in hybrid mode with global path active
    if hybrid_mode and global_path_planned and not local_mode:
        rx = copy.deepcopy(global_rx)
        ry = copy.deepcopy(global_ry)
        # Length measurement
        length = 0.0
        if algorithm == 'A*' or algorithm == 'Dijkstra' or algorithm == 'DFS' or algorithm == 'BFS' or algorithm == 'D*'\
            or algorithm == 'RVO':
            for j in range(len(rx) - 1):
                # Calculate Euclidean distance of one path segment
                temp = math.sqrt((rx[j + 1] - rx[j]) ** 2 + (ry[j + 1] - ry[j]) ** 2)
                length += temp

    # Move the ASV along the found path, as far as the particular run should go
    temp_length = 0.0
    run_length = length / (runs - i)
    path_result_run = []
    j = 0
    while temp_length < run_length:
        if algorithm == 'A*' or algorithm == 'Dijkstra' or algorithm == 'DFS' or algorithm == 'BFS' or algorithm == 'D*'\
            or algorithm == 'RVO':
            if j < (len(rx) - 1):
                # Calculate Euclidean distance for one path segment
                temp_length += math.sqrt((rx[j + 1] - rx[j]) ** 2 + (ry[j + 1] - ry[j]) ** 2)
                # Add path segment to run results
                path_result_run.append([rx[j], ry[j]])
                # Add last path segment if directly before goal
                if (runs - i) == 1 and j == (len(rx) - 2):
                    path_result_run.append([rx[j + 1], ry[j + 1]])
                # Set new starting point for the next run
                sx = rx[j]
                sy = ry[j]
                # Adapt path information if in hybrid mode with global path active
                # if hybrid_mode and global_path_planned:
                global_rx = copy.deepcopy(rx[j:])
                global_ry = copy.deepcopy(ry[j:])               
            else:
                break
        if algorithm == 'RRT':
            # Add whole path if it is the last run
            if (runs - i) == 1:
                path_result_run.extend(path)
                break
            else:
                if j < (len(path) - 1):
                    # Calculate Euclidean distance for one path segment
                    temp_length += math.sqrt((path[j + 1][0] - path[j][0]) ** 2 + (path[j + 1][1] - path[j][1]) ** 2)
                    # Add path segment to run results
                    path_result_run.append(path[j])
                    # Set new starting point for the next run
                    sx = path[j][0]
                    sy = path[j][1]                    
                else:
                    break
        if algorithm == 'DWA':
            if j < (len(trajectory) - 1):
                # Calculate Euclidean distance for one path segment
                temp_length += math.sqrt((trajectory[j + 1][0] - trajectory[j][0]) ** 2 
                                + (trajectory[j + 1][1] - trajectory[j][1]) ** 2)
                # Add path segment to run results
                path_result_run.append([trajectory[j][0], trajectory[j][1]])
                # Add last path segment if directly before goal
                if (runs - i) == 1 and j == (len(trajectory) - 2):
                    path_result_run.append([trajectory[j + 1][0], trajectory[j + 1][1]])
                # Set new starting point for the next run
                sx = trajectory[j][0]
                sy = trajectory[j][1]
                # Apply the last DWA parameters to the start of the next run
                dwa_start_paras = trajectory[j]                   
            else:
                break
        j += 1
    
    if algorithm == 'RVO':
        V_new = copy.deepcopy(V_list[j - 1])

    # Save the path segment of the run for overall results
    path_result.extend(path_result_run)

    # If in hybrid mode, set the flag for the global path accordingly
    if hybrid_mode and not global_path_planned:
        global_path_planned = True

    # If in hybrid mode check if security distance to other ships is fulfilled and measure closest encounters
    if hybrid_mode:
        secruity_distance_intact = True
        min_dist = 301
        for j in range(len(ship_x)):
            temp_min_dist = math.sqrt((sx - ship_x[j]) ** 2 + (sy - ship_y[j]) ** 2)
            if temp_min_dist < min_dist:
                min_dist = temp_min_dist
            # Keep track of closest encounters
            close.append(temp_min_dist)
        # Save the closest 10% of encounters
        close.sort()
        target = None
        if algorithm == 'A*' or algorithm == 'Dijkstra' or algorithm == 'DFS' or algorithm == 'BFS' or algorithm == 'D*'\
            or algorithm == 'RVO':
            target = len(rx) / 10
        if algorithm == 'RRT':
            target = len(path) / 10
        if algorithm == 'DWA':
            target = len(trajectory) / 10
        closest.append(close[:int(target)])

    # In local mode, check if the extended security distance is fulfilled, in order to leave local mode
    if hybrid_mode and local_mode:
        secruity_distance_intact = min_dist >= extend_security_distance
    # Otherwise, check for the standard security distance
    elif hybrid_mode:
        secruity_distance_intact = min_dist >= asv_radius

    # If in hybrid mode, check if the local mode should be activated (based on security distance breach)
    if hybrid_mode and not secruity_distance_intact:
        local_mode = True

    # If in hybrid mode and local mode, check if it can be deactivated (based on extended security distance)
    if hybrid_mode and local_mode and secruity_distance_intact:
        local_mode = False
        global_path_planned = False  # so a new global path is planned in the next step
        # Adapt the start position for a smooth transition to A* grid size
        # For x coordinate
        if gx >= sx:
            if round(sx / grid_size) * grid_size >= sx:
                sx = round(sx / grid_size) * grid_size
            else:
                sx = round(sx / grid_size) * grid_size + grid_size
        else:
            if round(sx / grid_size) * grid_size < sx:
                sx = round(sx / grid_size) * grid_size
            else:
                sx = round(sx / grid_size) * grid_size - grid_size
        # For y coordinate
        if gy >= sy:
            if round(sy / grid_size) * grid_size >= sy:
                sy = round(sy / grid_size) * grid_size
            else:
                sy = round(sy / grid_size) * grid_size + grid_size
        else:
            if round(sy / grid_size) * grid_size < sy:
                sy = round(sy / grid_size) * grid_size
            else:
                sy = round(sy / grid_size) * grid_size - grid_size

    # Move the other ships according to the activated scenario
    for j in range(len(ships)):
        ships[j][0][0] += ships_traj[j][i][0]  # move in x direction
        ships[j][0][1] += ships_traj[j][i][1]  # move in y direction
        ships[j][2][0] = ships_traj[j][i][0]  # update ships movement vector (x)
        ships[j][2][1] = ships_traj[j][i][1]  # update ships movement vector (y)    



"""
Overall results
"""

# Plot the overall results
plt.axis(axis_measure)
plt.grid(True)
img = plt.imread("foerde_cutout.png")
plt.imshow(img, extent=axis_measure)
# plt.plot(sx, sy, "og")
plt.plot(gx, gy, "xb", markersize=10, markeredgewidth=2)
plt.plot(ship_x, ship_y, "dk")
plt.title('Resulting Paths')
# Add all found path in distinguishable colors
if algorithm == 'A*' or algorithm == 'Dijkstra' or algorithm == 'DFS' or algorithm == 'BFS' or algorithm == 'D*'\
    or algorithm == 'RVO':
    for i in range(len(x_results)):
        color = '#' + hex(255).lstrip('0x') \
            + hex(239 - int((i + 1) * 223 / (len(x_results) + 1))).lstrip('0x') \
            + hex(239 - int((i + 1) * 223 / (len(x_results) + 1))).lstrip('0x')
        plt.plot(x_results[i], y_results[i], "-", color=color)
if algorithm == 'RRT':
    for i in range(len(paths)):
        color = '#' + hex(255).lstrip('0x') \
            + hex(239 - int((i + 1) * 223 / (len(paths) + 1))).lstrip('0x') \
            + hex(239 - int((i + 1) * 223 / (len(paths) + 1))).lstrip('0x')
        plt.plot([x for (x, y) in paths[i]], [y for (x, y) in paths[i]], "-", color=color)
if algorithm == 'DWA':
    for i in range(len(trajectories)):
        color = '#' + hex(255).lstrip('0x') \
            + hex(239 - int((i + 1) * 223 / (len(trajectories) + 1))).lstrip('0x') \
            + hex(239 - int((i + 1) * 223 / (len(trajectories) + 1))).lstrip('0x')
        plt.plot(trajectories[i][:, 0], trajectories[i][:, 1], "-", color=color)
# Plot the overall resulting path of all runs
path_res_x, path_res_y = [], []
for i in range(len(path_result)):
    path_res_x.append(path_result[i][0])
    path_res_y.append(path_result[i][1])
# path_res_x = np.unique(np.array(path_res_x), axis=0).tolist()
# path_res_y = np.unique(np.array(path_res_y), axis=0).tolist()

plt.plot(path_res_x, path_res_y, "-w", linewidth=4)
plt.plot(path_res_x, path_res_y, "-r")
plt.show()

# Reformat the Closest Encounter Measurement to prepare it for saving
dftemp = pd.DataFrame(closest)
for i in range(len(dftemp.columns)):
    dftemp.rename(columns={i: 'Closest ' + str(i + 1)}, inplace=True)

# Reformat the Course Changes Measurement to prepare it for saving
dftemp2 = pd.DataFrame(course_changes)
for i in range(len(dftemp2.columns)):
    dftemp2.rename(columns={i: 'Change ' + str(i + 1)}, inplace=True)

# Put all measurements in one dataframe
dftemp3 = pd.DataFrame({'Runtime': times, 'Runtime normalised': times_norm, 'Length': lengths, 
             'Length normalised': lengths_norm, 'Difference': differences, 'Suboptimality': suboptimality})
df = pd.concat([dftemp3, dftemp, dftemp2], axis='columns')
# Add a mean calculation for the closest 10% of encounters and one for biggest 10% of course changes 
df['Closeness Mean'] = dftemp.mean(axis='columns')
df['Course Variations Mean'] = dftemp2.mean(axis='columns')

# Save measurement results
if hybrid_mode:
    with open('measurements/' + datetime.datetime.now().strftime("%y-%m-%d_%H-%M") + ' ' + scenario + ' Hybrid', 'w') as f:
                f.write(df.to_csv())
else:
    with open('measurements/' + datetime.datetime.now().strftime("%y-%m-%d_%H-%M") + ' ' + scenario + ' ' + algorithm, 'w') as f:
                f.write(df.to_csv())

# Save resulting path
df_path = pd.DataFrame({'X': path_res_x, 'Y': path_res_y})
if hybrid_mode:
    with open('measurements/' + datetime.datetime.now().strftime("%y-%m-%d_%H-%M") + ' ' + scenario + ' Hybrid' + ' Path', 'w') as f:
                f.write(df_path.to_csv())
else:
    with open('measurements/' + datetime.datetime.now().strftime("%y-%m-%d_%H-%M") + ' ' + scenario + ' ' + algorithm + ' Path', 'w') as f:
                f.write(df_path.to_csv())
