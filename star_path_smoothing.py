"""

Smoothing of paths found by *-Algorithms.

Author: Tom Beyer


"""

import numpy as np
import math
import copy

from BezierPath.bezier_path import calc_bezier_path

# Determine all the indices of the points where the course changes.
def turning_point_indices(path):
    extreme_point_indices = [0]  # start point of the path is always included
    angles = []
    for i in range(len(path) - 1):
        # Calculate the angle
        angle = math.degrees(math.atan2((path[i + 1][1] - path[i][1]), 
                                        (path[i + 1][0] - path[i][0])))
        angles.append(angle)
        if len(angles) > 1:
            course_change = abs(angles[-1] - angles[-2])    
            # If there is a course change, save the index
            if course_change != 0:
                extreme_point_indices.append(i)
    extreme_point_indices.append(len(path) - 1)  # end point of the path is always included
    
    return extreme_point_indices


# Enrich a line by adding a point between each two neigbours
# consecutively until a certain density (number of points) is reached.
def enrich_line(start, end, min_num_of_points):
    line = [start, end]
    while len(line) < min_num_of_points / 2: 
        temp_line_len = len(line)
        enriched_line = []
        # Enrich the path by adding a new point in the midst of every two consecutive points
        for i in range(temp_line_len - 1):
            enriched_line.append(line[i])
            enriched_line.append([(line[i][0] + line[i + 1][0]) / 2, (line[i][1] + line[i + 1][1]) / 2])
            enriched_line.append(line[i + 1])
        line = copy.deepcopy(enriched_line)

    return line


# Enrich every line segment of a path with a certain number of new points per distance.
def enrich_path(path, points_per_distance):
    new_enriched_path = []
    for i in range(len(path) - 1):
        dist = math.sqrt((path[i + 1][0] - path[i][0]) ** 2 + (path[i + 1][1] - path[i][1]) ** 2)
        num_of_points = int(dist / points_per_distance)
        new_enriched_path += enrich_line(path[i], path[i + 1], num_of_points)
    # Remove eventual duplicates from the new path
    _, new_enriched_path_indices = np.unique(np.array(new_enriched_path), return_index=True, axis=0)
    new_enriched_path = np.array(new_enriched_path)[np.sort(new_enriched_path_indices)]

    return new_enriched_path.tolist()


# Check if a line (between given start and end points) collides with given obstacles.
def line_collision_check(first, second, obstacles):
    # First we have to build a line with appropriate granularity (based on the distance)
    dist = math.sqrt((second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2)
    line = enrich_line(first, second, dist)       
    
    # Second we have to check if the line collides with any obstacles by checking if the
    # Euclidean distance from any point to any obstacle is smaller than the obstacle size
    for i in range(len(line)):
        for j in range(len(obstacles)):
            if math.sqrt((line[i][0] - obstacles[j][0]) ** 2 + (line[i][1] - obstacles[j][1]) ** 2) < obstacles[j][2]:
                return False
    return True


# Performs smoothing for one corner, defined by a path with a start and end point as well as one turning point.
def smooth_one_corner(path, obstacles):
    # If there is no corner, return immediately
    if len(path) < 3:
        return path
    # Enrich the path
    path = enrich_path(path, 1)  # uses a default granular size of 1 for enrichment
    # Find the points where the course changes
    extreme_point_indices = turning_point_indices(path)
    # Build a new path
    new_path = []
    # Determine how short the shortcut can be
    temp_start = path[extreme_point_indices[1]]
    temp_end = path[extreme_point_indices[1]]
    step_counter = 1
    path_changed = True
    while (temp_start != path[extreme_point_indices[0]] or temp_end != path[extreme_point_indices[2]]) \
            and path_changed:
        path_changed = False
        if temp_start != path[extreme_point_indices[0]]:
            if extreme_point_indices[1] - step_counter >= extreme_point_indices[0] and \
                line_collision_check(path[extreme_point_indices[1] - step_counter], temp_end, obstacles):
                temp_start = path[extreme_point_indices[1] - step_counter]
                path_changed = True
        if temp_end != path[extreme_point_indices[2]]:
            if extreme_point_indices[1] + step_counter <= extreme_point_indices[2] and \
                line_collision_check(temp_start, path[extreme_point_indices[1] + step_counter], obstacles):
                temp_end = path[extreme_point_indices[1] + step_counter]
                path_changed = True
        step_counter += 1
    new_path.append(path[extreme_point_indices[0]])
    new_path.append(temp_start)
    new_path.append(temp_end)
    new_path.append(path[extreme_point_indices[2]])
    # Add the last point    
    new_path.append(path[-1])
    new_path = np.asarray(new_path)
    # Remove eventual duplicates from the new path
    _, new_path_indices = np.unique(new_path, return_index=True, axis=0)
    new_path = new_path[np.sort(new_path_indices)]
    
    return new_path.tolist()

# Performs one smoothing cycle over a given path from a *-algorithm.
def smooth_a_path(path, obstacles):
    # A straight line between two points is already optimal
    if len(path) < 3:
        return path
    # Find the points where the course changes
    extreme_point_indices = turning_point_indices(path)
    # Build the new path, starting with the first shortcut
    temp_path = []
    temp_path.extend([path[extreme_point_indices[0]], path[extreme_point_indices[1]], path[extreme_point_indices[2]]])
    new_path = smooth_one_corner(temp_path, obstacles)
    # Extend this new path over the whole length
    if len(extreme_point_indices) > 3:
        for i in range(2, len(extreme_point_indices) - 1):
            temp_path = []
            temp_path.extend([new_path[-2]] + [path[extreme_point_indices[i]], path[extreme_point_indices[i + 1]]])
            new_path = new_path[:-1] + smooth_one_corner(temp_path, obstacles)
    # Add the last point    
    new_path.append(path[-1])
    new_path = np.asarray(new_path)
    # Remove eventual duplicates from the new path
    _, new_path_indices = np.unique(new_path, return_index=True, axis=0)
    new_path = new_path[np.sort(new_path_indices)]
    
    return new_path.tolist()


# The complete smoothing of a path found by a *-algorithm in three steps.
def smoothing(path, obstacles, smoothing_iterations, granular_size_of_smoothing, number_of_points):
    # First, perform the smoothing as often as defined
    smoothed_path = smooth_a_path(path, obstacles)
    for _ in range(smoothing_iterations - 1):
        smoothed_path = smooth_a_path(smoothed_path, obstacles)
    # Second, enrich the found path based on the distances
    enriched_path = enrich_path(smoothed_path, granular_size_of_smoothing)
    # Third, compute the Bezier Curve for the enriched path
    final_path = calc_bezier_path(np.asarray(enriched_path), number_of_points)

    return final_path


# OLD VERSION !!!
# # Do the actual smoothing of a path found by a *-Algorithm.
# def smooth_a_star_path(path, obstacles):
#     if len(path) < 5:
#         return path
#     # Find the points where the course changes
#     extreme_point_indices = turning_point_indices(path)
#     new_path = []
#     # The list of turning points has to have an odd length
#     if len(extreme_point_indices) % 2 == 0:
#         extreme_point_indices = extreme_point_indices[:-1]
#     # Go through ever second turning point and try to connect its neighbours in order to shorten the path.
#     # Take obstacles into account and chose the shortcut accordingly.
#     for i in range(1, len(extreme_point_indices) - 1, 2):
#         temp_start = path[extreme_point_indices[i]]
#         temp_end = path[extreme_point_indices[i]]
#         step_counter = 1
#         path_changed = True
#         while (temp_start != path[extreme_point_indices[i - 1]] or temp_end != path[extreme_point_indices[i + 1]]) \
#                 and path_changed:
#                 # and line_collision_check(temp_start, temp_end, obstacles):
#             path_changed = False
#             if temp_start != path[extreme_point_indices[i - 1]]\
#                                   and extreme_point_indices[i] - step_counter >= extreme_point_indices[i - 1]:
#                 if line_collision_check(path[extreme_point_indices[i] - step_counter], temp_end, obstacles):
#                     temp_start = path[extreme_point_indices[i] - step_counter]
#                     path_changed = True
#             if temp_end != path[extreme_point_indices[i + 1]]\
#                                 and extreme_point_indices[i] + step_counter <= extreme_point_indices[i + 1]:
#                 if line_collision_check(temp_start, path[extreme_point_indices[i] + step_counter], obstacles):
#                     temp_end = path[extreme_point_indices[i] + step_counter]
#                     path_changed = True
#             step_counter += 1
#         new_path.append(path[extreme_point_indices[i - 1]])
#         new_path.append(temp_start)
#         new_path.append(temp_end)
#         new_path.append(path[extreme_point_indices[i + 1]])
#     # Add the last point    
#     new_path.append(path[-1])
#     # Remove eventual duplicates from the new path
#     new_path = np.asarray(new_path)
#     _, new_path_indices = np.unique(new_path, return_index=True, axis=0)
#     new_path = new_path[np.sort(new_path_indices)]

#     return new_path.tolist()


# OLD VERSION !!!
# # The complete smoothing of a path found by *-Algorithms in three steps
# def smoothing(path, obstacles, number_of_points):
#     # First, do the smoothing iteratively until the path can't be shortened any more
#     old_len = 0
#     if isinstance(path, np.ndarray):
#         path = path.tolist()
#     new_path = smooth_a_star_path(path, obstacles)
#     new_len = len(new_path)
#     while (old_len != new_len):
#         old_len = new_len
#         new_path = smooth_a_star_path(new_path, obstacles)
#         new_len = len(new_path)

#     # Second, enrich the found path based on the distances
#     enriched_path = []
#     for i in range(len(new_path) - 1):
#         distance = math.sqrt((new_path[i + 1][0] - new_path[i][0]) ** 2 + (new_path[i + 1][1] - new_path[i][1]) ** 2)
#         temp_enriched_path = enrich_line(new_path[i], new_path[i + 1], distance)
#         enriched_path += temp_enriched_path
#         # Remove eventual duplicates from the enriched path
#         _, enriched_path_indices = np.unique(np.array(enriched_path), return_index=True, axis=0)
#         enriched_path = np.array(enriched_path)[np.sort(enriched_path_indices)]
#         enriched_path = enriched_path.tolist()

#     # Third, compute the Bezier Curve for the enriched path
#     smoothed_path = calc_bezier_path(np.asarray(enriched_path), number_of_points)

#     return smoothed_path