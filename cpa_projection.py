"""

Compute projections based on CPA.

Author: Tom Beyer

"""



import math
import copy

# Build obstacles
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

# Build boundaries out of individual obstacles
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

# Calculate the coordinates of a vessels hull, given it's centre point, size and direction
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

# Determine the other vessels projections based on CPA.
def cpa_projection(asv_position, asv_goal, asv_speed, security_distance, ships):
    
    # Build the individual ship obstacles
    ship_obstacles = []
    for i in range(len(ships)):
        x_bow, y_bow, x_stern, y_stern = position_ship(ships[i][0], ships[i][1], ships[i][2])
        x_obstacle, y_obstacle = build_obstacle_line(x_bow, y_bow, x_stern, y_stern)
        temp_ship_obstacles = []
        for j in range(len(x_obstacle)):
            temp_ship_obstacles.append([x_obstacle[j], y_obstacle[j]])
        ship_obstacles.append(temp_ship_obstacles)

    # Compute the CPA with respect to the actual velocities
    cpa_projections = []
    # Determine ASV movement steps
    asv_move_vector_magnitude = math.sqrt((asv_goal[0] - asv_position[0]) ** 2 + (asv_goal[1] - asv_position[1]) ** 2)
    asv_move_unit_vector = [(asv_goal[0] - asv_position[0]) / asv_move_vector_magnitude, \
                            (asv_goal[1] - asv_position[1]) / asv_move_vector_magnitude]
    asv_move = [asv_move_unit_vector[0] * asv_speed, asv_move_unit_vector[1] * asv_speed]
    for i in range(len(ships)):
        temp_min = 301
        temp_cpa_projection = []
        asv_pos = copy.deepcopy(asv_position)
        ship_pos = copy.deepcopy(ships[i][0])
        # Determine ships movement steps
        ship_move_vector_magnitude = math.sqrt(ships[i][2][0] ** 2 + ships[i][2][1] ** 2)
        ship_move_unit_vector = [ships[i][2][0] / ship_move_vector_magnitude, ships[i][2][1] / ship_move_vector_magnitude]
        ship_move = [ship_move_unit_vector[0] * ships[i][3], ship_move_unit_vector[1] * ships[i][3]]
        # Search for minimum distance until area boundaries are reached or distances don't get smaller anymore with every step
        old_min = 302
        while asv_pos[0] >= 0 and asv_pos[0] <= 300 and asv_pos[1] >= 0 and asv_pos[1] <= 300 and \
                ship_pos[0] >= 0 and ship_pos[0] <= 300 and ship_pos[1] >= 0 and ship_pos[1] <= 300 and \
                temp_min < old_min:
            old_min = temp_min  # already preparation for the next round of while
            # Calculate the distances to all obstacles in order to find the minimum
            for j in range(len(ship_obstacles[i])):
                dist = math.sqrt((asv_pos[0] - ship_obstacles[i][j][0]) ** 2 + (asv_pos[1] - ship_obstacles[i][j][1]) ** 2)
                if dist < temp_min:
                    temp_min = dist
                    temp_cpa_projection = copy.deepcopy(ship_pos)
            # Update positions
            asv_pos[0] += asv_move[0]
            asv_pos[1] += asv_move[1]
            ship_pos[0] += ship_move[0]
            ship_pos[1] += ship_move[1]
            for j in range(len(ship_obstacles[i])):
                ship_obstacles[i][j][0] += ship_move[0]
                ship_obstacles[i][j][1] += ship_move[1]
        # Check if the closest point of approach is critical
        if temp_min < security_distance:
            cpa_projections.append(temp_cpa_projection)
        else:
            cpa_projections.append([999, 999])

    return cpa_projections