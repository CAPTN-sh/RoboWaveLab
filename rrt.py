"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT) (modified by Tom Beyer)

author: AtsushiSakai(@Atsushi_twi)

"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np

show_animation = True


class RRT:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    class AreaBounds:

        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])


    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 extra_rand_boundaries=None,  # for simulating foerde shores
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=100000,  # modified
                 play_area=None,
                 robot_radius=0.0,
                 steering_angle=15.0,  # for limiting the steering angle
                 # Modified for distinct plotting of ships and ship projections or COLREG (optional)
                 ships=[],
                 ships_proj=[],
                 ship_x=[],
                 ship_y=[],
                 ship_proj_x=[],
                 ship_proj_y=[],
                 colreg_x=[],
                 colreg_y=[]  
                 ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        extra_rand_boundaries: Limit the sampling area in a special way (foerde shores)
        play_area:stay inside this area [xmin,xmax,ymin,ymax]
        robot_radius: robot body modeled as circle with given radius

        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.extra_rand_boundaries = extra_rand_boundaries  # for simulating foerde shores
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.robot_radius = robot_radius
        self.steering_angle = steering_angle  # for limiting the steering angle
        # Modified for distinct plotting of ships and ship projections or COLREG (optional)
        self.ships = ships
        self.ships_proj = ships_proj
        self.ship_x = ship_x
        self.ship_y = ship_y
        self.ship_proj_x = ship_proj_x
        self.ship_proj_y = ship_proj_y
        self.colreg_x = colreg_x
        self.colreg_y = colreg_y

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # Modified
            # Limit the steering angle by samlping anew until it is appropriate
            while not self.is_angle_appropriate(nearest_node, rnd_node, self.steering_angle):
                rnd_node = self.get_random_node()
                nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
                nearest_node = self.node_list[nearest_ind]                

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_if_outside_play_area(new_node, self.play_area) and \
               self.check_collision(
                   new_node, self.obstacle_list, self.robot_radius):
                self.node_list.append(new_node)

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision(
                        final_node, self.obstacle_list, self.robot_radius):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(rnd_node)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            x_rand = random.uniform(self.min_rand, self.max_rand)
            # Modified to limit the sampling space to the play area (for speed reasons)
            y_rand = random.uniform(self.play_area.ymin, self.play_area.ymax)
            # Modified to respect the foerde shores
            if self.extra_rand_boundaries != None:
                x_east, x_west = [], []
                for i in range(len(self.extra_rand_boundaries[0])):
                    if self.extra_rand_boundaries[0][i][1] == round(y_rand):
                        x_east.append(self.extra_rand_boundaries[0][i][0])
                for i in range(len(self.extra_rand_boundaries[1])):        
                    if self.extra_rand_boundaries[1][i][1] == round(y_rand):
                        x_west.append(self.extra_rand_boundaries[1][i][0])
                if min(x_east) >= x_rand or max(x_west) <= x_rand:
                    x_rand = random.uniform(max(x_east) + 1, min(x_west) - 1)
            rnd = self.Node(x_rand, y_rand)
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        # Modified for plotting ships (and optional projections)
        if self.colreg_x:
            plt.plot(self.colreg_x, self.colreg_y, "o", color="orange")
        if not self.ships_proj:
            # Normal plot
            plt.plot(self.ship_x, self.ship_y, "dk")
        else:
            # Plot with projections
            x_proj, y_proj = [], []
            for j in range(len(self.ships)):
                x_proj = [self.ships[j][0][0], self.ships_proj[j][0][0]]
                y_proj = [self.ships[j][0][1], self.ships_proj[j][0][1]]
                plt.plot(x_proj, y_proj, ":w")
            plt.plot(self.ship_x, self.ship_y, "dw")
            plt.plot(self.ship_proj_x, self.ship_proj_y, "d", color="darkslategrey")

        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
            if self.robot_radius > 0.0:
                self.plot_circle(rnd.x, rnd.y, self.robot_radius, '-r')
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        # for (ox, oy, size) in self.obstacle_list:
            # self.plot_circle(ox, oy, size)
            # plt.plot(ox, oy, "dk")  # modified
        

        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin],
                     "--w", linewidth=1)  # modified

        # plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.start.x, self.start.y, "og")  # modified
        # plt.plot(self.end.x, self.end.y, "xr")
        plt.plot(self.end.x, self.end.y, "xb", markersize=10, markeredgewidth=2)  # modified
        plt.annotate('ASV', xy=(self.start.x, self.start.y), xytext=(self.start.x - 40, self.start.y - 40), 
            arrowprops=dict(facecolor='black', shrink=0.05),)  # modified
        # plt.axis("equal")
        # plt.axis([-2, 15, -2, 15])
        plt.axis([0, 300, 0, 300])  # modified for foerde simulation measures
        plt.grid(True)
        img = plt.imread("foerde_cutout.png")
        plt.imshow(img, extent=[0, 300, 0, 300])
        plt.pause(0.01)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):

        if play_area is None:
            return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def check_collision(node, obstacleList, robot_radius):

        if node is None:
            return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= (size+robot_radius)**2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    # Modified
    # A new function for limiting the steering angle
    @staticmethod
    def is_angle_appropriate(from_node, to_node, steering_angle):
        if from_node.parent is None:
            return True
        dx_1 = to_node.x - from_node.x
        dy_1 = to_node.y - from_node.y
        angle_1 = math.degrees(math.atan2(dy_1, dx_1))
        dx_2 = from_node.x - from_node.parent.x
        dy_2 = from_node.y - from_node.parent.y
        angle_2 = math.degrees(math.atan2(dy_2, dx_2))
        diff = abs(angle_1 - angle_2)
        # Limit angle
        if diff <= steering_angle:
            return True
        return False



def main(gx=6.0, gy=10.0):
    print("start " + __file__)

    # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    # Set Initial parameters
    rrt = RRT(
        start=[0, 0],
        goal=[gx, gy],
        rand_area=[-2, 15],
        obstacle_list=obstacleList,
        play_area=[0, 10, 0, 14],
        robot_radius=0.8
        )
    path = rrt.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()


if __name__ == '__main__':
    main()
