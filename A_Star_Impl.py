import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

occupancy_grid = []

# Maps cost(value) from start node s to v(key)
cost_to_reach_map = {}

# Maps parent(value) to child(key)
predecessor_map = {}

# Maps cost(value) of reaching the goal from the vertex(key)
EstTotalCost_map = {}

# Queue that stores node and their costs to reach goal
vertex_priorQ = []


def recover_path(start_vertex, goal_vertex):
    res_path = [goal_vertex]

    while res_path[-1] != start_vertex:
        res_path.append(predecessor_map[res_path[-1]])
    return res_path


def get_valid_children(vertex):
    neighbors = np.array([(-1,1), (0,1), (1,1), (-1,0), (1,0), (-1,-1), (0,-1), (1,-1)])
    children_indices = vertex + neighbors
    all_children = occupancy_grid[children_indices[:, 0], children_indices[:, 1]]
    return children_indices[all_children == 1]


def euclidean_weight(vertex_1, vertex_2):
    x1, y1 = vertex_1
    x2, y2 = vertex_2
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def est_cost_calculation(start_v, finish_v):
    x1, y1 = start_v
    x2, y2 = finish_v
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def a_star_search_implementation(vertex_list, start_v, goal_v):
    for vertex_v in vertex_list:
        cost_to_reach_map[vertex_v] = float("inf")
        EstTotalCost_map[vertex_v] = float("inf")

    cost_to_reach_map[start_v] = 0
    EstTotalCost_map[start_v] = est_cost_calculation(start_v, goal_v)

    vertex_priorQ.append((EstTotalCost_map[start_v], start_v))
    print(vertex_priorQ)
    while vertex_priorQ:
        _, current_v = vertex_priorQ.pop()
        print(current_v, "and", goal_v)
        if current_v == goal_v:
            return recover_path(start_v, goal_v)

        current_v_children = get_valid_children(current_v)
        for each_child in current_v_children:
            child_ctr = cost_to_reach_map[current_v] + euclidean_weight(current_v, each_child)
            each_child = tuple(each_child.tolist())
            if child_ctr < cost_to_reach_map[each_child]:
                predecessor_map[each_child] = current_v
                cost_to_reach_map[each_child] = child_ctr
                prev_total_cost_child = EstTotalCost_map[each_child]
                EstTotalCost_map[each_child] = child_ctr + est_cost_calculation(each_child, goal_v)

                if (prev_total_cost_child, each_child) in vertex_priorQ:
                    pop_index = vertex_priorQ.index((prev_total_cost_child, each_child))
                    vertex_priorQ.pop(pop_index)
                    vertex_priorQ.append((EstTotalCost_map[each_child], each_child))
                    vertex_priorQ.sort(reverse=True)
                else:
                    vertex_priorQ.append((EstTotalCost_map[each_child], each_child))
                    vertex_priorQ.sort(reverse=True)
    return "Path not Found"


def main():
    occupancy_map_img = Image.open(r'C:\Users\akhil\All_my_codes\Northeastern_Courses\Mobile_robotics\occupancy_map.png')
    global occupancy_grid
    occupancy_grid = (np.asarray(occupancy_map_img) > 0).astype(int)

    rows, cols = occupancy_grid.shape
    vertex_list = []

    for r in range(rows):
        vertex_list += (list(zip([r]*cols, np.arange(cols))))

    path_vertices = np.array(a_star_search_implementation(vertex_list, (635,140), (350,400)))

    implot = plt.imshow(occupancy_map_img, cmap='gray')

    # put a red dot, size 40, at 2 locations:
    plt.scatter(x=path_vertices[:,1], y=path_vertices[:,0], c='r', s=4)

    plt.show()


if __name__ == "__main__":
    main()