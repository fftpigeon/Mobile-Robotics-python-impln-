import numpy as np
import networkx as nx
from PIL import Image
from bresenham import bresenham
import matplotlib.pyplot as plt

occupancy_grid = []
Occupancy_grid_map = {}
V = []
E = []
G = nx.Graph
X_LEN = 0
Y_LEN = 0


def euclidean_distance(vertex_1, vertex_2):
    x1, y1 = vertex_1
    x2, y2 = vertex_2
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def rejection_sampling():
    global V, Y_LEN, X_LEN, Occupancy_grid_map
    x = np.random.randint(0, X_LEN)
    y = np.random.randint(0, Y_LEN)
    if (x, y) in V or Occupancy_grid_map[(x, y)] == 0:
        x, y = rejection_sampling()
        return x, y
    else:
        V.append((x, y))
        return x, y


def path_plan_attempt(vertex1, vertex2):
    global occupancy_grid
    x1, y1 = vertex1
    x2, y2 = vertex2
    points_on_line = np.array(list(bresenham(x1, y1, x2, y2)))[::2]
    points_val = occupancy_grid[points_on_line[:,0], points_on_line[:,1]]
    if 0 in points_val:
        return False
    else:
        return True


def AddVertex(prm_graph, dmax, vertex_x, vertex_y, N):
    global V
    exit_condition = 0
    index = 0
    print(len(prm_graph.nodes))
    while index != N:
        x_new, y_new = vertex_x[index], vertex_y[index]
        index += 1
        v_new = (x_new, y_new)

        list_nodes = list(prm_graph.nodes)
        for node in list_nodes:
            euc_weight = euclidean_distance(prm_graph.nodes[node]['pos'], v_new)
            if euc_weight <= dmax:
                if path_plan_attempt(prm_graph.nodes[node]['pos'], v_new):
                    if len(list_nodes)+1 in prm_graph.nodes:
                        prm_graph.add_edge(len(list_nodes)+1, node, weight=euc_weight)
                    else:
                        prm_graph.add_node(len(list_nodes)+1, pos=v_new)
                        prm_graph.add_edge(len(list_nodes)+1, node, weight=euc_weight)


def ConstructPRM(prm_graph, N, dmax):
    vertex_x, vertex_y = [], []
    for k in range(N):
        x_new, y_new = rejection_sampling()
        vertex_x.append(x_new)
        vertex_y.append(y_new)
    AddVertex(prm_graph, dmax, vertex_x, vertex_y, N)


def main():
    occupancy_map_img = Image.open(r'C:\Users\akhil\All_my_codes\Northeastern_Courses\Mobile_robotics\occupancy_map.png')
    global occupancy_grid, X_LEN, Y_LEN, Occupancy_grid_map, V
    occupancy_grid = (np.asarray(occupancy_map_img) > 0).astype(int)
    rows, cols = occupancy_grid.shape

    for r in range(rows):
        for c in range(cols):
            Occupancy_grid_map[(r,c)] = occupancy_grid[r, c]

    vertex_list = []

    for r in range(rows):
        vertex_list += (list(zip([r]*cols, np.arange(cols))))

    plt.imshow(occupancy_map_img, cmap='gray')

    start_vertex = (635, 140)
    goal_vertex = (350, 400)
    X_LEN = rows
    Y_LEN = cols
    no_of_samples = 2500
    dmax = 75
    prm_graph = nx.Graph()

    prm_graph.add_node(1, pos=start_vertex)
    prm_graph.add_node(2, pos=goal_vertex)

    ConstructPRM(prm_graph, no_of_samples, dmax)
    positions = {}
    for i in prm_graph.nodes:
        positions[i] = (prm_graph.nodes[i]['pos'][1], prm_graph.nodes[i]['pos'][0])


    astar_path_v = nx.astar_path(prm_graph, 1, 2)
    # V = np.asarray(V)
    # plt.scatter(V[:, 1], V[:, 0], color='b', s=2)

    path_loc = []
    for i in astar_path_v:
        path_loc.append(prm_graph.nodes[i]['pos'])

    path_loc = np.array(path_loc)
    # options = {"node_size": 300, "node_color": "red"}
    # nx.draw_networkx(prm_graph, positions, with_labels=False, **options)

    # plt.show()
    
    nx.draw(prm_graph, positions, node_size=1, node_color="blue", edge_color="orange")
    plt.plot(path_loc[:, 1], path_loc[:, 0], color='r')
    plt.show()


if __name__ == "__main__":
    main()
