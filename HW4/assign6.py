import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

# Call Karate_Club data using networkx
G = nx.karate_club_graph()

# undirected graph, unweighted graph
idx_node = list(G.nodes)  # nodes = 0~33 # [0,1,2,...,33]
edges_list = list(G.edges)  # num of edges # [(node1, node2), (node3, node4), ... , ]


# Hyperparameter
epochs = 100
hidden = 2
learning_rate = 0.001
walk_length = 10
window_size = 3
walks_per_vertex = 5


# depth = the number of parant node + 1
# node = [0,1,2,...,last_node]
# In our case, depth = 7.
def depth(nodes):
    depth = 0
    while len(nodes) > (2**depth):
        depth += 1
    return depth + 1  # count the int number from 0 to depth


# Initialize weights
input_dim = len(idx_node)
phi = np.random.rand(input_dim, hidden)  # Weight : input to hidden
path_len = depth(idx_node) - 1  # path length is depth - 1
psi = np.random.rand(hidden, path_len)


class Node:
    def __init__(self, left=None, right=None, vec=None):
        self.vec = vec
        self.left = left
        self.right = right


# hierarchical_node = {0 : {'vec':None, 'left':None, 'right':None}, ... , 64 : {'vec':np.array(2,), 'left':0, 'right':1}, ...}
def hierarchical_node_data(hidden_dim=2, node=idx_node):
    tree_depth = depth(node)

    node_vector_data = {}
    # tree_depth = 7. layer = 6,5,4,3,2,1,0
    # leaf data. They have no vector
    for i in range(2 ** (tree_depth - 1)):
        node_vector_data[i] = {"vec": None, "left": None, "right": None}

    # 가장 왼쪽의 data. 왼쪽을 채우고, 그 레이어의 idx를 찾아 data를 채움.
    # 0, 2**6, 2**6 + 2**5, ... , 2**6 + 2**5 + ... + 2**0
    most_left_idx = 0
    prev_most_left_idx = 0
    for layer in reversed(range(tree_depth)):
        most_left_idx += 2**layer
        if layer == tree_depth - 1:
            node_vector_data[most_left_idx] = {
                "vec": np.random.rand(
                    hidden_dim,
                ),
                "left": 0,
                "right": 1,
            }
        else:
            node_vector_data[most_left_idx] = {
                "vec": np.random.rand(
                    hidden_dim,
                ),
                "left": prev_most_left_idx,
                "right": prev_most_left_idx + 1,
            }
        prev_most_left_idx += 2**layer

    # layer = 6,5,4,3,2,1
    most_left_idx = 0
    for layer in reversed(range(1, tree_depth)):
        most_left_idx += 2**layer
        # parant nods
        for i in range(most_left_idx + 1, most_left_idx + (2 ** (layer - 1))):
            node_vector_data[i] = {
                "vec": np.random.rand(
                    hidden_dim,
                ),
                "left": node_vector_data[i - 1]["right"] + 1,
                "right": node_vector_data[i - 1]["right"] + 2,
            }

    return node_vector_data


def make_random_walk(start_node, edges_list, length, num_walk=1):
    result_random_walk = []
    for _ in range(num_walk):
        one_random_walk = [start_node]
        for walk_idx in range(length - 1):
            neighbors = []
            for pair in edges_list:
                if pair[0] == one_random_walk[walk_idx]:
                    neighbors.append(pair[1])
                elif pair[1] == one_random_walk[walk_idx]:
                    neighbors.append(pair[0])
            next_node = np.random.choice(neighbors, 1)[0]
            one_random_walk.append(next_node)
        result_random_walk.append(one_random_walk)

    return result_random_walk[0]


def make_onehot(non_zero_idx, length=input_dim):
    onehot = np.zeros(length)
    onehot[non_zero_idx] = 1
    return onehot


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# make the entire paths. [[1,1,...,1], ...., [-1,-1,...,-1]].
# the length of result is 2 ** (depth-1). the element length is depth-1.
def make_full_path(depth):
    if depth == 2:
        return [[1], [-1]]

    # Recurrence function
    sub_paths = make_full_path(depth - 1)

    # In case depth >= 3
    full_paths = []
    for sub_path in sub_paths:
        left_full_path = [i for i in sub_path]  # Deep copy sub_path
        left_full_path.append(1)
        full_paths.append(left_full_path)

        right_full_path = [i for i in sub_path]
        right_full_path.append(-1)
        full_paths.append(right_full_path)
    return full_paths


# Assign path to each node
def assign_path(full_path, idx_nodes):
    np.random.shuffle(idx_nodes)
    choice_path_idx = np.random.choice(len(full_path), size=len(idx_nodes))

    # node_path = {node index : path}. eg, {0 : [1,1,1,1,1,-1]}
    node_path = {}
    # idx_nodes에서 첫번째 노드에는 full_path의 idx번째 path를 할당
    for node, idx in enumerate(choice_path_idx):
        node_path[idx_nodes[node]] = full_path[idx]
    return node_path


def loss_hierarchical_softmax(depth, one_full_path, inner_node_vector, hidden):
    loss = 0
    for i in range(depth - 1):
        loss += -np.log(
            sigmoid(one_full_path[i], np.matmul(inner_node_vector.T, hidden))
        )
    return loss


hie_data = hierarchical_node_data(hidden_dim=2, node=idx_node)


def deepwalk_hierarchical_softmax(
    walks_per_vertex,
    idx_node,
    edge_list,
    walk_length,
    window_size,
    phi,
    psi,
    learning_rate,
    paths,
    hie_data,
):
    loss_value = 0

    for _ in range(walks_per_vertex):
        np.random.shuffle(idx_node)
        for node in idx_node:
            random_walk = make_random_walk(node, edge_list, walk_length)
            phi, psi, loss_value = hierarchical_softmax(
                random_walk,
                window_size,
                phi,
                psi,
                learning_rate,
                loss_value,
                paths,
                hie_data,
            )

    loss_value += loss_value / len(idx_node)

    return phi, psi, loss_value


def hierarchical_softmax(
    random_walk, window_size, phi, psi, learning_rate, loss, paths, hie_data
):
    temp_loss = 0
    for idx, target in enumerate(random_walk):
        target_onehot = make_onehot(target)
        # 이게 뭐하는 짓이지 34길이의 벡터에서 v_j의 위치만 1인 행렬 만들기
        ## def make_onehot(non_zero_idx, length=input_dim):
        # onehot = np.zeros(length)
        # onehot[non_zero_idx] = 1
        # return onehot

        # neighbors of target node
        neighbors = random_walk[
            max(0, idx - window_size) : min(len(random_walk), idx + window_size + 1)
        ]
        neighbors.remove(random_walk[idx])
        for neighbor in neighbors:
            # Forward : input to hidden
            hidden = np.matmul(phi.T, target_onehot)

            # Forward : hidden to output. hierarchical softmax
            neighbor_path = paths[neighbor]
            pred = 1
            start = max(hie_data.keys())
            vec = hie_data[start]["vec"]
            update_num_vec = [start]
            for direction in neighbor_path:
                # path direction * the i th column of psi * hidden
                pred *= sigmoid(direction * (np.matmul(vec.T, hidden)))
                temp_loss += -np.log(sigmoid(direction * (np.matmul(vec.T, hidden))))
                if direction == 1:
                    left_idx = hie_data[start]["left"]
                    update_num_vec.append(left_idx)
                    vec = hie_data[left_idx]["vec"]
                if direction == -1:
                    right_idx = hie_data[start]["right"]
                    update_num_vec.append(right_idx)
                    vec = hie_data[right_idx]["vec"]
            update_num_vec = update_num_vec[:-1]

            # Calcurate loss
            loss = temp_loss

            # Backward
            new_psi = np.zeros(hidden.shape)
            grad_phi = 0
            _EH = 0
            for i in range(psi.shape[1]):
                if neighbor_path[i] == 1:
                    t = 1
                else:
                    t = 0

                hie_data[update_num_vec[i]]["vec"] = hie_data[update_num_vec[i]]["vec"] / sum(hie_data[update_num_vec[i]]["vec"])
                phi[target, :] = phi[target, :] / sum(phi[target, :])

                temp = sigmoid(np.matmul(hie_data[update_num_vec[i]]["vec"].T, phi[target, :])) - t
                _EH = _EH + temp * hie_data[update_num_vec[i]]["vec"]
                hie_data[update_num_vec[i]]["vec"] -= learning_rate * temp * phi[target, :]

            #     new_psi = hie_data[update_num_vec[i]]["vec"] - learning_rate * (
            #         sigmoid(np.matmul(hie_data[update_num_vec[i]]["vec"].T, hidden)) - t
            #     )
            #     grad_phi += (
            #         sigmoid(np.matmul(hie_data[update_num_vec[i]]["vec"].T, hidden)) - t
            #     ) * hie_data[update_num_vec[i]]["vec"]
            # phi[target, :] -= learning_rate * grad_phi
            # hie_data[update_num_vec[i]]["vec"] = new_psi

            phi[target, :] -= learning_rate * _EH

        return phi, psi, loss


depth = depth(idx_node)
full_path = make_full_path(depth)
paths = assign_path(full_path, idx_node)

loss_list = []
for i in range(epochs):
    start_time = time.time()

    new_phi, new_psi, loss_value = deepwalk_hierarchical_softmax(
        walks_per_vertex,
        idx_node,
        edges_list,
        walk_length,
        window_size,
        phi,
        psi,
        learning_rate,
        paths,
        hie_data,
    )
    phi = new_phi
    psi = new_psi
    loss_list.append(loss_value)

    end_time = time.time()

loss_list = [i / (len(idx_node) ** 2) for i in loss_list]
print(loss_list)
print("time:", end_time - start_time)


# Plot loss
plt.plot(loss_list)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


# Plot label 0, 1 # Blue, Red
instructor = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17, 19, 21]  # label 1
club_admin = [8,9,14,15,18,20,22,23,24,25,26,27,28,29,30,31,32,33]  # label 0

z_embedded = phi

plt.figure()
i = 0
for i in range(len(idx_node)):
    if instructor.count(i) > 0:
        plt.scatter(z_embedded[i, 0], z_embedded[i, 1], c="crimson")
    else:
        plt.scatter(z_embedded[i, 0], z_embedded[i, 1], c="royalblue")
    plt.text(z_embedded[i, 0], z_embedded[i, 1], i + 1, fontsize=10)
plt.show()


# Perozzi et al., KDD'14
purple = [9, 15, 16, 19, 21, 23, 24, 27, 30, 31, 33, 34]
green = [3, 10, 25, 26, 28, 29, 32]
red = [1, 2, 4, 8, 12, 13, 14, 18, 20, 22]
blue = [5, 6, 7, 11, 17]

z_embedded = phi

plt.figure()
i = 0
for i in range(len(idx_node)):
    if purple.count(i + 1) > 0:
        plt.scatter(z_embedded[i, 0], z_embedded[i, 1], c="darkorchid")
    elif green.count(i + 1) > 0:
        plt.scatter(z_embedded[i, 0], z_embedded[i, 1], c="greenyellow")
    elif red.count(i + 1) > 0:
        plt.scatter(z_embedded[i, 0], z_embedded[i, 1], c="indianred")
    elif blue.count(i + 1) > 0:
        plt.scatter(z_embedded[i, 0], z_embedded[i, 1], c="turquoise")

    plt.text(z_embedded[i, 0], z_embedded[i, 1], i + 1)
plt.show()

