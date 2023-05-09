import router
import sys
import numpy as np
import random
import time
import graph
import matplotlib.pyplot as plt
import networkx as nx


random.seed()


def main():

    edges = []
    numRouters1 = 21
    # print('\n', '\n')
    # print(' ---------------------------------------------')
    # print('|Dynamic Routing with Node Failure Simulation:|')
    # print(' ---------------------------------------------')
    # print('\n')
    # while numRouters1 < 1 or numRouters1 > 21:
    #     numRouters1 = int(input(
    #         "Enter the number of routers in desired network (1 - 21): "))
    #     if numRouters1 < 1 or numRouters1 > 21:
    #         print("Invalid Number of Routers (1 to 21)")

    # print('\n')
    source_router = 0

    routerList = []
    routerList2 = []

    adjacencyMatrix = np.zeros(shape=(numRouters1, numRouters1), dtype=int)

    numRouters = adjacencyMatrix.shape[0]

    # for r in range(numRouters1):
    #     router.Router(r + 1, 1, 1, 1, 1, 1, random.randint(0, 1), list(range(1, numRouters1, 1)),
    #                   routerList2, [random.uniform(0, 17.17), random.uniform(0, 24.7)])

    # for r in range(len(routerList2)):
    #     routerList2[r].display()

    r1 = router.Router(1, 0, 0, 0, 0, 0, 0, [2, 3], routerList, [3.80, 10.036])
    r2 = router.Router(2, 0, 0, 0, 0, 0, 0, [
                       3, 4], routerList, [5.60, 11, 580])
    r3 = router.Router(3, 0, 0, 0, 0, 0, 0, [
                       1, 2, 5, 6, 7], routerList, [6.30, 9.50])
    r4 = router.Router(4, 0, 0, 0, 0, 0, 0, [2, 8], routerList, [7.79, 11.350])
    r5 = router.Router(5, 0, 0, 0, 0, 0, 0, [
                       3, 6, 7], routerList, [8.718, 9.1])
    r6 = router.Router(6, 0, 0, 0, 0, 0, 0, [
                       3, 5, 7, 13, 17], routerList, [9.2, 6.850])
    r7 = router.Router(7, 0, 0, 0, 0, 0, 0, [
                       3, 5, 6, 8, 9, 10, 12, 13], routerList, [9.90, 11.350])
    r8 = router.Router(8, 0, 0, 0, 0, 0, 0, [
                       4, 7, 9], routerList, [9.50, 15.3])
    r9 = router.Router(9, 0, 0, 0, 0, 0, 0, [
                       7, 8, 10, 11], routerList, [11.410, 14.668])
    r10 = router.Router(10, 0, 0, 0, 0, 0, 0, [
                        7, 9, 11, 12, 14], routerList, [12.950, 11.950])
    r11 = router.Router(11, 0, 0, 0, 0, 0, 0, [
                        9, 10, 14], routerList, [13.710, 14.282])
    r12 = router.Router(12, 0, 0, 0, 0, 0, 0, [
                        7, 10, 13], routerList, [12.340, 9.70])
    r13 = router.Router(13, 0, 0, 0, 0, 0, 0, [
                        6, 7, 12, 15], routerList, [11.150, 8.492])
    r14 = router.Router(14, 0, 0, 0, 0, 0, 0, [
                        10, 11, 15], routerList, [14.910, 11.712])
    r15 = router.Router(15, 0, 0, 0, 0, 0, 0, [
                        13, 16, 18], routerList, [13.710, 7.56])
    r16 = router.Router(16, 0, 0, 0, 0, 0, 0, [
                        15, 17, 18], routerList, [11.701, 5.672])
    r17 = router.Router(17, 0, 0, 0, 0, 0, 0, [
                        6, 16, 19, 20], routerList, [10.610, 5.350])
    r18 = router.Router(18, 0, 0, 0, 0, 0, 0, [
                        15, 16, 21], routerList, [13.412, 5.190])
    r19 = router.Router(19, 0, 0, 0, 0, 0, 0, [
                        17, 18, 20, 21], routerList, [11.701, 3.795])
    r20 = router.Router(20, 0, 0, 0, 0, 0, 0, [
                        17, 19, 21], routerList, [10.610, 1.930])
    r21 = router.Router(21, 0, 0, 0, 0, 0, 0, [
                        18, 19, 20], routerList, [13.412, 2, 316])
    # for r in range(len(routerList)):
    #     routerList[r].display()

    for r in range(len(routerList)):
        for j in range(len(routerList)):
            if routerList[r].has_edge(routerList[j]):
                adjacencyMatrix[r, j] = 1

    newAdjacencyMatrix = adjacencyMatrix.copy()

    og_graph = graph.Graph(numRouters1)
    og_indices = np.argwhere(newAdjacencyMatrix == 1)

    for i in range(og_indices.shape[0]):
        edge_weight = random.randint(1, 15)
        row = og_indices[i][0]
        col = og_indices[i][1]
        og_graph.addEdge(row, col, edge_weight)
        edges.append(edge_weight)

    print('\n')
    print(
        f"Current Network Shortest Paths from source router #{source_router} below")
    print('\n')
    og_graph.BellmanFord(source_router)
    print("\n")
    print(f"All {numRouters1} network routers currently active!")
    print('\n')

    did_failure_occur = random.randint(0, 100)
    while did_failure_occur > 5:
        did_failure_occur = random.randint(0, 100)
        t = time.localtime()
        current_time = time.strftime("%H: %M: %S", t)
        print(f'Current time: {current_time}')
        print('\n')

    failed_node1 = 1
    failed_node2 = 2
    while abs(failed_node1 - failed_node2) == 1 or abs(failed_node1 - failed_node2) == 0:
        failed_node1 = random.randrange(0, numRouters1)
        failed_node2 = random.randrange(0, numRouters1)

    real_node_1 = failed_node1 + 1
    real_node_2 = failed_node2 + 1

    print(f"Nodes #{failed_node1} and #{failed_node2} have failed!")
    print('\n')
    print("Updating routing...")
    print('\n')

    for r in range(len(routerList)):
        if routerList[r].has_edge(routerList[failed_node1]):
            newAdjacencyMatrix[r, failed_node1] = 0
            newAdjacencyMatrix[failed_node1, r] = 0

    for r in range(len(routerList)):
        if routerList[r].has_edge(routerList[failed_node2]):
            newAdjacencyMatrix[r, failed_node2] = 0
            newAdjacencyMatrix[failed_node2, r] = 0

    dist = np.linalg.norm(adjacencyMatrix - newAdjacencyMatrix)

    g = graph.Graph(adjacencyMatrix.shape[0])

    indices = (np.argwhere(newAdjacencyMatrix == 1))

    for i in range(indices.shape[0]):
        row = indices[i][0]
        col = indices[i][1]
        g.addEdge(row, col, edges[i])

    print(
        f"New routing paths identified from source router #{source_router} below")
    print('\n')
    g.BellmanFord(source_router)
    print('\n')

    numRouter2 = 21
    G = nx.Graph()
    G.add_nodes_from(range(numRouter2))
    for i in range(adjacencyMatrix.shape[0]):
        for j in range(adjacencyMatrix.shape[1]):
            if adjacencyMatrix[i][j] > 0:
                G.add_edge(i, j, weight=edges[i])

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

    # positions for all nodes - seed for reproducibility
    pos = nx.spring_layout(G, seed=7)

    plt.figure(figsize=(15, 15))

# nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

# edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )

# node labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
# edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    G2 = nx.Graph()
    G2.add_nodes_from(range(numRouter2))
    for i in range(newAdjacencyMatrix.shape[0]):
        for j in range(newAdjacencyMatrix.shape[1]):
            if newAdjacencyMatrix[i][j] > 0:
                G2.add_edge(i, j, weight=edges[i])

    elarge = [(u, v) for (u, v, d) in G2.edges(data=True) if d["weight"] > 0.5]
    esmall = [(u, v)
              for (u, v, d) in G2.edges(data=True) if d["weight"] <= 0.5]

    # positions for all nodes - seed for reproducibility

    plt.figure(figsize=(25, 25))


# nodes
    nx.draw_networkx_nodes(G2, pos, node_size=700)


# edges
    nx.draw_networkx_edges(G2, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(
        G2, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
    )


# node labels
    nx.draw_networkx_labels(G2, pos, font_size=20, font_family="sans-serif")


# edge weight labels
    edge_labels = nx.get_edge_attributes(G2, "weight")
    nx.draw_networkx_edge_labels(G2, pos, edge_labels, font_size=20)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


#     G2 = nx.Graph()
#     G2.add_nodes_from(range(numRouter2))
#     for i in range(newAdjacencyMatrix.shape[0]):
#         for j in range (newAdjacencyMatrix.shape[1]):
#             if newAdjacencyMatrix[i][j] > 0:
#                 G2.add_edge(i, j, weight = edges[i])

#     elarge = [(u, v) for (u, v, d) in G2.edges(data=True) if d["weight"] > 0.5]
#     esmall = [(u, v) for (u, v, d) in G2.edges(data=True) if d["weight"] <= 0.5]


#       # positions for all nodes - seed for reproducibility

#     plt.figure(figsize=(25,25))


# # nodes
#     nx.draw_networkx_nodes(G2, pos, node_size=700)


# # edges
#     nx.draw_networkx_edges(G2, pos, edgelist=elarge, width=6)
#     nx.draw_networkx_edges(
#     G2, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
# )


# # node labels
#     nx.draw_networkx_labels(G2, pos, font_size=20, font_family="sans-serif")


# # edge weight labels
#     edge_labels = nx.get_edge_attributes(G2, "weight")
#     nx.draw_networkx_edge_labels(G2, pos, edge_labels,font_size = 20)

#     ax = plt.gca()
#     ax.margins(0.08)
#     plt.axis("off")
#     plt.tight_layout()
#     plt.show()


if __name__ == '__main__':
    main()
