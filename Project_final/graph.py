class Graph:

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []

    def remove_edges_from_node(self, node):
    edges_to_remove = []
    for edge in self.edges:
        if node in edge:
            edges_to_remove.append(edge)
    for edge in edges_to_remove:
        self.edges.remove(edge)

    # function to add an edge to graph

    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    # utility function used to print the solution
    def printArr(self, dist):
        print("Vertex\t\tDistance from Source")
        for i in range(self.V):
            print("{0}\t\t{1}".format(i, dist[i]))

    def BellmanFord(self, src):

        dist = [float("Inf")] * self.V
        dist[src] = 0

        for _ in range(self.V - 1):

            for u, v, w in self.graph:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        for u, v, w in self.graph:
            if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                print("Graph contains negative weight cycle")
                return

        # print all distance
        self.printArr(dist)

    def view_graph_list(self):
        return self.graph
