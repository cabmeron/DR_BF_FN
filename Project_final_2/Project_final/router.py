class Router:

    def __init__(self, id, connected_nodes, routerList):
        self.id = id
        self.connected_nodes = connected_nodes
        routerList.append(self)


    def get_id(self):
        return self.id

    def get_connect_nodes(self):
        return self.connected_nodes

    def display(self):
        print(f'ID: {self.id}')
        print(f'Nodal Delay: {self._dnodal}')
        print(f'Processing Delay: {self._dproc}')
        print(f'Queue Delay: {self._dqueue}')
        print(f'Transmission Delay: {self._dtrans}')
        print(f'Propagation Delay: {self._dtrans}')
        print(f'Connected Nodes: {self.connected_nodes} \n')

    def has_edge(self, router2):
        if router2.get_id() in self.get_connect_nodes():
            return True
        else:
            return False

