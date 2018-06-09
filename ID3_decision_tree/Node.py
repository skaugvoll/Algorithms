
class Node:
    def __init__(self, test):
        self.test = None  # corresponds to attribute
        self.edges = []
        self.childen = []

    def _add_edge(self, edge):
        self.edges.append(edge)

    def _add_child(self, child):
        self.childen.append(child)

    def add_branch(self, edge, child):
        self._add_edge(edge)
        self._add_child(child)