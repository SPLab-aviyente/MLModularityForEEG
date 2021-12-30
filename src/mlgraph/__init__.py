from . import read

# from igraph import Graph

# class MLGraph(Graph):
    
#     def __init__(self):
#         super().__init__()

#     def add_vertex(self, name=None, **kwds):
#         if "layer" not in kwds:
#             raise Exception("Layer of the vertex must be provided.")

#         return super().add_vertex(name=name, **kwds)

#     def add_vertices(self, n, attributes=None):
#         if "layer" not in attributes:
#             raise Exception("Layers of the vertices must be provided within attributes argument.")

#         return super().add_vertices(n, attributes=attributes)

#     def add_edge(self, source, target, **kwds):
#         return super().add_edge(source, target, **kwds)