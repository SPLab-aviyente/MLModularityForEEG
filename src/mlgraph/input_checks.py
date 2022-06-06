import igraph as ig

def _layer(layer, layers):
    if layer is not None:
        if layer not in layers:
            raise Exception("{} is not a valid layer.".format(layer))

def _edge_attribute(attr, attrs):
    if attr is not None:
        if attr not in attrs:
            raise ValueError("{} is not a valid edge attribute.".format(attr))
        
def _layer_pairs(layer1, layer2):
    if (layer1 is None) and (layer2 is not None):
        raise Exception("Paremeter layer1 cannot be None, when layer2 is given.")