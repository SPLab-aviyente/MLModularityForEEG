import gzip
import shutil
import os

import igraph as ig

def from_gml(file_name):
    # TODO: check if the read graph is multilayer?
    return ig.read(file_name, format="gml")

def from_zipped_gml(file_name):
    # TODO: Is there a way to do this without creating a temprary file?

    # create a temporary gml file to be able to read with igraph
    with gzip.open(file_name, 'rb') as f_in:
        with open(file_name[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    G = from_gml(file_name[:-3]) # read the graph
    os.remove(file_name[:-3]) # remove the temporary gml file

    return G 