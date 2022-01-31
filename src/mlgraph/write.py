import os
import gzip
import shutil

def to_gml(G, file_name):
    file_name += ".gml"
    G.write(file_name, format="gml")

def to_zipped_gml(G, file_name):
    file_name += ".gml"
    G.write(file_name, format="gml")
    
    with open(file_name, "rb") as f_in:
        with gzip.open(file_name + ".gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(file_name)