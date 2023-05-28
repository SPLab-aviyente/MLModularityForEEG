# MLModularityForEEG - a python package for multilayer community detection
# Copyright (C) 2023 Abdullah Karaaslanli <evdilak@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import gzip
import shutil
import os

from pathlib import Path

import igraph as ig

def read_gml(file_name):
    """Read a graph from a GML file.

    Parameters
    ----------
    file_name : str
        File to read. The file can also be gzipped file (ends with .gz).

    Returns
    -------
    G : igraph.Graph
        Read graph.
    """

    # If a Path object is provided as file, convert it to string
    if isinstance(file_name, Path):
        file_name = str(file_name)

    is_zipped = file_name.endswith(".gz")

    # create a temporary gml file to be able to read with igraph
    if is_zipped:
        with gzip.open(file_name, 'rb') as f_in:
            file_name = file_name[:-3]
            with open(file_name, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    
    G = ig.read(file_name, format="gml") # read the graph

    if is_zipped:
        os.remove(file_name) # remove the temporary gml file

    return G 