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

import os
import gzip
import shutil

from pathlib import Path

def write_gml(G, file_name):
    """Save a graph as a GML file.

    Parameters
    ----------
    G : igraph.Graph
        The graph to save.
    file_name : str
        Save name. It should include the extension. If it ends with .gz, the file
        is gzipped.
    """
    # If a Path object is provided as file, convert it to string
    if isinstance(file_name, Path):
        file_name = str(file_name)

    is_zipped = file_name.endswith(".gz")
    
    if is_zipped:
        file_name = file_name[:-3]

    G.write(file_name, format="gml")
    
    if is_zipped:
        with open(file_name, "rb") as f_in:
            with gzip.open(file_name + ".gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        os.remove(file_name)