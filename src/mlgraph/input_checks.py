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
