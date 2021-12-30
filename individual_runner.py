import os
import argparse

import yaml

from src import mlgraph

def get_config_file():
    # get the path to config file
    parser = argparse.ArgumentParser(
        description="Find the community structures of a set of multilayer networks."
        )

    parser.add_argument("--config", help="path to the configuration file", required=True)
    
    args = parser.parse_args()
    config_file = args.config

    return config_file

def get_inputs(config_file):
    # parse input arguments from the config file
    # TODO: Input Check
    with open(config_file, 'r') as conf:
        inputs = yaml.load(conf, Loader=yaml.CLoader)

    return inputs

def read_networks(inputs):
    input_dir = inputs["input_dir"]
    networks_dir = inputs["networks_dir"]

    # File names (without extension) of observed networks
    network_names = [net["network"] for net in inputs["networks"]]

    graphs = {}

    for network_name in network_names:

        network_path = os.path.join(input_dir, networks_dir, network_name)

        # check if the network file exist (either gml or gml.gz)
        gml_exists = os.path.exists(network_path + ".gml")
        gmlz_exists = os.path.exists(network_path + ".gml.gz")

        # Read the graphs
        if gml_exists:
            graphs[network_name] = mlgraph.read.from_gml(network_path + ".gml")
        elif gmlz_exists:
            graphs[network_name] = mlgraph.read.from_zipped_gml(network_path + ".gml.gz")
        else:
            print("File '{}' does not exists, skipping this network.".format(network_name))

    return graphs

def find_communities(inputs):
    pass

def create_null_networks():
    pass

def find_null_communities():
    pass

def select_params():
    pass

def find_consensus_comms():
    pass

if __name__ == "__main__":
    
    config_file = get_config_file() # get path to the config file
    inputs = get_inputs(config_file) # get input arguments from config file

    graphs = read_networks(inputs) 

    if len(graphs) == 0:
        print("There is no network to analyze, runner is aborted.")
        quit()

    for net_name, net in graphs.items():
        print(net.summary())

    # find_communities(inputs, graphs) # find the communities of observed multilayer graphs

    # create_null_networks() # create or read null networks 

    # find_null_communities()

    # select_params()

    # find_consensus_comms()

