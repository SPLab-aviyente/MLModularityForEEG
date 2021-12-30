import argparse

import yaml

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

    find_communities()

    create_null_networks()

    find_null_communities()

    select_params()

    find_consensus_comms()

