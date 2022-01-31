import os
import argparse
import glob
from pathlib import Path
from itertools import product
from datetime import datetime

import yaml
import numpy as np
import pandas as pd

from src import mlgraph
from src.commdetect import modularity
from src.nulls import weighted_undirected as wu_nulls

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
            print("File '{}' does not exist, skipping this network.".format(network_name))

    return graphs

def find_communities(inputs, graphs):
    output_dir = inputs["output_dir"]
    networks_dir = inputs["networks_dir"]

    # Get modularity parameters
    gamma_min = inputs["gamma"][0]["min"]
    gamma_max = inputs["gamma"][1]["max"]
    n_gammas = inputs["gamma"][2]["n_points"]

    omega_min = inputs["omega"][0]["min"]
    omega_max = inputs["omega"][1]["max"]
    n_omegas = inputs["omega"][2]["n_points"]

    gammas = np.linspace(gamma_min, gamma_max, n_gammas, endpoint=True)
    omegas = np.linspace(omega_min, omega_max, n_omegas, endpoint=True)

    params = list(product(gammas, omegas))
    null_model = inputs["null_model"]
    
    n_params = len(params)
    n_runs = inputs["n_runs"]

    for graph_name, G in graphs.items():
        modularities = np.zeros((n_runs, n_params))

        # get edge weights and expected edge weights under the null model
        w_intra, w_inter = modularity.get_edge_weights(G)
        p_intra, p_inter = modularity.get_pijs(G, null_model)

        # at this step, we only need modularity values of partitions
        for p, param in enumerate(params):
            gamma = param[0]
            omega = param[1]

            _, q = modularity.find_comms(G, w_intra, w_inter, p_intra, p_inter, gamma, omega, n_runs)

            modularities[:, p] = q

        df_index = pd.MultiIndex.from_product([gammas, omegas], names=["gamma", "omega"])
        modularities_df = pd.DataFrame(modularities, columns=df_index)

        # save modularity values to csv
        save_dir = os.path.join(output_dir, networks_dir, graph_name, "modularities")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        now_str = "{}{:02d}{:02d}{}{}".format(now.year, now.month, now.day, now.hour, now.minute)
        save_name = "{}_gmin_{:.3f}_gmax_{:.3f}_ng_{:d}_omin_{:.3f}_omax_{:.3f}_no_{:d}_"\
                    "{}_nruns_{:d}.csv".format(now_str, gamma_min, gamma_max, n_gammas, 
                                               omega_min, omega_max, n_omegas, null_model, n_runs)
        save_file = os.path.join(save_dir, save_name)
        modularities_df.to_csv(save_file)

def create_null_networks(inputs, graphs):
    output_dir = inputs["output_dir"]
    networks_dir = inputs["networks_dir"]

    null_names = {net["network"]: net["null_networks_type"] for net in inputs["networks"]}

    null_functions = {
        "weight_preserved": lambda G: wu_nulls.weight_preserved(G), 
        "weight_and_layer_preserved": lambda G: wu_nulls.weight_preserved(G, preserve_layer=True),
        "strength_preserved": lambda G: wu_nulls.strength_preserved(G),
        "strength_and_layer_preserved": lambda G: wu_nulls.strength_preserved(G, preserve_layer=True)
    }

    nulls = {}
    for graph_name, G in graphs.items():
        
        null_name = null_names[graph_name]
        nulls_dir = os.path.join(output_dir, networks_dir, graph_name, null_name)

        n_runs = inputs["n_runs"]

        create_nulls = True
        if os.path.isdir(nulls_dir): 
            # There exists a folder with null networks. We will check if there exist at least 
            # n_runs number of null networks in this folder since we need enough samples to make 
            # any statictical conclusion. If not, we will create new nulls. 

            # get gml and gziped files
            files = glob.glob(os.path.join(nulls_dir, "*.gml")) + \
                    glob.glob(os.path.join(nulls_dir, "*.gml.gz"))

            if len(files) >= n_runs:
                create_nulls = False
            else:
                create_nulls = True

        nulls[graph_name] = []
        if create_nulls:
            # Create new null networks
            for r in range(n_runs):
                Gn = null_functions[null_name](G)
                nulls[graph_name].append(Gn)

                # make sure the null directory exists
                Path(nulls_dir).mkdir(parents=True, exist_ok=True)

                # Save the null
                file_name = os.path.join(nulls_dir, "network_{:d}".format(r+1))
                mlgraph.write.to_zipped_gml(Gn, file_name)

        else:
            # Read existing null networks
            for file in files:
                if file.endswith(".gml"):
                    nulls[graph_name].append(mlgraph.read.from_gml(file))
                elif file.endswith(".gml.gz"):
                    nulls[graph_name].append(mlgraph.read.from_gml(file))

    return nulls

def find_null_communities():
    pass

def select_params():
    pass

def find_consensus_comms():
    pass

def find_group_comms():
    pass

if __name__ == "__main__":
    
    config_file = get_config_file() # get path to the config file
    inputs = get_inputs(config_file) # get input arguments from config file

    graphs = read_networks(inputs) 

    if len(graphs) == 0:
        print("There is no network to analyze, runner is aborted.")
        quit()

    # find_communities(inputs, graphs) # find the communities of observed multilayer graphs

    create_null_networks(inputs, graphs) # create or read null networks

    find_null_communities()

    select_params()

    find_consensus_comms()

    if len(graphs) > 1:
        find_group_comms()