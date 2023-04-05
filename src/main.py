import os
import pickle
import pandas as pd
from utils import make_graphs, get_all_pdbs, get_ligand_site, get_resn_attributes, generate_pytroch_graph

dir_path = os.path.dirname(os.path.realpath(__file__))
dir = os.path.dirname(dir_path)


if __name__ == '__main__':

    pdb_path = dir + "/data/pdbs/"
    pdbs_list = dir + "/data/chol_pdbs.txt"
    active_site_pickle =pdbs_list.replace("txt", "pkl")

    excel_path = dir + "/data/combined_sites_032923.xlsx"
    binding_site_list = dir + "/data/subclusters_0312.xlsx"
    graphs_path = dir + "/data/torch_graphs_040523.pkl"

    print("Start generate graphs")

    with open(active_site_pickle, "rb") as handle:
        print("Retrieve the pickle!")
        atom_dict, chain_dict, res_dict = pickle.load(handle)

    if not os.path.isfile(graphs_path):

        datasets = generate_pytroch_graph(excel_path, res_dict, binding_site_list)

        f = open(graphs_path, "wb")
        pickle.dump(datasets, f)






