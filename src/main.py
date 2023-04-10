import os
import pickle
import pandas as pd
from utils import make_graphs, get_all_pdbs, get_ligand_site, get_resn_attributes

dir_path = os.path.dirname(os.path.realpath(__file__))
dir = os.path.dirname(dir_path)


if __name__ == '__main__':

    positive_excel_path = dir + "/data/combined_sites_032923.xlsx"
    negative_excel_path = dir + "/data/negative_sites.xlsx"
    pickle_path = dir + "/data/residue_small.pkl"
    pdb_path = dir + "/data/pdbs/"

    pdbs_list = dir + "/data/chol_pdbs_small.txt"

    print("Start generate positive label graphs")
    # code that generate positive label dataset
    active_res_dict, lig_set, res_dict, site_dict = None, None, None, None
    if not os.path.isfile(pickle_path):
        res_dict = {}
        site_dict = {}
        pdb_ids = get_all_pdbs(pdbs_list, pdb_path)
        for id in pdb_ids:
            print(id)
            active_res_dict, lig_set = get_ligand_site(id, pdb_path, "CLR")
            res_dict.update(({id: active_res_dict}))
            site_dict.update(({id: lig_set}))
        f = open(pickle_path, "wb")
        pickle.dump((res_dict, site_dict), f)

    if res_dict == None:
        with open(pickle_path, "rb") as handle:
            print("Retrieve the pickle!")
            res_dict, site_dict = pickle.load(handle)

    negative_pdbs_list = dir + "/data/negative_pdbs_small.txt"

    negative_pdbs_dict = {"2AOC": "2NC", "3R6U": "CHT", "5EVZ": "ADP", "6GVZ": "CHO",
                          "3CMF": "PDN", "6T0G": "VD3", "3GWS": "T3", "7ZBE": "RET",
                          "7XE4": "XKP", "8E9X": "WEC"}

    print("Start generate negative label graphs")
    # code to generate negative label dataset
    neg_pickle_path = dir + "/data/neg_residue_small.pkl"
    neg_act_dict, neg_lig_set, neg_res_dict, neg_site_dict = None, None, None, None

    if not os.path.isfile(neg_pickle_path):
        neg_res_dict = {}
        neg_site_dict = {}

        neg_pdbs_id = get_all_pdbs(negative_pdbs_list, pdb_path)
        for neg_id in neg_pdbs_id:
            print(neg_id)
            neg_act_dict, neg_lig_set = get_ligand_site(neg_id, pdb_path, negative_pdbs_dict[neg_id.upper()])
            neg_res_dict.update(({neg_id: neg_act_dict}))
            neg_site_dict.update(({neg_id: neg_lig_set}))
        f = open(neg_pickle_path, "wb")
        pickle.dump((neg_res_dict, neg_site_dict), f)

    if neg_res_dict == None:
        with open(neg_pickle_path, "rb") as handle:
            print("Retrieve the pickle!")
            neg_res_dict, neg_site_dict = pickle.load(handle)

    if not os.path.isfile(negative_excel_path):
        print("Start generate negative label graphs")
        neg_res_writer = pd.ExcelWriter(negative_excel_path)
        neg_res_columns = ["CHOL ID", "RESIDUE NAME", "RESIDUE SEQ", "ATOM ID", "CHOL_ATOM",
                   "DIS TO CLOSEST CHOL", "SECONDARY STRUCTURE", "ASA", "PHI", "PSI", "SASA"]

        # for each pdb
        neg_res_rows = []

        for neg_id in neg_res_dict.keys():

            print(neg_id)
            pdb_file = pdb_path + neg_id + ".pdb"
            if os.path.isfile(pdb_file):

                # TODO need to catch PDBConstructionWarning error
                neg_res_row = get_resn_attributes(neg_id, neg_res_dict[neg_id])
                neg_res_rows += neg_res_row

        df = pd.DataFrame(neg_res_rows, columns=neg_res_columns)
        print(df)
        df.to_excel(neg_res_writer)
        neg_res_writer.save()
        print("Residues dataframe is save as " + negative_excel_path)

    graphs_path = dir + "/data/graphs.pkl"

    if not os.path.isfile(graphs_path):

        # generate graph for local binding site
        positive_graphs = make_graphs(positive_excel_path, res_dict)
        negative_graphs = make_graphs(negative_excel_path, neg_res_dict)

        f = open(graphs_path, "wb")
        pickle.dump((positive_graphs, negative_graphs), f)




