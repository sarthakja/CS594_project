import os
import pickle
from utils import make_graphs, get_all_pdbs, get_ligand_site

dir_path = os.path.dirname(os.path.realpath(__file__))
dir = os.path.dirname(dir_path)


if __name__ == '__main__':

    excel_path = dir + "/data/combined_sites.xlsx"
    pickle_path = dir + "/data/residue_small.pkl"
    pdb_path = dir + "/data/pdbs/"

    pdbs_list = dir + "/data/chol_pdbs_small.txt"

    active_res_dict, lig_set = None, None
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

    if active_res_dict == None:
        with open(pickle_path, "rb") as handle:
            print("Retrieve the pickle!")
            res_dict, site_dict = pickle.load(handle)

    # generate graph for local binding site
    make_graphs(excel_path, res_dict, site_dict)

