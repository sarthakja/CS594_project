import os
import pickle
from utils import make_graphs

dir_path = os.path.dirname(os.path.realpath(__file__))
dir = os.path.dirname(dir_path)


if __name__ == '__main__':

    excel_path = dir + "/data/combined_sites.xlsx"
    pickle_path = dir + "/data/residue_small.pkl"

    with open(pickle_path, "rb") as handle:
        print("Retrieve the pickle!")
        res_dict = pickle.load(handle)

    # generate graph for local binding site
    make_graphs(excel_path, res_dict)

