import os
import pickle
from utils import make_graphs

dir_path = os.path.dirname(os.path.realpath(__file__))
dir = os.path.dirname(dir_path)


if __name__ == '__main__':

    excel_path = dir + "/data/combined_sites.xlsx"
    pickle_path = dir + "/data/chol_pdbs.pkl"

    # generate graph for local binding site
    make_graphs(excel_path, pickle_path)

