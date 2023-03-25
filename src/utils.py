import os
import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))
dir = os.path.dirname(dir_path)

def make_graphs(attribute_file, residue_dict, distance=7.0):

    # read in the excel file
    df = pd.read_excel(attribute_file)

    for pdb in residue_dict.keys():
        print(pdb)
        break
