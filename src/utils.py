import os
import pandas as pd
import networkx as nx


dir_path = os.path.dirname(os.path.realpath(__file__))
dir = os.path.dirname(dir_path)

def make_graphs(attribute_file, residue_dict, distance=7.0):

    # read in the excel file
    df = pd.read_excel(attribute_file)

    for pdb in residue_dict.keys():

        for chol_res in residue_dict[pdb].keys():

            # create an empty graph
            G = nx.Graph()

            # list of empty nodes for the graph
            nodes = []

            chol_chain = chol_res.get_parent().id
            chol_resi = chol_res.id[1]

            key = f"{pdb}_{chol_chain}_{chol_resi}"

            for residue in residue_dict[pdb][chol_res][0].keys():
                df_res = df.loc[(df["CHOL ID"] == key) & (df["RESIDUE NAME"] == residue.get_resname())
                                                          & (df["RESIDUE SEQ"] == residue.id[1])]
                # print(df_res.iloc[0]["PERCENT NON POLAR"])

                node_key = f"{key}_{residue.get_resname()}_{residue.id[1]}"
                print(node_key)

                # TODO keep the features as string or use one hot coding?
                # node = (node_key, {"res_name": residue.get_resname(),
                #                    res})
