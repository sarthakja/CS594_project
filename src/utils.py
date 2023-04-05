import os
import pandas as pd
import networkx as nx
from Bio.PDB.PDBList import PDBList
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.DSSP import DSSP
from Bio.PDB.PDBIO import PDBIO
from itertools import combinations

dir_path = os.path.dirname(os.path.realpath(__file__))
dir = os.path.dirname(dir_path)

# 20 amino acid
one_hot_code_aa = {
        'ALA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ARG': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ASN': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ASP': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'CYS': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'GLU': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'GLN': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'GLY': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'HIS': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ILE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'LEU': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'LYS': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'MET': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'PHE': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'PRO': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'SER': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'THR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'TRP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'TYR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'VAL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }

one_hot_code_ss = {
        'H': [1, 0, 0, 0, 0, 0, 0, 0, 0],
        'B': [0, 1, 0, 0, 0, 0, 0, 0, 0],
        'E': [0, 0, 1, 0, 0, 0, 0, 0, 0],
        'G': [0, 0, 0, 1, 0, 0, 0, 0, 0],
        'I': [0, 0, 0, 0, 1, 0, 0, 0, 0],
        'T': [0, 0, 0, 0, 0, 1, 0, 0, 0],
        'S': [0, 0, 0, 0, 0, 0, 1, 0, 0],
        'P': [0, 0, 0, 0, 0, 0, 0, 1, 0],
        '-': [0, 0, 0, 0, 0, 0, 0, 0, 1]
    }
residue_list = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU",
                "GLN", "GLY", "HIS", "ILE", "LEU", "LYS",
                "MET", "PHE", "PRO", "SER", "THR", "TRP",
                "TYR", "VAL"]

def get_ligand_site(pdb_code, path, ligand, cutoff=5.0):
  """
  Get all the atoms in the active site of the protein
  for CLR within a certain cutoff
  :param pdb_code:
  :param pdb_path:
  :param ligand:
  :param cutoff:
  :return: an active site dictionary
  """
  parser = PDBParser()
  cif_parser = MMCIFParser()

  # Parse the structure into a PDB.Structure object
  pdb_path = path + pdb_code + ".pdb"
  cif_path = path + pdb_code + ".cif"

  if os.path.isfile(pdb_path):
    struct = parser.get_structure(pdb_code, pdb_path)
  else:
    struct = cif_parser.get_structure(pdb_code, cif_path)

  lig_set = set()
  active_res_dict = {}

  # Get the list of chol atom
  for c in struct.get_chains():

    for res in c.get_residues():
      if res.get_resname() == ligand:
        lig_set.add(f"{pdb_code}_{c.id}_{res.id[1]}")
        res_unique_dict = {}

        # for each of cholesterol atom
        for atom in res.get_atoms():

          # find all other atoms in pdb that within a certain length

          for c2 in struct.get_chains():

            for res2 in c2.get_residues():

              if res2.get_resname() != ligand:
                for atom2 in res2.get_atoms():
                  distance = atom - atom2

                  if distance <= cutoff:

                    if res2.get_resname() in residue_list:
                      if res2 not in res_unique_dict.keys() or res_unique_dict[res2][2] > distance:
                        # value (cholesterol atom, protein atom, distance)
                        res_unique_dict.update({res2:(atom, atom2, distance)})

        if len(res_unique_dict.keys()) > 0:
            active_res_dict.update({res: res_unique_dict})

  return active_res_dict, lig_set

def get_all_pdbs(filename, pdb_path):
  """
  Download all pdbs file
  :param filename: the file contains all PDB IDs
  :param pdb_path: the directory to download the file too
  :return: the list contains all pdb ids
  """
  pdbl = PDBList()
  pdb_ids = []
  # Read the pdb file into a list
  with open(filename) as f:
    for line in f:
      pdb_id = line.lower().replace("\n", "")
      if not os.path.isfile(pdb_path + pdb_id + ".pdb"):
        native_pdb = pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_path, file_format='pdb')
        if os.path.isfile(native_pdb):
          os.rename(pdb_path + "pdb" + pdb_id + ".ent", pdb_path + "/" + pdb_id + ".pdb")
        else:
          pdbl.retrieve_pdb_file(pdb_id, pdir=pdb_path, file_format='mmCif')

      pdb_ids.append(pdb_id)
  return pdb_ids

def make_graphs(attribute_file, residue_dict, binding_site_list):

    # read in the excel file
    df = pd.read_excel(attribute_file)

    # read in the clusters
    clusters_df = pd.read_excel(binding_site_list)

    column_name = list(clusters_df.columns)

    graphs = []
    labels = []

    for pdb in residue_dict.keys():
        print(pdb)

        for chol_res in residue_dict[pdb].keys():

            # create an empty graph
            G = nx.Graph()

            # list of empty nodes for the graph
            nodes = []

            chol_chain = chol_res.get_parent().id
            chol_resi = chol_res.id[1]

            key = f"{pdb}_{chol_chain}_{chol_resi}"

            not_found = True
            counter = 0

            while not_found and counter < len(column_name):

                if f"{pdb.upper()}_{chol_resi}_{chol_chain}" in clusters_df[column_name[counter]].values:
                    labels.append(counter)

                    not_found = False

                    for residue in residue_dict[pdb][chol_res][0].keys():
                        df_res = df.loc[(df["CHOL ID"] == key) & (df["RESIDUE NAME"] == residue.get_resname())
                                                                  & (df["RESIDUE SEQ"] == residue.id[1])]

                        node_key = f"{key}_{residue.get_resname()}_{residue.id[1]}"

                        node = (node_key, {"res_name": one_hot_code_aa[residue.get_resname()],
                                           "res_ss": one_hot_code_ss[df_res.iloc[0]["SECONDARY STRUCTURE"]],
                                           "ASA": df_res.iloc[0]["ASA"],
                                           "PHI": df_res.iloc[0]["PHI"],
                                           "PSI": df_res.iloc[0]["PSI"],
                                           "SASA": df_res.iloc[0]["PSI"]})
                        nodes.append(node)


                    G.add_nodes_from(nodes)
                    edges = get_neighbor_res(residue_dict[pdb][chol_res][0].keys(), key)
                    G.add_edges_from(edges)

                    graphs.append(G)

                counter += 1

    print(labels)
    print(len(graphs))
    return graphs, labels



def get_neighbor_res(residue_list, key, cutoff=7.0):

    all_pairs = combinations(residue_list, 2)
    edges = []

    for res1, res2 in all_pairs:
        res1_key = f"{key}_{res1.get_resname()}_{res1.id[1]}"
        res2_key = f"{key}_{res2.get_resname()}_{res2.id[1]}"

        distance = res1["CA"] - res2["CA"]
        if distance < cutoff:
            edges.append((res1_key, res2_key))

    return edges
def get_resn_attributes(pdb_id, binding_site_dict):

  p = PDBParser()
  path = dir + "/data/pdbs/{pdb_id}.pdb".format(pdb_id=pdb_id)
  structure = p.get_structure(pdb_id, path)
  model = structure[0]

  try:
    # Get DSSP
    dssp = DSSP(model, path)
  except Exception:
    dssp = {}

  # get the SASA
  sr = ShrakeRupley()
  sr.compute(structure, level="R")

  residues_rows = []


  for chol_site in binding_site_dict.keys():

    site_id = "{pdb_id}_{chol_chain}_{chol_id}".format(pdb_id=pdb_id, chol_chain=chol_site.get_parent().id,
                                                       chol_id=chol_site.id[1])

    chain_set = set()

    for res in binding_site_dict[chol_site].keys():

      res_value = binding_site_dict[chol_site][res]
      res_chain = res.get_parent()
      key = (str(res_chain.id), res.id)

      chain_set.add(str(res_chain.id))

      if key in dssp.keys():
        dssp_value = dssp[key]

      else:
        chain_path = dir + "/data/pdbs/{pdb_id}_{chain_id}.pdb".format(pdb_id=pdb_id, chain_id=res_chain.id)
        if not os.path.isfile(chain_path):
          for chain in structure.get_chains():
            if chain.id == res_chain.id:
              io = PDBIO()
              io.set_structure(chain)
              io.save(chain_path.replace(".pdb", "_temp.pdb"))

          f = open(chain_path.replace(".pdb", "_temp.pdb"), 'r')
          newf = open(chain_path, 'w')
          lines = f.readlines()  # read old content

          newf.write("CRYST1\n")  # write new content at the beginning

          for line in lines:  # write old content after new
            newf.write(line)
          newf.close()
          f.close()

        smaller_structure = p.get_structure(pdb_id, chain_path)
        smaller_model = smaller_structure[0]

        small_dssp = DSSP(smaller_model, chain_path)

        dssp_value = small_dssp[key]

      sasa = round(model[str(res_chain.id)][res.id[1]].sasa, 2)


      # attributes = (secondary structure, asa, phi, psi, sasa)
      chol_atom = res_value[0]
      res_atom = res_value[1]


      attributes = [site_id, res.get_resname(), res.id[1], res_atom.id, chol_atom.id,
                    round(chol_atom - res_atom, 2), dssp_value[2], round(dssp_value[3], 2),
                    dssp_value[4], dssp_value[5], sasa]

      residues_rows.append(attributes)

  return residues_rows
