import sys
sys.path.append("/sansom/n15/shil3498/scripts/mda_contacts/v1.0/")
import MDAnalysis as mda
import numpy as np
from contacts import LipidContactGenerator

U = mda.Universe(sys.argv[1],sys.argv[2])

generate = LipidContactGenerator(U)

contacts = generate.build_contacts(ligand_selection="resname POPE or resname POPG or resname CARD",frameskip=1,cutoff=6,KDTree=True)

contacts.aggregate(group_protein_by="resid",group_ligand_by="segid",aggregate_function=lambda x:x.max())

data = contacts.time_aggregate(aggregate_function=lambda x:np.sum(x.values())/contacts.n_frames)

data.to_dataframe().to_csv("LIPID-contacts.csv")

contacts = generate.build_contacts(ligand_selection="resname POPE or resname POPG or resname CARD",frameskip=1,cutoff=6,KDTree=True)

contacts.aggregate(group_protein_by="resid",group_ligand_by="resname",aggregate_function=lambda x:x.max())

data = contacts.time_aggregate(aggregate_function=lambda x:np.sum(x.values())/contacts.n_frames)

data.to_dataframe().to_csv("POPG-POPE-CARD-contacts.csv")
