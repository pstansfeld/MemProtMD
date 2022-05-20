import matplotlib as mpl
mpl.use('Agg')
import sys
sys.path.append("/storage/chem/lfsmgr/SRG/MemProtMD3/")
import MDAnalysis as mda
import numpy as np
from contacts import LipidContactGenerator
import matplotlib.pyplot as plt
import pandas as pd
from pandas import *
import MDAnalysis.analysis.rms
from matplotlib import rc, rcParams
import pylab

mpl.rc_file_defaults()

d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

U = mda.Universe(sys.argv[1],sys.argv[2])

A = U.select_atoms("protein and name BB")

All = U.atoms

resids = A.atoms.residues.resids
resnames = A.atoms.residues.resnames

df = pd.DataFrame(data=A.atoms.residues.resnames, index=A.atoms.residues.resids, columns=['Residue'])

generate = LipidContactGenerator(U)

contacts = generate.build_contacts(protein_selection="protein",ligand_selection="resname W or resname POPE or resname POPG or resname CARD",frameskip=1,cutoff=6,KDTree=True)
contacts.aggregate(group_protein_by="resid",group_ligand_by="resname",aggregate_function=lambda x:x.max())
data = contacts.time_aggregate(aggregate_function=lambda x:sum(x.values())/contacts.n_frames)

data.to_dataframe().to_csv("Lipid-contacts.csv")

oant = data.to_dataframe()

oant = pd.concat([oant, df], axis=1, sort=False, join='inner')

oant['Resid']=oant.index
oant['combined']=oant['Residue']+oant.index.astype(str)

U.add_TopologyAttr(mda.core.topologyattrs.Tempfactors(np.zeros(len(All))))

for i in All:
	if i.residue.resid in oant.index:
		i.tempfactor = oant.loc[i.residue.resid,['POPE']].values[0]

U.trajectory[0]
All.write("POPE-contacts.pdb")

U.add_TopologyAttr(mda.core.topologyattrs.Tempfactors(np.zeros(len(All))))

for i in All:
        if i.residue.resid in oant.index:
                i.tempfactor = oant.loc[i.residue.resid,['POPG']].values[0]

U.trajectory[0]
All.write("POPG-contacts.pdb")

U.add_TopologyAttr(mda.core.topologyattrs.Tempfactors(np.zeros(len(All))))

for i in All:
        if i.residue.resid in oant.index:
                i.tempfactor = oant.loc[i.residue.resid,['CARD']].values[0]

U.trajectory[0]
All.write("CARD-contacts.pdb")

U.add_TopologyAttr(mda.core.topologyattrs.Tempfactors(np.zeros(len(All))))

for i in All:
        if i.residue.resid in oant.index:
                i.tempfactor = oant.loc[i.residue.resid,['W']].values[0]

U.trajectory[0]
All.write("Water-contacts.pdb")
