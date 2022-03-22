#/usr/bin/python 

import os
import sys
import gromacs
import shutil

inpdb = sys.argv[1]

if not os.path.exists('temp'):
	os.makedirs('temp')
os.chdir('temp')

print("Assessing "+inpdb+"\n")

os.system('pdb2pqr --ff CHARMM  --chain ../'+inpdb+' pqr.pdb')

gromacs.pdb2gmx(f='pqr.pdb',ignh=True,ff='charmm27',water='tip3p',o='conf.pdb')

gromacs.editconf(f='conf.pdb',d=8,c=True,o='conf.pdb')

with open('em.mdp','w') as em:
            em.write('integrator = steep\nnsteps = 5000\nemtol = 100\nemstep = 0.001')

gromacs.grompp(f='em.mdp',maxwarn=5,o='em',c='conf.pdb')

gromacs.mdrun(deffnm='em',c='clean.pdb')

gromacs.trjconv(f='clean.pdb',o='../fixed-'+inpdb,s='em.tpr',input=('system'))

os.chdir('..')

shutil.rmtree('temp', ignore_errors=True)
