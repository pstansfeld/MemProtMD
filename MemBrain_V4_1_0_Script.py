import os
import sys
import warnings
import jax.numpy as jnp
from jax.config import config
from jax.experimental.maps import xmap,Mesh
from jax import tree_util
import jax
import jax.profiler
import numpy as np
import matplotlib.pyplot as plt
import time
import MemBrain_V4_1_0 as ori
import argparse

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

#Reading command line args
parser = argparse.ArgumentParser(prog="MemBrain",description="Orients a protein in a membrane. Currently only for E-coli membranes, but this will change in future updates. Currently flags -c -c_ni -ps -fc are WIP, these can still be used at the your own peril.")
parser.add_argument("-f", "--file_name",help = "Input file name (.pdb)")
parser.add_argument("-o","--output",help="Name of the output directory (Default: Orient)")
parser.add_argument("-ni","--iters",help="Number of minimisation iterations (Default: 150)")
parser.set_defaults(iters=150)
parser.add_argument("-ng","--grid_size",help="Number of starting configurations (Default: 20)")
parser.set_defaults(grid_size=20)
parser.add_argument("-dm","--dual_membrane",action="store_true",help="Toggle dual membrane orientation")
parser.set_defaults(dual_membrane=False)
parser.add_argument("-pg","--predict_pg_layer",action="store_true",help="Toggle peptidogylcan call wall prediction")
parser.set_defaults(predict_pg_layer=False)
parser.add_argument("-pr","--peripheral",action="store_true",help="Toggle peripheral (or close to) orientation")
parser.set_defaults(peripheral=False)
parser.add_argument("-w","--use_weights",action="store_true",help="Toggle use of b-factors to weight orientation")
parser.set_defaults(use_weights=False)
parser.add_argument("-fc","--force_calc",help="Calculates the force in the z-axis for ranks < n (Default: n=0)")
parser.set_defaults(force_calc=0)
parser.add_argument("-c","--curvature",action="store_true",help="Toggle curvature minimisation. Cannot use with -mt_opt")
parser.set_defaults(curvature=False)
parser.add_argument("-c_ni","--curvature_iters",help="Number of iterations for curvature minimisation")
parser.set_defaults(curvature_iters=150)
parser.add_argument("-itp","--itp_file",help="Path to force field (martini_v3.itp)")
parser.set_defaults(itp_file="/home/maths/maukjg/Cardiolipins/E-coli_complex/Orient/Orient_JAX/Orient_build/martini_v3.itp")
parser.add_argument("-bd","--build_system",help = "Build a MD ready CG-system for ranks < n (Default: n=0)")
parser.set_defaults(build_system=0)
parser.add_argument("-bd_args","--build_arguments",help="Arguments to pass to insane when building system")
parser.set_defaults(build_arguments="")
parser.add_argument("-ch","--charge",help="Partial charge of (inner) membrane (Deafult: 0.1)")
parser.set_defaults(charge=-0.1)
parser.add_argument("-ch_o","--charge_outer",help="Partial charge of outer membrane (Deafult: 0.02)")
parser.set_defaults(charge_outer=-0.02)
parser.add_argument("-mt","--membrane_thickness",help="Initial thickness of (inner) membrane in Angstroms (Deafult: 28)")
parser.set_defaults(membrane_thickness=28.0)
parser.add_argument("-mt_o","--outer_membrane_thickness",help="Initial thickness of outer membrane in Angstroms (Deafult: 24)")
parser.set_defaults(outer_membrane_thickness=24.0)
parser.add_argument("-mt_opt","--membrane_thickness_optimisation",action="store_true",help="Toggle membrane thickness optimisation. Cannot use with -c")
parser.set_defaults(membrane_thickness_optimisation=False)
parser.add_argument("-tm","--transmembrane_residues",help="This indicates if there are residues known to be in the membrane. Input is formatted as a comma seperated list of inclusive-exclusive ranges e.g. -tm 1-40,50-60.")
parser.set_defaults(transmembrane_residues = "")
args = parser.parse_args()

#Error checking user inputs
try:
	grid_size = int(args.grid_size)
except:
	print("ERROR: Could not read value of -ng. Must be an integer > 3.")
	exit()
if(grid_size < 4):
	print("ERROR: Could not read value of -ng. Must be an integer > 3.")
	exit()


mem_data = [0,0,0,0]
try:
	mem_data[0] = -float(args.charge)
except:
	print("ERROR: Could not read value of -ch. Must be a float.")
	exit()


try:
	mem_data[1] = -float(args.charge_outer)
except:
	print("ERROR: Could not read value of -ch_o. Must be a float.")
	exit()

try:
	mem_data[2] = float(args.membrane_thickness)
except:
	print("ERROR: Could not read value of -mt. Must be a float > 0.")
	exit()
if(mem_data[2] <= 0):
	print("ERROR: Could not read value of -mt. Must be a float > 0.")
	exit()
	
try:
	mem_data[3] = float(args.outer_membrane_thickness)
except:
	print("ERROR: Could not read value of -mt_o. Must be a float > 0.")
	exit()
if(mem_data[3] <= 0):
	print("ERROR: Could not read value of -mt_o. Must be a float > 0.")
	exit()

	
try:
	iters = int(args.iters)
except:
	print("ERROR: Could not read value of -ni. Must be an integer >= 0.")
	exit()
if(iters < 0):
	print("ERROR: Could not read value of -ni. Must be an integer >= 0.")
	exit()

try:
	curvature_iters = int(args.curvature_iters)
except:
	print("ERROR: Could not read value of -c_ni. Must be an integer > 0.")
	exit()
if(curvature_iters < 1):
	print("ERROR: Could not read value of -c_ni. Must be an integer > 0.")
	exit()

try:
	build_system = int(args.build_system)
except:
	print("ERROR: Could not read value of -bd. Must be an integer > -1.")
	exit()
if(build_system < 0):
	print("ERROR: Could not read value of -bd. Must be an integer > -1.")
	exit()
	
if(args.curvature):
	if(build_system > 0):
		print("ERROR: Cannot build system with curvature minimisation.")
		exit()
	if(args.membrane_thickness_optimisation):
		print("ERROR: Cannot optimise membrane thickness and run curvature minimisation.")
		exit()
		
if(args.predict_pg_layer):
	if(not args.dual_membrane):
		print("Error: cannot predict PG cell wall if not using -dm")
		exit()

if(build_system > 0):
	if(args.build_arguments == ""):
		print("ERROR: -bd_args must be supplied when using -bd.")
		exit()
	if(args.membrane_thickness_optimisation):
		print("WARNING: Cannot build with optimised membrane thickness.")
	
try:
	force_calc = int(args.force_calc)
except:
	print("ERROR: Could not read value of -fc. Must be an integer > -1.")
	exit()
if(force_calc < 0):
	print("ERROR: Could not read value of -fc. Must be an integer > -1.")
	exit()
	
	
list_ranges = args.transmembrane_residues.split(",")
ranges = []
if(list_ranges != [""]):
	for i in list_ranges:
		rang = i.strip()
		rang = rang.split("-")
		if(len(rang) != 2):
			print("ERROR: Could not read value of -tm. Must be a comma seperated list of ranges e.g. 10-40,50-60.")
			exit()
		try:
			ranges.append([int(rang[0]),int(rang[1])])
		except:
			print("ERROR: Could not read value of -tm. All values must be an integer > 0.")
			exit()
		if(ranges[-1][0] <= 0 or ranges[-1][1] <= 0):
			print("ERROR: Could not read value of -tm. All values must be an integer > 0.")
			exit()
		if(ranges[-1][0] > ranges[-1][1]):
			print("ERROR: Could not read value of -tm. For a range x-y, x < y must be true.")
			exit()
	ranges = np.array(ranges)
	if(args.use_weights):
		print("ERROR: Cannot use -tm with -w.")
		exit()
else:
	ranges = np.array([])



fn = args.file_name

if(not os.path.exists(fn)):
	print("ERROR: Cannot find file: "+fn)
	exit()

orient_dir = args.output

#Setting Martini itp file
martini_file = args.itp_file

if(not os.path.exists(martini_file)):
	print("ERROR: Cannot find file: "+martini_file)
	exit()
	
	
#Creating folders to hold data
if(orient_dir == None):
	if(not os.path.exists("Orient/")):
		os.mkdir("Orient/")
	orient_dir = "Orient/"
else:
	if(not os.path.exists(orient_dir)):
		os.mkdir(orient_dir)

timer = time.time()

#Creating a helper class that deals with loading the PDB files
PDB_helper_test = ori.PDB_helper(fn,args.use_weights,build_system,ranges)

#Loading PDB
PDB_helper_test.load_pdb()


#Getting surface
print("Getting surface residues:")
PDB_helper_test.get_surface()
print("Done")

#For periplasmic spanning proteins we start with the longest dimension in the Z-axis
if(args.dual_membrane):
	_ = PDB_helper_test.starting_orientation_v2()


### FOR TESTING PURPOSES ONLY ##########################################
#PDB_helper_test.test_start([0,1.2,np.pi])
########################################################################

#Here we create the potential field from a martini file (This may become a class aswell which can be used to create arbritray membranes)
int_data = ori.get_mem_def(martini_file)

#We extract the data from the PDB helper class and pass it to the main orientation class
data = PDB_helper_test.get_data()


Mem_test = ori.MemBrain(data,int_data,args.peripheral,force_calc,args.predict_pg_layer,mem_data,False,args.dual_membrane)

#Setting up a sphereical grid for local minima calculations
angs = ori.create_sph_grid(grid_size)

#Getting a starting insertion depth
print("Getting initial insertion depth:")
if(args.dual_membrane):
	zdist,zs = Mem_test.calc_start_for_ps_imp()
	start_z = jnp.array([0,0,zs])
else:
	start_z = Mem_test.get_hydrophobic_core_jit()
	zdist = 0
	
print("Done")
	
print("Starting initial minimisation:")

#Minimising on the grid
Mem_test.minimise_on_grid(grid_size,start_z,zdist,angs,iters)
print("Done")

print("Collecting minima information:")
#Collating minima information
cols,pot_grid = Mem_test.collect_minima_info(grid_size)
print("Done")

if(args.membrane_thickness_optimisation):
	print("Optimising membrane thickness:")
	Mem_test.optimise_memt_all()
	print("Done")
	
if(args.curvature or False):
	if(False):
		print("Starting membrane thickness optimisation:")
	else:
		print("Starting curvature minimisation:")
	timert = time.time()	
	
	#We find the number of gaussians need to describe the membrane (depends on the size of the protein)
	Mem_test.get_no_gauss()
	
	#We create a starting membrane for each minima found
	Mem_test.format_data()
	
	#Iteratively finding best position and curvature (WIP)
	Mem_test.minimise_on_grid_c(curvature_iters)
	print("Done")
	#Extracting the data from the result grid
	normal_spos = Mem_test.get_all_normal_spos_jit()#Jitted functions need to be pure
	Mem_test.normal_spos = normal_spos 
	
	#We recollected all the data as the local minima will have changed
	print("Recollecting minima information:")
	Mem_test.recollect_minima_info_c()
	print("Done")
	#We use the data produced above to make some graphs
	pot_grid,cols  = Mem_test.make_graph_grids(angs,grid_size)
	Mem_test.make_mem_graphs(orient_dir,100)
	
	#print(time.time()-timert)
	
	
	
#print(Mem_test.get_pg_pos(400,0))


print("Writing data:")
#Writing to a file
Mem_test.write_oriented(fn,orient_dir)
print("Done")

print("Building system:")
Mem_test.build_oriented(orient_dir,args.build_arguments)
print("Done")

#Displaying the local minima graphs
print("Making graphs:")
ori.create_graphs(orient_dir,cols,pot_grid,angs,int(np.floor((np.sqrt(grid_size)*0.8))))
print("Done")

print("Total: "+str(np.round(time.time()-timer,3))+" s")



