#Version 4 of MemBrain (WIP)
import os

import warnings
import jax.numpy as jnp
from jax.config import config
from jax.experimental.maps import xmap,Mesh
import jax
import numpy as np
import matplotlib.pyplot as plt
import time
from enum import Enum
from functools import partial
import datetime
from jax import tree_util


PATH_TO_INSANE = "/content/Insane4MemBrain.py"

warnings.filterwarnings('ignore')

#We define some enumerations for use later
class Reses(Enum):
	ALA = 0
	GLY = 1
	ILE = 2
	LEU = 3
	PRO = 4
	VAL = 5
	PHE = 6
	TYR = 7
	TRP = 8
	ARG = 9
	LYS = 10
	HIS = 11
	SER = 12
	THR = 13
	ASP = 14
	GLU = 15
	ASN = 16
	GLN = 17
	CYS = 18
	MET = 19
	
class Beads(Enum):
	BB =  0
	SC1 = 1
	SC2 = 2
	SC3 = 3
	SC4 = 4
	SC5 = 5
	SC6 = 6
	
class Beadtype(Enum):
	SP2 = 0 #p4
	TC3 = 1 #p4
	SP1 = 2 #P1
	P2 = 3 #p5
	SC2 = 4 #AC2
	SP2a = 5 #p5
	SC3 = 6 #c3
	SC4 = 7 #SC5
	TC5 = 8 #SC5
	TC4 = 9 #SC4
	TN6 = 10 #SP1
	TN6d = 11 #SNd
	SQ3p = 12 #Qp
	SQ4p = 13 #Qp
	TN5a = 14 #SP1
	TP1 = 15 #P1
	SQ5n = 16 #Qa
	Q5n = 17 #Qa
	TC6 = 18 #c5
	C6 = 19 #c5
	P5 = 20 #p5
	SP5 = 21 #p4
	
#We need to force JAX to fully utilize a multi-core cpu
no_cpu = int(os.environ["NUM_CPU"])
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count="+str(no_cpu)

#This is useful for debugging nans in grad
#config.update("jax_debug_nans",True)

#forcing JAX to use x64 floats rather than x32
config.update("jax_enable_x64",True)
config.update("jax_platform_name","cpu")
#This is a class that deals with the pdb files and all construction of position arrays
class PDB_helper:
	def __init__(self,fn,use_weights,build_no,ranges):
		#A list to help with getting the correct bead types from a cg pdb
		self.beads = [["SP2","TC3"],["SP1"],["P2","SC2"],["P2","SC2"],["SP2a","SC3"],["SP2","SC3"],["P2","SC4","TC5","TC5"],["P2","TC4","TC5","TC5","TN6"],["P2","TC4","TN6d","TC5","TC5","TC5"],["P2","SC3","SQ3p"],["P2","SC3","SQ4p"],["P2","TC4","TN6d","TN5a"],["P2","TP1"],["P2","SP1"],["P2","SQ5n"],["P2","Q5n"],["P2","P5"],["P2","SP5"],["P2","TC6"],["P2","C6"]]
		
		self.fn = fn
		self.use_weights = use_weights
		self.build_no = build_no
		self.ranges = ranges
		
		
	def in_ranges(self,num):
		for i in self.ranges:
			if(i[1] > num >= i[0]):
				return True
		return False
	#detects format of input
	def detect_atomistic(self):
		lfile = open(os.path.join(self.fn),"r")
		content = lfile.read()
		lfile.close()
		content = content.split("\n")
		for c in content:
			if(len(c) > 46):
				if("[" not in c and c[:4]=="ATOM"):	
					bead = c[12:17].strip()
					if(bead not in ["BB","SC1","SC2","SC3","SC4","SC5"]):
						return True
		return False
	
	#converts a atomistic resiude to cg representation
	def convert_to_cg(self,res_type,poses,b_vals,atom_types):
		if(res_type == 0):
			num = 2
			atom_to_bead = [["N","C","O"],["CB"]]
		if(res_type == 1):
			num = 1
			atom_to_bead = [["N","C","O","CA"]]
		if(res_type == 2):
			num = 2
			atom_to_bead = [["N","C","O"],["CB","CD1","CG1","CG2"]]
		if(res_type == 3):
			num = 2
			atom_to_bead = [["N","C","O"],["CB","CG","CD1","CD2"]]
		if(res_type == 4):
			num = 2
			atom_to_bead = [["C","O"],["CB","CA","CD","CG","N"]]
		if(res_type == 5):
			num = 2
			atom_to_bead = [["N","C","O"],["CB","CG1","CG2"]]
		if(res_type == 6):
			num = 4
			atom_to_bead = [["N","C","O"],["CB","CA","CG"],["CD1","CE1"],["CD2","CE2"]]
		if(res_type == 7):
			num = 5
			atom_to_bead = [["N","C","O"],["CB"],["CD1"],["CD2"],["CZ"]]
		if(res_type == 8):
			num = 6
			atom_to_bead = [["N","C","O"],["CB"],["CG","NE1","CD1"],["CD2","CE2"],["CZ3","CE3"],["CZ2","CH2"]]
		if(res_type == 9):
			num = 3
			atom_to_bead = [["N","C","O"],["CB","CG","CD"],["NE","CZ","NH2","NH1"]]
		if(res_type == 10):
			num = 3
			atom_to_bead = [["N","C","O"],["CB","CG"],["CD","CE","NZ"]]
		if(res_type == 11):
			num = 4
			atom_to_bead = [["N","C","O"],["CB","CG"],["CD2","NE2"],["ND1","CE1"]]
		if(res_type == 12):
			num = 2
			atom_to_bead = [["N","C","O"],["CB","OG"]]
		if(res_type == 13):
			num = 2
			atom_to_bead = [["N","C","O"],["CB","CG2","OG1"]]
		if(res_type == 14):
			num = 2
			atom_to_bead = [["N","C","O"],["CB","CG","OD2"]]
		if(res_type == 15):
			num = 2
			atom_to_bead = [["N","C","O"],["CD","CG","OE2","OE1"]]
		if(res_type == 16):
			num = 2
			atom_to_bead = [["N","C","O"],["CB","CG","ND2"]]
		if(res_type == 17):
			num = 2
			atom_to_bead = [["N","C","O"],["CD","CG","NE2"]]
		if(res_type == 18):
			num = 2
			atom_to_bead = [["N","C","O"],["SG","CB"]]
		if(res_type == 19):
			num = 2
			atom_to_bead = [["N","C","O"],["SD","CE","CG"]]
		
		reses = np.zeros(num)+res_type
		beads_pos = np.zeros((num,3))
		avs = np.zeros(num)
		bead_types = np.zeros(num)
		new_bvals = np.zeros(num)
		for i in range(num):
			bead_types[i] = Beadtype[self.beads[res_type][i]].value
		for i,xi in enumerate(poses):
			#print("here",atom_types[i])
			for k in range(num):
				if(atom_types[i] in atom_to_bead[k]):
					beads_pos[k] += xi
					avs[k] += 1
					new_bvals[k] += b_vals[i]
		bad = []
		good = []
		for i in range(num):
			if(avs[i] != 0):
				beads_pos[i] /= avs[i]
				new_bvals[i] /= avs[i]
				good.append(i)
			else:
				bad.append(i)
		for i in bad:
			beads_pos[i] = np.mean(beads_pos[good],axis=0)
			new_bvals[i] = np.mean(new_bvals[good])
		return np.array(beads_pos),np.array(bead_types),np.array(reses),np.array(new_bvals)


	#Loads a atomistic pdb into a cg representation
	def load_atomistic_pdb(self):
		self.reses = []
		self.poses = np.empty((0,3))
		self.bead_types = []
		self.b_vals = []
		
		reses2 = []
		poses2 = []
		self.all_poses = []
		atom_types = []
		b_vals2 = []
		lfile = open(os.path.join(self.fn),"r")
		content = lfile.read()
		lfile.close()
		content = content.split("\n")
		prev_atom_num = -1
		for c in content:
			if(len(c) > 46):
				if("[" not in c and c[:4]=="ATOM"):	
					res = c[17:20].strip()
					bead = c[12:17].strip()
					zpos = c[46:54]
					ypos = c[38:46]
					xpos = c[30:38]
					b_val = float(c[60:66].strip())
					atom_num = int(c[22:26].strip())
					pos = np.array([float(xpos.strip()),float(ypos.strip()),float(zpos.strip())])
					if(not np.any(np.isnan(pos))):
						self.all_poses.append(pos)
						if(atom_num == prev_atom_num):
							atom_types.append(bead)
							reses2.append(Reses[res].value)
							poses2.append(pos)
							if(self.ranges.size != 0):
								if(self.in_ranges(atom_num)):
									b_vals2.append(1.0)
								else:
									b_vals2.append(0.01)
							else:
								b_vals2.append(b_val)
						elif(prev_atom_num != -1):
							bead_pos,bead_types2,reses3,b_vals2 = self.convert_to_cg(reses2[0],poses2,b_vals2,atom_types)
							self.poses =np.concatenate((self.poses,bead_pos))
							self.bead_types = np.concatenate((self.bead_types,bead_types2))
							self.reses = np.concatenate((self.reses,reses3))
							self.b_vals = np.concatenate((self.b_vals,b_vals2))				
							reses2 = []
							poses2 = []
							b_vals2 = []
							atom_types = []
							atom_types.append(bead)
							reses2.append(Reses[res].value)
							poses2.append(pos)
							if(self.ranges.size != 0):
								if(self.in_ranges(atom_num)):
									b_vals2.append(1.0)
								else:
									b_vals2.append(0.01)
							else:
								b_vals2.append(b_val)
						else:
							atom_types.append(bead)
							reses2.append(Reses[res].value)
							poses2.append(pos)
							if(self.ranges.size != 0):
								if(self.in_ranges(atom_num)):
									b_vals2.append(1.0)
								else:
									b_vals2.append(0.01)
							else:
								b_vals2.append(b_val)
						prev_atom_num = atom_num
		 
		bead_pos,bead_types2,reses3,b_vals2 = self.convert_to_cg(reses2[0],poses2,b_vals2,atom_types)
		self.poses =np.concatenate((self.poses,bead_pos))
		self.bead_types = np.concatenate((self.bead_types,bead_types2))
		self.reses = np.concatenate((self.reses,reses3))
		self.b_vals = np.concatenate((self.b_vals,b_vals2))	
		self.b_vals = jnp.array(self.b_vals)	
		self.poses = jnp.array(self.poses)
		self.all_poses = jnp.array(self.all_poses)
		self.reses = jnp.array(self.reses)
		self.bead_types = jnp.array(self.bead_types)
		pos_mean = np.mean(self.poses,axis=0)
		self.poses = self.poses - pos_mean
		self.all_poses = self.all_poses - pos_mean
		self.b_vals = self.b_vals/jnp.max(self.b_vals)
		
	#Loads a cg pdb into jnp arrays containg position, residue index and beadtype index
	def load_cg_pdb(self):
		self.reses = []
		self.poses = []
		self.bead_types = []
		self.b_vals = []
		lfile = open(os.path.join(self.fn),"r")
		content = lfile.read()
		lfile.close()
		content = content.split("\n")
		for c in content:
			if(len(c) > 46):
				if("[" not in c and c[:4]=="ATOM"):	
					res = c[17:20].strip()
					bead = c[12:15].strip()
					zpos = c[46:54]
					ypos = c[38:46]
					xpos = c[30:38]
					atom_num = int(c[22:26].strip())
					b_val = float(c[60:66].strip())
					pos = np.array([float(xpos.strip()),float(ypos.strip()),float(zpos.strip())])
					if(not np.any(np.isnan(pos))):
						self.bead_types.append(Beadtype[self.beads[Reses[res].value][Beads[bead].value]].value)
						self.reses.append(Reses[res].value)
						self.poses.append(pos)
						if(self.ranges.size != 0):
							if(self.in_ranges(atom_num)):
								self.b_vals.append(1.0)
							else:
								self.b_vals.append(0.01)
						else:
							self.b_vals.append(b_val)

		self.poses = jnp.array(self.poses)
		self.reses = jnp.array(self.reses)
		self.bead_types = jnp.array(self.bead_types)
		self.poses = self.poses - jnp.mean(self.poses,axis=0)
		self.b_vals = jnp.array(self.b_vals)
		self.b_vals = self.b_vals/jnp.max(self.b_vals)
		self.all_poses = self.poses.copy()

	#loads a pdb
	def load_pdb(self):
		if(self.detect_atomistic()):
			if(self.build_no > 0):
				print("WARNING: Cannot build a CG-System for MD with atomistic input")
			self.build_no = 0
			self.load_atomistic_pdb()
		else:
			self.load_cg_pdb()
		#Setting read b-vals to 1 if not using weights
		if(not self.use_weights and self.ranges.size == 0):
			self.b_vals = self.b_vals.at[:].set(1.0)

			


	### Begining of code for getting the surface of the CG protein ###

	#We create a sphere using a fibonacci spiral lattice to ensure a even distribution 
	def create_ball(self,brad,bsize):
		self.ball = jnp.zeros((bsize,3))
		gr = (1+jnp.sqrt(5))/2
		def ball_fun_3(ball,ind):
			phi = jnp.arccos(1-2*(ind+0.5)/(bsize))
			theta = jnp.pi*(ind+0.5)*(gr)*2
			ball = ball.at[ind].set(jnp.array([brad*jnp.cos(phi),brad*jnp.sin(phi)*jnp.sin(theta),brad*jnp.sin(phi)*jnp.cos(theta)]))
			return ball, ind
			
		self.ball,_ = jax.lax.scan(ball_fun_3,self.ball,np.arange(bsize))
		return self.ball
		
		
	#This function gets the surface residues of a given cg protein
	def get_surface(self):
		bsize = 19#19#38
		brad = 4
		self.create_ball(brad,bsize)
		
		#Calculating normals at each point of the cg protein
		def normal_fun_2(carry,ind):
			def normal_fun_1(ind_fix,ind):
				normal_count = jnp.zeros(4)
				def in_sphere(normal,posa,posb):
					normal = normal.at[3].set(normal[3]+1)
					normal = normal.at[:3].set(normal[:3]+(-posa+posb))
					return normal
				def not_in_sphere(normal,posa,posb):
					return normal
				normal = jax.lax.cond(jnp.linalg.norm(self.poses[ind]-self.poses[ind_fix])<8,in_sphere,not_in_sphere,normal_count,self.poses[ind],self.poses[ind_fix])
				return ind_fix,normal
				
			_,normals = jax.lax.scan(normal_fun_1,ind,jnp.arange(self.poses.shape[0]))
			normals_sum = jnp.sum(normals,axis=0)
			normal = normals_sum[:3]/normals_sum[3]
			normal = normal/jnp.linalg.norm(normal)
			return carry,normal
		_,normals = jax.lax.scan(normal_fun_2,0,jnp.arange(self.poses.shape[0]))

		#Using the normals beads on the surface are flagged
		self.surface = jnp.zeros(self.poses.shape[0])
		def find_surface_fun_2(surface,ind):
			def find_surface_fun_1(ind_fix,ind):
				def in_prot():
					return 0
				def not_in_prot():
					return 1
				
				return ind_fix,jax.lax.cond(jnp.linalg.norm(self.poses[ind_fix]+normals[ind_fix]*5-self.poses[ind])<4.9,in_prot,not_in_prot)###################################################################<-----------------
			_,on_surf_arr = jax.lax.scan(find_surface_fun_1,ind,jnp.arange(self.poses.shape[0]))
			on_surf = jnp.prod(on_surf_arr)
			surface = surface.at[ind].set(on_surf)
			return surface,ind

		self.surface,_ = jax.lax.scan(find_surface_fun_2,self.surface,jnp.arange(self.poses.shape[0]))
		surface_number = jnp.sum(self.surface,dtype = "int")


		#A new array is created containg only surface positions
		self.surface_poses = self.poses[self.surface == 1]

		#Each point on the surface had an asocciated sphere indicating directions which are exposed
		#Here we flag the directions which are free
		def sphere_fun_3(ind_fix,ind):
			def sphere_fun_2(ind_fix,ind):
				def sphere_fun_1(ind_fix,ind):
					def in_prot():
							return 0
					def not_in_prot():
						return 1
					return ind_fix,jax.lax.cond(jnp.linalg.norm(self.poses[ind_fix[0]]+normals[ind_fix[0]]*4-self.poses[ind]+self.ball[ind_fix[1]])<4,in_prot,not_in_prot)
				_,on_surf_arr = jax.lax.scan(sphere_fun_1,jnp.array([ind_fix,ind]),jnp.arange(self.poses.shape[0]))
				on_surf = jnp.prod(on_surf_arr)
				return ind_fix,on_surf
			_,on_surf_ball = jax.lax.scan(sphere_fun_2,ind,jnp.arange(self.ball.shape[0]))
			ball_arr = on_surf_ball*self.surface[ind]
			return ind_fix,ball_arr
			
		_,spheres = jax.lax.scan(sphere_fun_3,0,jnp.arange(self.poses.shape[0]))

		#Here we set some values associated with each bead to be for surface only
		surface_sphere = spheres[self.surface == 1]
		self.bead_types = self.bead_types[self.surface == 1]
		self.surf_b_vals = self.b_vals[self.surface == 1]

		#Here we set the cartesean directions for each sphere (as before we only had flags)
		self.spheres = jnp.zeros((surface_number,self.ball.shape[0],3))
		def set_sphere_poses_fun_2(sphere_poses,ind):
			timeser = jnp.transpose(jnp.array([surface_sphere[ind],surface_sphere[ind],surface_sphere[ind]]))
			sphere_poses = sphere_poses.at[ind].set(sphere_poses[ind,:]+timeser*self.ball)
			return sphere_poses,ind
		self.spheres,_ = jax.lax.scan(set_sphere_poses_fun_2,self.spheres,jnp.arange(surface_sphere.shape[0]))
		surf_mean = jnp.mean(self.surface_poses,axis=0)
		self.surface_poses = self.surface_poses-surf_mean
		self.poses = self.poses-surf_mean
		self.all_poses = self.all_poses-surf_mean
		self.bead_types = jnp.array(self.bead_types,dtype="int")
		
	#For perisplasm spanning proteins it starts oriented with its farthest point at [0,0,1]
	def starting_orientation(self):
		all_dists = jnp.linalg.norm(self.surface_poses,axis=1)
		max_ind = jnp.argmax(all_dists)
		far = self.surface_poses[max_ind]
		far_dir = far/jnp.linalg.norm(far)
		ang2 = jnp.arccos(jnp.dot(far_dir,jnp.array([0,0,1])))
		far_projxy = far_dir.at[2].set(0)
		far_projxy /= jnp.linalg.norm(far_projxy)
		ang1 = jnp.arccos(jnp.dot(far_projxy,jnp.array([0,1,0])))
		self.surface_poses = position_point_jit(0,-ang1,ang2,self.surface_poses)
		self.poses = position_point_jit(0,-ang1,ang2,self.poses)
		self.all_poses = position_point_jit(0,-ang1,ang2,self.all_poses)
		def rot_spheres(carry,ind):
			new_spheres = position_point_jit(0,-ang1,ang2,self.spheres[ind])
			return carry,new_spheres
		_,self.spheres = jax.lax.scan(rot_spheres,0,jnp.arange(self.spheres.shape[0]))
		
		
	def starting_orientation_v2(self):
		all_dists = jnp.linalg.norm(self.surface_poses,axis=1)
		max_ind = jnp.argmax(all_dists)
		max_inds = jnp.argsort(all_dists)
		far = self.surface_poses[max_ind]
		
		#jax.debug.print("{x}",x=far)
		all_dists2 = jnp.linalg.norm(self.surface_poses-far,axis=1)
		min_inds = jnp.argsort(all_dists2)
		
		allowed1 = min_inds[all_dists2[min_inds]<jnp.linalg.norm(far)]
		
		#jax.debug.print("{x}",x=self.surface_poses[allowed1])
		allowed2 = max_inds[all_dists[max_inds]>jnp.linalg.norm(far)*0.85]
		#jax.debug.print("{x}",x=self.surface_poses[allowed2])
		all_allowed = jnp.intersect1d(allowed1,allowed2)
		
		#jax.debug.print("{x}",x=self.surface_poses[all_allowed])
		
		far = jnp.mean(self.surface_poses[all_allowed],axis=0)
		#jax.debug.print("{x}",x=far)
		far_dir = far/jnp.linalg.norm(far)
		ang2 = jnp.arccos(jnp.dot(far_dir,jnp.array([0,0,1])))
		far_projxy = far_dir.at[2].set(0)
		far_projxy /= jnp.linalg.norm(far_projxy)
		ang1 = jnp.arccos(jnp.dot(far_projxy,jnp.array([0,1,0])))
		self.surface_poses = position_point_jit(0,-ang1,ang2,self.surface_poses)
		self.poses = position_point_jit(0,-ang1,ang2,self.poses)
		self.all_poses = position_point_jit(0,-ang1,ang2,self.all_poses)
		def rot_spheres(carry,ind):
			new_spheres = position_point_jit(0,-ang1,ang2,self.spheres[ind])
			return carry,new_spheres
		_,self.spheres = jax.lax.scan(rot_spheres,0,jnp.arange(self.spheres.shape[0]))
		return self.surface_poses[all_allowed]
	
	#Use for testing different input orientations
	def test_start(self,new_pos):
		in_depth,ang1,ang2 = new_pos
		self.surface_poses = position_point_jit(in_depth,ang1,ang2,self.surface_poses)
		self.poses = position_point_jit(in_depth,ang1,ang2,self.poses)
		self.all_poses = position_point_jit(in_depth,ang1,ang2,self.all_poses)
		def rot_spheres(carry,ind):
			new_spheres = position_point_jit(0,ang1,ang2,self.spheres[ind])
			return carry,new_spheres
		_,self.spheres = jax.lax.scan(rot_spheres,0,jnp.arange(self.spheres.shape[0]))
	
	def get_data(self):
		return (self.surface,self.spheres,self.surface_poses,self.bead_types,self.surf_b_vals,self.poses,self.all_poses,self.b_vals,self.build_no)
		
#This is the main orientation class
class MemBrain:
	def __init__(self,data,int_data,peri,force_calc,pg_layer_pred,mem_data,memopt,dbmem):
		#We use interactions strengths from martini using a POPE(Q4p)/POPG(P4)/POPC(Q1)? lipid as a template
		self.int_data = int_data
		self.W_B_mins = int_data[0]
		self.LH1_B_mins = int_data[1]
		self.LH2_B_mins = int_data[2]
		self.LH3_B_mins = int_data[3]
		self.LH4_B_mins = int_data[4]
		self.LT1_B_mins = int_data[5]
		self.LT2_B_mins = int_data[6]
		self.Charge_B_mins =int_data[7]
		
		
		self.mem_data = mem_data
		
		self.charge_mult = mem_data[0]#0-2?
		self.charge_mult_om = mem_data[1]
				
		#These values could be changed via input. The current ones seem to be resonably good
		#However values such as charge of the bilayers and also the thickness are dependent on the type of membrane
		#and this would likley be something that could be an input.
		self.gamma_val = 0.999
		self.lr_pos = 0.00004 #5e-5
		self.lr_heights = 0.025 #0.05
		self.lr_cens = 0.02 #0.01
	
		#We need to define the PG layer differently (as it is outdated)
		self.pg_layer = -jnp.array([4.61,4.61,4.64,5.0,2.62,5.0,2.66,2.89,2.89,2.71,4.64,4.32,4.88,4.88,4.64,4.64,4.71,4.71,3.28,3.28,5.0,4.61])
		self.pg_charge = 0.1
		self.pg_water = -jnp.array([5.0,5.0,4.5,5.6,2.3,5.6,2.7,3.1,3.1,2.7,4.5,4.5,5.6,5.6,4.5,4.5,5.6,5.6,3.1,3.1,5.6,5.0])
		self.pg_thickness = 40
		self.sheet_charge = 0.0005
		
		#We now define the structure of the membrane (Possibly can be defined in an input file in future)
		self.memt_tails = mem_data[2]
		self.memt_heads = 10.0
		self.memt_total = self.memt_tails+self.memt_heads
		
		h1_w = self.memt_heads/6.0
		h2_w = self.memt_heads/6.0
		h3_w = self.memt_heads/6.0
		
		l_w = self.memt_tails
		meml = -l_w/2.0 -h1_w -h2_w-h3_w
		self.mem_structure_im = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
		self.mem_structure = self.mem_structure_im.copy()
		
		self.memt_tails_om = mem_data[3]
		self.memt_heads_om = 10.0
		self.memt_total_om = self.memt_tails_om+self.memt_heads_om
		
		h1_w_om = self.memt_heads_om/6.0
		h2_w_om = self.memt_heads_om/6.0
		h3_w_om = self.memt_heads_om/6.0
		
		l_w_om = self.memt_tails_om
		meml_om = -l_w_om/2.0 -h1_w_om -h2_w_om-h3_w_om
		self.mem_structure_om = jnp.array([meml_om,meml_om+h1_w_om,meml_om+h2_w_om+h1_w_om,meml_om+h2_w_om+h1_w_om+h3_w_om,meml_om+h2_w_om+h1_w_om+h3_w_om+l_w_om,meml_om+h2_w_om+h1_w_om+2*h3_w_om+l_w_om,meml_om+2*h2_w_om+h1_w_om+2*h3_w_om+l_w_om,meml_om+2*h2_w_om+2*h1_w_om+2*h3_w_om+l_w_om])
		
		
		
		self.mem_struture_dm = [self.mem_structure_im,self.mem_structure_om]
		
		
		#Here we set all the data produced by PDB_Helper
		self.data = data
		self.surface = data[0]
		self.spheres = data[1]
		self.surface_poses = data[2]
		self.bead_types = data[3]
		self.surf_b_vals = data[4]
		self.poses = data[5]
		self.all_poses = data[6]
		self.b_vals = data[7]
		
		#We need a flag for peripheral proteins
		self.peri = peri
		
		#A flag for preforming force calculationss
		self.force_calc  = force_calc
		
		#We need to flag if a curvature minimisation has occured
		self.curva = False
		
		#A flag for predicting the PG cell wall
		self.pg_layer_pred = pg_layer_pred
		
		#Flag for membrane thickness optimisation
		self.memopt = memopt
		
		#flag for dm
		self.dbmem = dbmem
		
		self.build_no = data[8]
		
		#We define some fixed values for curvature minimisation
		self.numa = 25
		self.numb = 25
		self.cut_a = 12
		self.cut_b = 12
		self.nums = 5
		self.lss_a = jnp.array([0.03]*self.numa)
		self.lss_b = jnp.array([0.03]*self.numb)#0.03
		
		
		#We define some values for limiting curvature mins (Memory/Speed reasons)
		self.only_top = 10000
		self.red_cpus = 18

		
		#Placeholder vars
		self.min_mem = jnp.zeros((1,4*self.numa+4*self.numb+2))
		self.min_mem = self.min_mem.at[:,-2].set(self.cut_a)
		self.min_mem = self.min_mem.at[:,-1].set(self.cut_b)
		self.no_mins=0
		self.result_grid_c = jnp.zeros((1,1))
		self.result_grid = jnp.zeros((1,1))
		self.normal_spos = jnp.zeros((1,1))
		self.minima = jnp.zeros((1,7))
	
	#We need the following two methods to tell jax what is and what is not static
	def _tree_flatten(self):
		children = (self.min_mem,self.mem_structure,self.result_grid_c,self.result_grid,self.normal_spos,self.curva,self.mem_structure_im,self.mem_structure_om,self.minima)
		aux_data = {"int_data":self.int_data,"Data":self.data,"Poses":self.poses,"Bead_types":self.bead_types,"All_Poses":self.all_poses,
		"B_vals":self.b_vals,"Surface_poses":self.surface_poses,"Sphere_poses":self.spheres,
			"Surface":self.surface,"Surface_b_vals":self.surf_b_vals,
			"WBmins":self.W_B_mins,"LH1B_mins":self.LH1_B_mins,"LH2B_mins":self.LH2_B_mins,
			"LH3B_mins":self.LH3_B_mins,"LH4B_mins":self.LH4_B_mins,"LT1B_mins":self.LT1_B_mins,
			"LT2B_mins":self.LT2_B_mins,"ChargeBmins":self.Charge_B_mins,"Peri":self.peri,"numa":self.numa,
			"numb":self.numb,"cuta":self.cut_a,"cutb":self.cut_b,"Nums":self.nums,"top_min":self.only_top,
			"red_cpus":self.red_cpus,"lssa":self.lss_a,"lssb":self.lss_b,"no_mins":self.no_mins,"force_calc":self.force_calc,"pg_layer":self.pg_layer_pred,
			"mem_Data":self.mem_data,"memop":self.memopt,"cm":self.charge_mult,"cmo":self.charge_mult_om,"dbmem":self.dbmem}
		return (children,aux_data)
	
	@classmethod
	def _tree_unflatten(cls,aux_data,children):
		obj = cls(aux_data["Data"],aux_data["int_data"],aux_data["Peri"],aux_data["force_calc"],aux_data["pg_layer"],aux_data["mem_Data"],aux_data["memop"],aux_data["dbmem"])
		obj.min_mem = children[0]
		obj.mem_structure = children[1]
		obj.mem_structure_im = children[6]
		obj.mem_structure_om = children[7]
		obj.no_mins = aux_data["no_mins"]
		obj.result_grid_c = children[2]
		obj.result_grid = children[3]
		obj.normal_spos = children[4]
		obj.numa = aux_data["numa"]
		obj.numb = aux_data["numb"]
		obj.cut_a = aux_data["cuta"]
		obj.cut_b = aux_data["cutb"]
		obj.curva = children[5]
		obj.lss_a = aux_data["lssa"]
		obj.lss_b = aux_data["lssb"]
		obj.charge_mult = aux_data["cm"]
		obj.charge_mult_om = aux_data["cmo"]
		obj.minima = children[8]
		return obj



	### Begining of orientation code ###
	
	#We use sigmoids to create a smoothly varying funtion to determin the potential of a single point
	@jax.jit
	def smjh(self,x,grad,bead_num,mem_structure):
		bd1 = (1.0-sj(x-mem_structure[0],grad))*self.W_B_mins[bead_num]+sj(x-mem_structure[0],grad)*self.LH1_B_mins[bead_num]
		bd2 = (1.0-sj(x-mem_structure[1],grad))*bd1+sj(x-mem_structure[1],grad)*self.LH2_B_mins[bead_num]
		bd3 = (1.0-sj(x-mem_structure[2],grad))*bd2+sj(x-mem_structure[2],grad)*self.LH3_B_mins[bead_num]
		bd4 = (1.0-sj(x-mem_structure[3],grad))*bd3+sj(x-mem_structure[3],grad)*self.LT1_B_mins[bead_num]
		bd5 = (1.0-sj(x-mem_structure[4],grad))*bd4+sj(x-mem_structure[4],grad)*self.LH3_B_mins[bead_num]
		bd6 = (1.0-sj(x-mem_structure[5],grad))*bd5+sj(x-mem_structure[5],grad)*self.LH2_B_mins[bead_num]
		bd7 = (1.0-sj(x-mem_structure[6],grad))*bd6+sj(x-mem_structure[6],grad)*self.LH1_B_mins[bead_num]
		bd8 = (1.0-sj(x-mem_structure[7],grad))*bd7+sj(x-mem_structure[7],grad)*self.W_B_mins[bead_num]
		return bd8-self.W_B_mins[bead_num]
		
	#Currently unsure which one to use. It doesn't seem to have a large effect really
	@jax.jit
	def smj(self,x,grad,bead_num,mem_structure):
		l1min = (self.W_B_mins[bead_num]+self.LH1_B_mins[bead_num]-jnp.abs(self.W_B_mins[bead_num]-self.LH1_B_mins[bead_num]))/2
		l2min = (self.W_B_mins[bead_num]+self.LH2_B_mins[bead_num]-jnp.abs(self.W_B_mins[bead_num]-self.LH2_B_mins[bead_num]))/2
		l3min = (self.W_B_mins[bead_num]+self.LH3_B_mins[bead_num]-jnp.abs(self.W_B_mins[bead_num]-self.LH3_B_mins[bead_num]))/2
		bd1 = (1.0-sj(x-mem_structure[0],grad))*self.W_B_mins[bead_num]+sj(x-mem_structure[0],grad)*l1min
		bd2 = (1.0-sj(x-mem_structure[1],grad))*bd1+sj(x-mem_structure[1],grad)*l2min
		bd3 = (1.0-sj(x-mem_structure[2],grad))*bd2+sj(x-mem_structure[2],grad)*l3min
		bd4 = (1.0-sj(x-mem_structure[3],grad))*bd3+sj(x-mem_structure[3],grad)*self.LT1_B_mins[bead_num]
		bd5 = (1.0-sj(x-mem_structure[4],grad))*bd4+sj(x-mem_structure[4],grad)*l3min
		bd6 = (1.0-sj(x-mem_structure[5],grad))*bd5+sj(x-mem_structure[5],grad)*l2min
		bd7 = (1.0-sj(x-mem_structure[6],grad))*bd6+sj(x-mem_structure[6],grad)*l1min
		bd8 = (1.0-sj(x-mem_structure[7],grad))*bd7+sj(x-mem_structure[7],grad)*self.W_B_mins[bead_num]
		return bd8-self.W_B_mins[bead_num]
		
	
	#This is a function for calculating an estimate of the potential of the PG layer at a point
	@jax.jit
	def pg_pot(self,x,grad,bead_num):
		bd1 = (1.0-sj(x+self.pg_thickness/2,grad))*self.pg_water[bead_num]+sj(x+self.pg_thickness/2,grad)*self.pg_layer[bead_num]
		bd2 = (1.0-sj(x-self.pg_thickness/2,grad))*bd1+sj(x-self.pg_thickness/2,grad)*self.pg_water[bead_num]
		return bd2-self.pg_water[bead_num]

		
	#We define new functions that let us have a double membrane (OM/IM)
	@jax.jit
	def dmsj(self,x,zdist,grad,bead_num,mem_structure_im,mem_structure_om):
		def not_zero():
			def memcon_s():
				return mem_structure_im.copy()
			def memcon_d():
				return mem_structure_om.copy()
			mem_structure = jax.lax.cond(x > 0,memcon_d,memcon_s)
			return self.smj(-jnp.abs(x)+zdist,grad,bead_num,mem_structure)
		def zero():
			#jax.debug.print("{x}",x=mem_structure_im)
			return self.smj(-jnp.abs(x)+zdist,grad,bead_num,mem_structure_im)
		return jax.lax.cond(not self.dbmem,zero,not_zero)
		

	#We define a smooth value indicating if a bead is in the membrane
	@jax.jit
	def sibj(self,x,grad,mem_structure):
		bd1 = sj(x-mem_structure[0],grad)
		bd2 = (1.0-sj(x-mem_structure[7],grad))*bd1
		return bd2
		

	
	#A function to calculate the potential of a bead at a given z position
	#This is done by calculating the potential of each free direction 
	@jax.jit
	def calc_pot_at_z_jit(self,z,zdist,ball,bead_type,mem_structure_im,mem_structure_om):
		smoothness = 2
		tot_pot_count = jnp.zeros(2)
		def calc_pot_fun_1(tot_pot_count,ind):
			def pot_cond_0(tot_pot_count):
				return tot_pot_count
			def not_pot_cond_0(tot_pot_count):
				zpos = ball[ind,2]+z
				zpos_abs = jnp.abs(zpos)
				tot_pot_count = tot_pot_count.at[1].set(tot_pot_count[1]+1)
				tot_pot_count = tot_pot_count.at[0].set(tot_pot_count[0]+self.dmsj(zpos,zdist,smoothness,bead_type,mem_structure_im,mem_structure_om))
				return tot_pot_count
			tot_pot_count = jax.lax.cond(jnp.linalg.norm(ball[ind]) < 1e-5,pot_cond_0,not_pot_cond_0,tot_pot_count)
			return tot_pot_count,ind
		tot_pot_count,_ = jax.lax.scan(calc_pot_fun_1,tot_pot_count,jnp.arange(ball.shape[0]))
		tot_pot = tot_pot_count[0]/(tot_pot_count[1]+1e-5)
		return tot_pot
		
	#This is function fot calculating the potential of the PG layer for a bead at a z value
	@jax.jit
	def pg_pot_at_z(self,z,ball,bead_type):
		smoothness = 2
		tot_pot_count = jnp.zeros(2)
		def calc_pot_fun_1(tot_pot_count,ind):
			def pot_cond_0(tot_pot_count):
				return tot_pot_count
			def not_pot_cond_0(tot_pot_count):
				zpos = ball[ind,2]+z
				zpos_abs = jnp.abs(zpos)
				tot_pot_count = tot_pot_count.at[1].set(tot_pot_count[1]+1)
				tot_pot_count = tot_pot_count.at[0].set(tot_pot_count[0]+self.pg_pot(zpos,smoothness,bead_type))
				#jax.debug.print("{x}",x=self.pg_pot(zpos,smoothness,bead_type))
				return tot_pot_count
			tot_pot_count = jax.lax.cond(jnp.linalg.norm(ball[ind]) < 1e-5,pot_cond_0,not_pot_cond_0,tot_pot_count)
			return tot_pot_count,ind
		tot_pot_count,_ = jax.lax.scan(calc_pot_fun_1,tot_pot_count,jnp.arange(ball.shape[0]))
		tot_pot = tot_pot_count[0]/(tot_pot_count[1]+1e-5)
		return tot_pot
	
	#This is a more accurate estimation of the potential due to electrostatic effects
	@jax.jit
	def new_charge(self,z,zdist,bead_type,mem_structure_im,mem_structure_om):
		charge_const = -(1.6*1.6*10000)/(4*jnp.pi*15*8.86)
		charge_const_im = charge_const*(self.charge_mult)*self.Charge_B_mins[bead_type]
		charge_const_om = charge_const*(self.charge_mult_om)*self.Charge_B_mins[bead_type]
		tot_charge = 0
		zpos = z+zdist
		grid_ex = 10
		grid_nums = 10
		dx = (2*(grid_ex)/grid_nums)*(2*(grid_ex)/grid_nums)
		grid = jnp.linspace(-grid_ex,grid_ex,grid_nums)
		
		def in_charge(x,msa,msb):
			return x
		def not_in_charge(x,msa,msb):
			def above(x,msa,msb):
				retval = msa
				return retval
			def below(x,msa,msb):
				retval = msb
				return retval
			retval = jax.lax.cond(x>msa,above,below,x,msa,msb)
			return retval
		zposa = jax.lax.cond((zpos<mem_structure_im[3])*(zpos>mem_structure_im[0]),in_charge,not_in_charge,zpos,mem_structure_im[3],mem_structure_im[0])
		zposb = jax.lax.cond((zpos<mem_structure_im[7])*(zpos>mem_structure_im[4]),in_charge,not_in_charge,zpos,mem_structure_im[7],mem_structure_im[4])
		charge_val = 0
		def calc_charge_fun(tot_charge,ind):
			zpos = z+tot_charge[2]
			def calc_charge_fun2(tot_charge,ind):
				point_a = jnp.array([0.0,0.0,zpos])
				point_b = jnp.array([grid[ind],grid[ind_fix],tot_charge[1]+1e-5])
				#jax.debug.print("{x}",x=tot_charge[1])
				dist = jnp.linalg.norm(point_a-point_b)
				#jax.debug.print("{x}",x=dist)
				def not_cutoff(tot_charge):
					def lfive(tot_charge):
						tot_charge = tot_charge.at[0].set(tot_charge[0]+dx*(tot_charge[3]/5.0-tot_charge[3]/10.0))
						return tot_charge
					def gfive(tot_charge):
						tot_charge = tot_charge.at[0].set(tot_charge[0]+dx*(tot_charge[3]/(jnp.abs(dist))-tot_charge[3]/10.0))
						return tot_charge
					tot_charge = jax.lax.cond(dist < 5, lfive,gfive,tot_charge)
					#tot_charge += charge_const/(jnp.abs(dist)+1e-5)
					return tot_charge
				def cutoff(tot_charge):
					return tot_charge
				tot_charge = jax.lax.cond(dist > 10.0,cutoff,not_cutoff,tot_charge)
				return tot_charge,ind
			ind_fix=ind
			tot_charge,_=jax.lax.scan(calc_charge_fun2,tot_charge,jnp.arange(grid_nums))
			return tot_charge,ind
		retval,_=jax.lax.scan(calc_charge_fun,jnp.array([0,zposa,zdist,charge_const_im]),jnp.arange(grid_nums))
		retvalb,_=jax.lax.scan(calc_charge_fun,jnp.array([0,zposb,zdist,charge_const_im]),jnp.arange(grid_nums))
		charge_val = retval[0]*1.01+retvalb[0]
		def zero(charge_val):
			return charge_val
		def not_zero(charge_val):
			zpos = z-zdist
			zposa = jax.lax.cond((zpos<mem_structure_om[3])*(zpos>mem_structure_om[0]),in_charge,not_in_charge,zpos,mem_structure_om[3],mem_structure_om[0])
			zposb = jax.lax.cond((zpos<mem_structure_om[7])*(zpos>mem_structure_om[4]),in_charge,not_in_charge,zpos,mem_structure_om[7],mem_structure_om[4])
			retvalc,_=jax.lax.scan(calc_charge_fun,jnp.array([0,zposa,-zdist,charge_const_om]),jnp.arange(grid_nums))
			retvald,_=jax.lax.scan(calc_charge_fun,jnp.array([0,zposb,-zdist,charge_const_om]),jnp.arange(grid_nums))
		
			charge_val += retvalc[0]*1.01 + retvald[0]
			
			return charge_val
		charge_val = jax.lax.cond(not self.dbmem,zero,not_zero,charge_val)#?
		#
		return charge_val
		
	#As above but for the PG layer
	@jax.jit
	def pg_charge_fun(self,z,bead_type):
		charge_const = -(1.6*1.6*10000)/(4*jnp.pi*15*8.86)
		charge_const = charge_const*(self.pg_charge)*self.Charge_B_mins[bead_type]
		tot_charge = 0
		zpos = z
		grid_ex = 10
		grid_nums = 5
		dx = (2*(grid_ex)/grid_nums)*(2*(grid_ex)/grid_nums)
		grid = jnp.linspace(-grid_ex,grid_ex,grid_nums)
		def in_charge(x,msa,msb):
			return x
		def not_in_charge(x,msa,msb):
			def above(x,msa,msb):
				retval = msa
				return retval
			def below(x,msa,msb):
				retval = msb
				return retval
			retval = jax.lax.cond(x>msa,above,below,x,msa,msb)
			return retval
		zposa = jax.lax.cond((zpos<self.pg_thickness/2)*(zpos>-self.pg_thickness/2),in_charge,not_in_charge,zpos,self.pg_thickness/2,-self.pg_thickness/2)
		charge_val = 0
		def calc_charge_fun(tot_charge,ind):
			def calc_charge_fun2(tot_charge,ind):
				point_a = jnp.array([0.0,0.0,zpos])
				point_b = jnp.array([grid[ind],grid[ind_fix],tot_charge[1]+1e-5])
				#jax.debug.print("{x}",x=tot_charge[1])
				dist = jnp.linalg.norm(point_a-point_b)
				#jax.debug.print("{x}",x=dist)
				def not_cutoff(tot_charge):
					def lfive(tot_charge):
						tot_charge = tot_charge.at[0].set(tot_charge[0]+dx*(charge_const/5.0-charge_const/10.0))
						return tot_charge
					def gfive(tot_charge):
						tot_charge = tot_charge.at[0].set(tot_charge[0]+dx*(charge_const/(jnp.abs(dist))-charge_const/10.0))
						return tot_charge
					tot_charge = jax.lax.cond(dist < 5, lfive,gfive,tot_charge)
					#tot_charge += charge_const/(jnp.abs(dist)+1e-5)
					return tot_charge
				def cutoff(tot_charge):
					return tot_charge
				tot_charge = jax.lax.cond(dist > 10.0,cutoff,not_cutoff,tot_charge)
				#tot_charge += charge_const/(jnp.abs(dist)+1e-5)
				return tot_charge,ind
			ind_fix=ind
			tot_charge,_=jax.lax.scan(calc_charge_fun2,tot_charge,jnp.arange(5))
			return tot_charge,ind
		retval,_=jax.lax.scan(calc_charge_fun,jnp.array([0,zposa]),jnp.arange(5))
		charge_val = retval[0]
		return charge_val
			
		
	#A function that calculates the penalty for displacing lipids
	@jax.jit
	def get_lipid_disp_jit(self,poses,disp_penalty,zdist,mem_structure):
		tot_pot_disp = jnp.zeros(2)
		def get_lipd_fun_1(tot_pot_disp,ind):
			def in_bilayer(tot_pot_disp):
				tot_pot_disp = tot_pot_disp.at[0].set(tot_pot_disp[0]+disp_penalty*self.b_vals[ind]*self.sibj(poses[ind,2]+zdist,2,mem_structure))
				tot_pot_disp = tot_pot_disp.at[1].set(tot_pot_disp[1]+1)
				return tot_pot_disp
			def not_in_bilayer(tot_pot_disp):
				return tot_pot_disp
			tot_pot_disp = jax.lax.cond((mem_structure[0] > poses[ind,2]+zdist)+(poses[ind,2]+zdist > mem_structure[7]),not_in_bilayer,in_bilayer,tot_pot_disp)
			return tot_pot_disp,ind
		tot_pot_disp,_ = jax.lax.scan(get_lipd_fun_1,tot_pot_disp,jnp.arange(self.poses.shape[0]))
		return tot_pot_disp
			
	#A function that calculates if a protein is fully ejected from the membrane
	@jax.jit
	def calc_in_water_jit(self,position,tol,mem_structure_im,mem_structure_om):
		zdist_temp = position[0]
		in_depth = position[1]
		ang1 = position[2:5]
		ang2 = position[5:7]
		ang1 /= jnp.linalg.norm(ang1)
		zdist = jnp.abs(zdist_temp*jnp.dot(ang1,jnp.array([0.0,0.0,1.0])))
		tester_poses = position_pointv2_jit(in_depth,ang1,ang2,self.surface_poses)
		def rot_spheres(carry,ind):
			new_spheres = position_pointv2_jit(0,ang1,ang2,self.spheres[ind])
			return carry,new_spheres
		_,test_spheres = jax.lax.scan(rot_spheres,0,jnp.arange(self.spheres.shape[0]))
		tot_pot = 0
		def calc_pot_fun_1(tot_pot,ind):
			tot_pot += jnp.abs(self.surf_b_vals[ind]*self.calc_pot_at_z_jit(tester_poses[ind,2],zdist,test_spheres[ind],self.bead_types[ind],mem_structure_im,mem_structure_om))
			tot_pot += jnp.abs(self.surf_b_vals[ind]*self.new_charge(tester_poses[ind,2],zdist,self.bead_types[ind],mem_structure_im,mem_structure_om))
			return tot_pot,ind
		tot_pot,_ = jax.lax.scan(calc_pot_fun_1,tot_pot,jnp.arange(tester_poses.shape[0]))
		#jax.debug.print("{x},{y}",x=zdist,y=tot_pot)
		return tot_pot<tol


	#A function that calculates the potential of the protein for a given position
	@jax.jit			
	def calc_pot_jit(self,position,mem_structure_im,mem_structure_om):
		disp_penalty = 0.025
		zdist_temp = position[0]
		in_depth = position[1]
		ang1 = position[2:5]
		ang2 = position[5:7]
		ang1 /= jnp.linalg.norm(ang1)
		zdist = jnp.abs(zdist_temp*jnp.dot(ang1,jnp.array([0.0,0.0,1.0])))
		tester_poses = position_pointv2_jit(in_depth,ang1,ang2,self.surface_poses)
		tester_fposes = position_pointv2_jit(in_depth,ang1,ang2,self.poses)
		def rot_spheres(carry,ind):
			new_spheres = position_pointv2_jit(0,ang1,ang2,self.spheres[ind])
			return carry,new_spheres
		_,test_spheres = jax.lax.scan(rot_spheres,0,jnp.arange(self.spheres.shape[0]))
		tot_pot = 0
		def calc_pot_fun_1(tot_pot,ind):
			tot_pot += self.surf_b_vals[ind]*self.calc_pot_at_z_jit(tester_poses[ind,2],zdist,test_spheres[ind],self.bead_types[ind],mem_structure_im,mem_structure_om)
			tot_pot += self.surf_b_vals[ind]*self.new_charge(tester_poses[ind,2],zdist,self.bead_types[ind],mem_structure_im,mem_structure_om)
			return tot_pot,ind
		tot_pot,_ = jax.lax.scan(calc_pot_fun_1,tot_pot,jnp.arange(tester_poses.shape[0]))
		ldisp_p = self.get_lipid_disp_jit(tester_fposes,disp_penalty,zdist,mem_structure_im)
		def zero(tot_pot):
			return tot_pot
		def not_zero(tot_pot):
			ldisp_n = self.get_lipid_disp_jit(tester_fposes,disp_penalty,-zdist,mem_structure_om)
			tot_pot += ldisp_n[0]
			return tot_pot
		tot_pot = jax.lax.cond(not self.dbmem,zero,not_zero,tot_pot)
		tot_pot += ldisp_p[0]
		return tot_pot
		
	#Function to calculate the potential of the PG layer at a given z
	@jax.jit			
	def calc_pot_pg(self,position,pg_z,mem_structure_im,mem_structure_om):
		charge_const = -(1.6*1.6*10000)/(4*jnp.pi*15*8.86)
		zdist_temp = position[0]
		in_depth = position[1]
		ang1 = position[2:5]
		ang2 = position[5:7]
		ang1 /= jnp.linalg.norm(ang1)
		zdist = jnp.abs(zdist_temp*jnp.dot(ang1,jnp.array([0.0,0.0,1.0])))
		tester_poses = position_pointv2_jit(in_depth,ang1,ang2,self.surface_poses)
		def rot_spheres(carry,ind):
			new_spheres = position_pointv2_jit(0,ang1,ang2,self.spheres[ind])
			return carry,new_spheres
		_,test_spheres = jax.lax.scan(rot_spheres,0,jnp.arange(self.spheres.shape[0]))
		tot_pot = 0
		def calc_pot_fun_1(tot_pot,ind):
			tot_pot += self.surf_b_vals[ind]*self.pg_pot_at_z(tester_poses[ind,2]+pg_z,test_spheres[ind],self.bead_types[ind])
			tot_pot += self.surf_b_vals[ind]*self.pg_charge_fun(tester_poses[ind,2]+pg_z,self.bead_types[ind])
			return tot_pot,ind
		tot_pot,_ = jax.lax.scan(calc_pot_fun_1,tot_pot,jnp.arange(tester_poses.shape[0]))
		#jax.debug.print("{x},{y}",x=self.mem_structure[0],y=self.pg_thickness/2)
		tot_pot += potential_between_sheets(zdist-pg_z+mem_structure_om[0]-self.pg_thickness/2,10,20,-self.sheet_charge*charge_const)+potential_between_sheets(zdist+pg_z+mem_structure_im[0]-self.pg_thickness/2,10,20,-self.sheet_charge*charge_const)
		return tot_pot
		

	#Differentiating the potential to get gradients for minimisation
	@jax.jit
	def pot_grad(self,position):
		return jax.grad(self.calc_pot_jit,argnums=0)(position,self.mem_structure_im,self.mem_structure_om)

	#A function that takes a weighted mean of points to try and get a good starting guess for insertion depth
	@jax.jit
	def get_hydrophobic_core_jit(self):
		hydro_cut_off = -3
		hydro_range = 10
		av_pos_count = jnp.zeros(4)
		def hydro_fun_1(carry,ind):
			av_countp = 0
			def is_hydro_1(av_countp):
				def hydro_fun2(ind_fix,ind):
					def is_close(av_countpp):
						def is_hydro_2(av_countpp):
							av_countpp += 1
							return av_countpp
						def not_hydro_2(av_countpp):
							return av_countpp
						av_countpp = jax.lax.cond(self.W_B_mins[self.bead_types[ind]] > hydro_cut_off,is_hydro_2,not_hydro_2,av_countpp)
						return av_countpp
					def not_close(av_countpp):
						return av_countpp
					ind_fix = ind_fix.at[1].set(jax.lax.cond(jnp.linalg.norm(self.surface_poses[ind]-self.surface_poses[ind_fix[0]]) < hydro_range,is_close,not_close,ind_fix[1]))
					return ind_fix,ind
				out,_ = jax.lax.scan(hydro_fun2,jnp.array([ind,av_countp]),jnp.arange(self.surface_poses.shape[0]))
				av_countp = out[1]
				return av_countp
			def not_hydro_1(av_countp):				
				return av_countp
			av_countp = jax.lax.cond(self.W_B_mins[self.bead_types[ind]] > hydro_cut_off,is_hydro_1,not_hydro_1,av_countp)
			return carry,av_countp
		_,all_count = jax.lax.scan(hydro_fun_1,0,jnp.arange(self.surface_poses.shape[0]))
		lower_bound = jnp.max(all_count)*0.9
		
		def hydro_fun_3(av_pos_count,ind):
			def is_dens(av_pos_count):
				av_pos_count = av_pos_count.at[3].set(av_pos_count[3]+all_count[ind]*all_count[ind])
				av_pos_count = av_pos_count.at[:3].set(av_pos_count[:3]+all_count[ind]*all_count[ind]*self.surface_poses[ind])
				return av_pos_count
			def is_not_dens(av_pos_count):
				return av_pos_count
			av_pos_count = jax.lax.cond(all_count[ind] >=lower_bound,is_dens,is_not_dens,av_pos_count)
			return av_pos_count,ind
		av_pos_count,_ = jax.lax.scan(hydro_fun_3,av_pos_count,jnp.arange(self.surface_poses.shape[0]))
		return av_pos_count[:3]/av_pos_count[3]
		

	#An improved version of the above
	def hydro_core_imp(self,surface_poses,bead_types):
		num_binsx = jnp.array(jnp.floor(jnp.max(jnp.abs(surface_poses[:,0]))/10),dtype=int)
		num_binsy = jnp.array(jnp.floor(jnp.max(jnp.abs(surface_poses[:,1]))/10),dtype=int)
		num_binsz = jnp.array(jnp.floor(jnp.max(jnp.abs(surface_poses[:,2]))/10),dtype=int)
		rangerx = jnp.max(surface_poses[:,0])
		rangery = jnp.max(surface_poses[:,1])
		rangerz = jnp.max(surface_poses[:,2])
		x = jnp.linspace(-rangerx,rangerx,num_binsx)
		y = jnp.linspace(-rangery,rangery,num_binsy)
		z = jnp.linspace(-rangerz,rangerz,num_binsz)
		xm,ym,zm = jnp.meshgrid(x,y,z)
		ys = jnp.zeros((num_binsx,num_binsy,num_binsz,2))
		def hydro_fingerprint_fun(ys,ind):
			xpos = surface_poses[ind][0]
			ypos = surface_poses[ind][1]
			zpos = surface_poses[ind][2]
			hydro_val =self.W_B_mins[bead_types[ind]]
			x_ind = jnp.array(num_binsx*(xpos+rangerx)/(rangerx*2),dtype=int)
			y_ind = jnp.array(num_binsy*(ypos+rangery)/(rangery*2),dtype=int)
			z_ind = jnp.array(num_binsz*(zpos+rangerz)/(rangerz*2),dtype=int)
			ys = ys.at[x_ind,y_ind,z_ind,0].set(ys[x_ind,y_ind,z_ind,0]+hydro_val)
			ys = ys.at[x_ind,y_ind,z_ind,1].set(ys[x_ind,y_ind,z_ind,1]+1)
			return ys,ind
		ys,_ = jax.lax.scan(hydro_fingerprint_fun,ys,jnp.arange(surface_poses.shape[0]))
		y_vals = ys[:,:,:,0]/(ys[:,:,:,1]+1e-5)
		y_vals = y_vals.ravel()
		xmf = xm.ravel()
		ymf = ym.ravel()
		zmf = zm.ravel()
		def hydro_norm(y_vals,ind):
			def zero(y_vals):
				y_vals = y_vals.at[ind].set(-8)
				return y_vals
			def not_zero(y_vals):
				return y_vals
			y_vals = jax.lax.cond(y_vals[ind] == 0,zero,not_zero,y_vals)
			return y_vals,ind
		y_vals,_ = jax.lax.scan(hydro_norm,y_vals,jnp.arange(y_vals.shape[0]))
		is_hydro = xmf[y_vals>-3].size
		hydro_core = jnp.zeros(3)
		hydro_corex = jnp.mean(xmf[y_vals>-3])
		hydro_corey = jnp.mean(ymf[y_vals>-3])
		hydro_corez = jnp.mean(zmf[y_vals>-3])
		def empty():
			return jnp.mean(surface_poses,axis=0)
		def not_empty():
			return jnp.array([hydro_corex,hydro_corey,hydro_corez])
		hydro_core = jax.lax.cond(is_hydro == 0,empty,not_empty)
		return hydro_core
		
	#A function to finds the minimal insertion depth (within range) via grid search
	#This is in general expensive but is good for peripheral proteins
	def z_pot(self,max_iter,ranger,zdir,xydir,shift = 0):
		zs = jnp.linspace(-ranger+shift,ranger+shift,max_iter)#20
		pots = jnp.zeros(max_iter)
		def calc_pot_fun(pots,ind):
			posi = jnp.zeros(7)
			posi = posi.at[1].set(zs[ind])
			posi = posi.at[2:5].set(zdir)
			posi = posi.at[5:7].set(xydir)
			pots = pots.at[ind].set(self.calc_pot_jit(posi,self.mem_structure_im,self.mem_structure_om))
			return pots,ind
		pots,_ = jax.lax.scan(calc_pot_fun,pots,jnp.arange(max_iter))
		best_ind = jnp.argmin(pots)
		return zs[best_ind]
		
		
	#The same as above but it plots the potential (for debugging purposes)
	def z_pot_plot(self,max_iter,ranger,zdir,xydir):
		zs = jnp.linspace(-ranger,ranger,max_iter)#20
		pots = jnp.zeros(max_iter)
		def calc_pot_fun(pots,ind):
			posi = jnp.zeros(7)
			posi = posi.at[1].set(zs[ind])
			posi = posi.at[2:5].set(zdir)
			posi = posi.at[5:7].set(xydir)
			pots = pots.at[ind].set(self.calc_pot_jit(posi))
			return pots,ind
		pots,_ = jax.lax.scan(calc_pot_fun,pots,jnp.arange(max_iter))
		best_ind = jnp.argmin(pots)
		plt.plot(pots)
		plt.ylabel("Potential energy")
		plt.xlabel("Z")
		plt.title("Potential energy against z position relative to center of membrane")
		plt.show()
	
	#This function plots the PG layer potential surface(curve)
	def pg_pot_plot(self,nums,min_ind,mem_structure_im,mem_structure_om):
		if(self.curva):
			mins = np.array(self.minima_c)
		else:
			mins = np.array(self.minima)
		no_mins = (mins.shape[0]//4)
		min_poses = mins[:no_mins]
		ranger = jnp.abs(min_poses[min_ind][0]*jnp.cos(min_poses[min_ind][3]))
		xs = jnp.linspace(-ranger-self.mem_structure[0]+self.pg_thickness/2,ranger+self.mem_structure[0]-self.pg_thickness/2,nums)
		start_zdir = position_point_jit(0,min_poses[min_ind][2],min_poses[min_ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
		start_xydir =  position_point_jit(0,min_poses[min_ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
		test_pos = jnp.concatenate([jnp.array([min_poses[min_ind][0],min_poses[min_ind][1]]),start_zdir,start_xydir])
		ys = [self.calc_pot_pg(test_pos,z,mem_structure_im,mem_structure_om) for z in xs]
		plt.plot(xs,ys)
		bot,top = plt.ylim()
		plt.ylim(0,3000)
		plt.show()
	
	#This function evaluates the position of the PG layer 
	def get_pg_pos(self,nums,min_ind,mem_structure_im,mem_structure_om,rank_dir):
		if(self.curva):
			mins = np.array(self.minima_c)
		else:
			mins = np.array(self.minima)
		#mins = self.minima
		no_mins = (mins.shape[0]//4)
		min_poses = mins[:no_mins]
		ranger = jnp.abs(min_poses[min_ind][0]*jnp.cos(min_poses[min_ind][3]))
		xs = jnp.linspace(-ranger-self.mem_structure[0]+self.pg_thickness/2,ranger+self.mem_structure[0]-self.pg_thickness/2,nums)
		start_zdir = position_point_jit(0,min_poses[min_ind][2],min_poses[min_ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
		start_xydir =  position_point_jit(0,min_poses[min_ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
		test_pos = jnp.concatenate([jnp.array([min_poses[min_ind][0],min_poses[min_ind][1]]),start_zdir,start_xydir])
		ps = jnp.zeros(nums)
		def pg_pot_fun(ps,ind):
			ps = ps.at[ind].set(self.calc_pot_pg(test_pos,xs[ind],mem_structure_im,mem_structure_om))
			return ps, ind
		ps,_ = jax.lax.scan(pg_pot_fun,ps,jnp.arange(nums))
		plt.plot(xs[jnp.abs(ps)<3000],ps[jnp.abs(ps)<3000])
		bot,top = plt.ylim()
		#plt.ylim(-1000,3000)
		plt.title("Potential energy of PG layer at z")
		plt.ylabel("Potential energy(rel)")
		plt.xlabel("z")
		plt.savefig(rank_dir+"PG_potential_curve.png")
		plt.clf()
		best_ind = jnp.argmin(ps)
		return xs[best_ind]
	
	#This function probably doesnt work anymore
	@partial(jax.jit,static_argnums = 1)	
	def z_force_plot(self,max_iter,ranger,cen,zdir,xydir):
		zs = jnp.linspace(-ranger,ranger,max_iter)#20
		pots = jnp.zeros(max_iter)
		def calc_pot_fun(pots,ind):
			posi = jnp.zeros(7)
			posi = posi.at[1].set(zs[ind]+cen)
			posi = posi.at[2:5].set(zdir)
			posi = posi.at[5:7].set(xydir)
			pots = pots.at[ind].set(-self.pot_grad(posi)[1])
			return pots,ind
		pots,_ = jax.lax.scan(calc_pot_fun,pots,jnp.arange(max_iter))
		plus_max = jnp.min(pots[max_iter//2:])
		minus_max = jnp.max(pots[:max_iter//2])
		return plus_max, minus_max


	#A good starting point for periplasmic spanning proteins is very important
	#This is as the potential energy landscape in the z direction has many minima
	#The current method used certainly biases the final position
	#However for these proteins the starting positions is known to be close already
	#Overall not much additional work is needed if the starting z pos is good.

	#A function that uses the hydrophobic core functions to determine a good starting periplasmic space width
	def calc_start_for_ps(self):
		hydro_core =  self.hydro_core_imp(self.surface_poses,self.bead_types)
		surface_poses = self.surface_poses - hydro_core
		bead_pos = self.bead_types[surface_poses[:,2]>0]
		bead_neg = self.bead_types[surface_poses[:,2]<0]
		surface_poses_pos = surface_poses[surface_poses[:,2]>0]
		surface_poses_neg = surface_poses[surface_poses[:,2]<0]
		
		pos_core = self.hydro_core_imp(surface_poses_pos,bead_pos)
		neg_core = self.hydro_core_imp(surface_poses_neg,bead_neg)
		zdist = (pos_core-neg_core)[2]/2
		start_z = (pos_core+neg_core)[2]/2
		return zdist,start_z
		
		 
	#A function that uses the z potemtail scanning functions to determine a good starting periplasmic space width
	def calc_start_for_ps_imp(self):
		#hydro_core =  self.hydro_core_imp(self.surface_poses,self.bead_types)
		maxz = jnp.max(self.surface_poses[:,2])
		minz = jnp.min(self.surface_poses[:,2])
		#jax.debug.print("{x},{y}",x=maxz,y=minz)
		half_mem = self.memt_total/2.0
		rplus = jnp.abs(maxz/2)-half_mem
		rminus = jnp.abs(minz/2)-half_mem
		zplus = self.z_pot(100,rplus-rplus/4,jnp.array([0,0,1]),jnp.array([1,0]),shift = maxz/2+rplus/4)
		zminus = self.z_pot(100,rminus-rminus/4,jnp.array([0,0,1]),jnp.array([1,0]),shift = minz/2-rminus/4)
		#jax.debug.print("{x},{y}",x=zplus,y=zminus)
		
		zdist = (zplus-zminus)/2
		start_z = (zplus+zminus)/2
		return zdist,start_z
		
	#A function the minimises potential from a given starting position
	def minimise(self,starting_pos,max_iter):		
		pot = 0
		data = jnp.zeros(22)
		num = 7
		gamma = jnp.array([self.gamma_val]*num,)
		ep = 1e-8
		tol = 1e-10
		data = data.at[:num].set(starting_pos[:num])
		decend = 0.0
		def zero(decend):
			return decend
		def not_zero(decend):
			decend = 1.0
			return decend
		decend = jax.lax.cond(not self.dbmem,zero,not_zero,decend)
		data = data.at[num*2:num*3].set(jnp.array([self.lr_pos]*num))#???
		data = data.at[num*3].set(1)
		def min_fun_1(data,ind):
			grad = jnp.zeros(num)
			def go_g(grad):
				grad = jnp.array(self.pot_grad(data[:num]))
				return grad
			def stop_g(grad):
				return grad
			grad = jax.lax.cond(data[num*3] > 0.5,go_g,stop_g,grad)
			def close(data):
				data = data.at[num*3].set(0)
				return data
			def far(data):
				data = data.at[num*3].set(1)
				return data
			data = jax.lax.cond(jnp.linalg.norm(grad)<tol,close,far,data)	
			data = data.at[num:num*2].set(data[num:num*2]*gamma + (1-gamma)*grad*grad)
			rms_grad = jnp.sqrt(ep + data[num:num*2])
			rms_change = jnp.sqrt(ep+data[num*2:num*3])
			change = -(rms_change/rms_grad)*grad
			change = change.at[0].set(change[0]*decend)
			data = data.at[:num].set(data[:num]+change)
			diff_data = jnp.array([data[0],-data[1],data[2],jnp.pi+data[3]])
			data = data.at[num*2:num*3].set(gamma*data[num*2:num*3]+(1-gamma)*change*change)
			return data,ind
		final_data,_ = jax.lax.scan(min_fun_1,data,None,length = max_iter)
		final_pot = self.calc_pot_jit(final_data[:num],self.mem_structure_im,self.mem_structure_om)
		flipped_data = final_data.copy()
		
		#There were issue flipping the protein so a very explicit method is used (can be replaced to be something that is cleaner)
		normed = jnp.array([flipped_data[5],flipped_data[6],0.0])
		normed2 = jnp.array([flipped_data[2],flipped_data[3],flipped_data[4]])
		
		normed2 /= jnp.linalg.norm(normed2)
		
		direc1 = jnp.cross(normed2,jnp.array([0.0,0.0,1.0]))
		ang1 = (1-1e-8)*jnp.dot(normed2,jnp.array([0.0,0.0,1.0]))
		
		rot1qi = jnp.array([ang1+1,direc1[0],direc1[1],direc1[2]])
		rot1qi /= jnp.linalg.norm(rot1qi)
		rot1q = qcong_jit(rot1qi)

		
		xydir_cor = jnp.array([flipped_data[5],flipped_data[6],0.0])
		xydir_cor /= jnp.linalg.norm(xydir_cor)
		
		rotated_zdir = position_point_jit(0.0,0.0,jnp.pi,jnp.array([normed2]))[0]
		xycqp = jnp.array([0.0,xydir_cor[0],xydir_cor[1],xydir_cor[2]])
		xycrot_qp = qmul_jit(rot1q,qmul_jit(xycqp,rot1qi))
		xycrot_p = xycrot_qp[1:]
		
		rotated_xydir = position_point_jit(0.0,0.0,jnp.pi,jnp.array([xycrot_p]))[0]
		
		direc1 = jnp.cross(rotated_zdir,jnp.array([0.0,0.0,1.0]))
		ang1 = (1-1e-8)*jnp.dot(rotated_zdir,jnp.array([0.0,0.0,1.0]))
		
		rot1q = jnp.array([ang1+1,direc1[0],direc1[1],direc1[2]])
		rot1q /= jnp.linalg.norm(rot1q)
		rot1qi = qcong_jit(rot1q)
		
		xycqp = jnp.array([0.0,rotated_xydir[0],rotated_xydir[1],rotated_xydir[2]])
		xycrot_qp = qmul_jit(rot1q,qmul_jit(xycqp,rot1qi))
		xycrot_p = xycrot_qp[1:]

		flipped_data = flipped_data.at[2:5].set(rotated_zdir)
		flipped_data = flipped_data.at[5:7].set(xycrot_p[:2])

		flipped_data = flipped_data.at[1].set(-flipped_data[1])
		final_flipped = self.calc_pot_jit(flipped_data[:num],self.mem_structure_im,self.mem_structure_om)
		def flip(final_data):
			return flipped_data
		def no_flip(final_data):
			return final_data
		final_data = jax.lax.cond(final_pot > final_flipped,flip,no_flip,final_data)
		final_pot_pos = jnp.zeros(num+1)
		final_pot_pos = final_pot_pos.at[:num].set(final_data[:num])
		final_pot_pos = final_pot_pos.at[num].set(self.calc_pot_jit(final_data[:num],self.mem_structure_im,self.mem_structure_om))
		#final_pot_pos = final_pot_pos.at[-1].set(final_pot - final_flipped)
		return final_pot_pos
				
	#We turn minimise into a paralelised method
	@partial(jax.jit,static_argnums=2)
	def minimise_p(self,starting_pos,max_iter):
		return jax.pmap(self.minimise,static_broadcasted_argnums=1,in_axes=(0,None))(starting_pos,max_iter)


	#A function that minimises on a set of different starting positions. This is important as there can be local minima
	def minimise_on_grid(self,grid_size,start_z,zdist,angs,max_iter):
		pos_grid = jnp.zeros((grid_size,8))
		def min_plot_fun_1(pos_grid,ind):
			start_zdir = position_point_jit(0,angs[ind][1],angs[ind][0],jnp.array([[0.0,0.0,1.0]]))[0]
			start_xydir =  position_point_jit(0,angs[ind][1],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
			
			#Using a different starting insertion depth for peripheral proteins
			def is_peri():
				new_start_z = self.z_pot(100,self.memt_total,start_zdir,start_xydir)
				return new_start_z
			def is_not_peri():
				new_start_z = position_point_jit(0,angs[ind][1],angs[ind][0],jnp.array([start_z]))[0,2]
				return new_start_z
			new_start_z = jax.lax.cond(self.peri,is_peri,is_not_peri)
			pos = jnp.concatenate([jnp.array([zdist,new_start_z]),start_zdir,start_xydir,jnp.zeros(1)])
			pos_grid = pos_grid.at[ind].set(pos)
			return pos_grid,ind

		pos_grid,_ = jax.lax.scan(min_plot_fun_1,pos_grid,jnp.arange(grid_size))

		result_grid = jnp.zeros_like(pos_grid)
		no_runs = jnp.ceil(grid_size/no_cpu).astype(int)
		print("Batch size:",no_cpu)
		print("Number of batches:",no_runs)
		times_taken = jnp.zeros(no_runs)
		#This non JAX for loop is needed beacuse pmap doesn't interact well with scan
		for i in range(no_runs):
			timers = time.time()
			print(str(i+1)+"/"+str(no_runs))
			result_grid = result_grid.at[i::no_runs].set(self.minimise_p(pos_grid[i::no_runs],max_iter).block_until_ready())
			print("Time per single minimisation: "+str(np.round((time.time()-timers)/no_cpu,3))+" s")
			times_taken = times_taken.at[i].set((time.time()-timers))
			if(i > 5):
				time_rem = jnp.array((no_runs-(i+1))*jnp.mean(times_taken[i-5:i+1]),dtype=int)
			elif(i>1):
				time_rem = jnp.array((no_runs-(i+1))*jnp.mean(times_taken[1:i+1]),dtype=int)
			else:
				time_rem = jnp.array((no_runs-(i+1))*jnp.mean(times_taken[:i+1]),dtype=int)
			print("Estimated remaining time: {:02}h {:02}m {:02}s".format(time_rem//3600,(time_rem%3600)//60,time_rem%60))
		self.result_grid = result_grid


	#A function to normalise positions to allow comparisons
	def normalise_pos(self,position):
		#jax.debug.print("{x}",x=position[:2])
		new_pos = jnp.zeros(4)
		new_pos = new_pos.at[:2].set(position[:2])
		zdir_test_a = position[2:5]
		zdir_test_a /= jnp.linalg.norm(zdir_test_a)
		new_pos = new_pos.at[3].set(jnp.arccos(jnp.dot(zdir_test_a,jnp.array([0.0,0.0,1.0]))))
		xydir_test = jnp.array([position[5],position[6],0])
		zdir_test = jnp.array([position[2],position[3],0])
		anga = 0.0
		def z_dir_zero(anga,xydir_test,zdir_test):
			return 0.0
		def z_dir_not_zero(anga,xydir_test,zdir_test):
			xydir_test /= jnp.linalg.norm(xydir_test)
			zdir_test /= jnp.linalg.norm(zdir_test)
			anga = jnp.arctan2(jnp.dot(jnp.cross(zdir_test,xydir_test),jnp.array([0.0,0.0,1.0])),jnp.dot(xydir_test,zdir_test))
			return anga
		anga = jax.lax.cond(jnp.linalg.norm(zdir_test)<1e-5,z_dir_zero,z_dir_not_zero,anga,xydir_test,zdir_test)

		new_pos = new_pos.at[2].set(anga)
		
		def norm_fun_1(new_pos):
			new_pos = new_pos.at[3].set(new_pos[3]-2*jnp.pi)
			return new_pos
		def norm_cond_1(new_pos):
			return new_pos[3] > jnp.pi
		
		def norm_fun_2(new_pos):
			new_pos = new_pos.at[3].set(new_pos[3]+2*jnp.pi)
			return new_pos
		def norm_cond_2(new_pos):
			return new_pos[3] < -jnp.pi
			
		new_pos = jax.lax.while_loop(norm_cond_1,norm_fun_1,new_pos)
		new_pos = jax.lax.while_loop(norm_cond_2,norm_fun_2,new_pos)
		
		def is_neg(new_pos):
			new_pos = new_pos.at[3].set(-new_pos[3])
			new_pos = new_pos.at[2].set(new_pos[2]+jnp.pi)
			return new_pos
		def is_pos(new_pos):
			return new_pos
		new_pos = jax.lax.cond(new_pos[3]< 0,is_neg,is_pos,new_pos)
		
		def norm_fun_3(ang):
			ang += jnp.pi*2
			return ang
		def norm_cond_3(ang):
			return ang < 0
			
		def norm_fun_4(ang):
			ang -= jnp.pi*2
			return ang
		def norm_cond_4(ang):
			return ang > 2*jnp.pi
			
		new_pos = new_pos.at[2].set(jax.lax.while_loop(norm_cond_3,norm_fun_3,new_pos[2]))
		new_pos = new_pos.at[2].set(jax.lax.while_loop(norm_cond_4,norm_fun_4,new_pos[2]))
		
		def in_water(new_pos):
			new_pos = jnp.zeros_like(new_pos)
			new_pos = new_pos.at[1].set(-300)
			return new_pos
		def not_in_water(new_pos):
			return new_pos
		#jax.debug.print("{x}",x=self.calc_in_water_jit(position,1e-5))
		new_pos = jax.lax.cond(self.calc_in_water_jit(position,1e-5,self.mem_structure_im,self.mem_structure_om),in_water,not_in_water,new_pos)
		
		return new_pos

	#A function for calculating which positions are equivilant. 
	#This will return all unique minima and there frequency
	def collect_minima_info(self,grid_size):
		pos_tracker = jnp.array([[0,0,1]],dtype="float64")
		results = jnp.reshape(self.result_grid,(grid_size,8))
		res_spos = jnp.zeros((grid_size,1,3))
		color_grid = jnp.zeros((grid_size,3),dtype="float64")
		pot_grid = self.result_grid[:,-1]
		sposs = jnp.zeros((grid_size,4))
		
		def calc_spos(spos,ind):
			spos = spos.at[ind].set(self.normalise_pos(results[ind][:7]))
			return spos,ind
		sposs,_ = jax.lax.scan(calc_spos,sposs,jnp.arange(grid_size))
		#Calculating a simplified version of the oriented position
		def calc_poses_fun_1(res_spos,ind):
			spos = sposs[ind]
			part_pos = position_point(0,spos[2],spos[3],pos_tracker)
			res_spos = res_spos.at[ind].set(position_point(0,-spos[2],0,part_pos))
			return res_spos,ind
			
		res_spos,_ = jax.lax.scan(calc_poses_fun_1,res_spos,jnp.arange(grid_size))

		
		#Collecting minima which are equivilant
		minima_types = jnp.zeros(grid_size+1,dtype = "int")
		dtol_a = jnp.pi/24.0
		dtol_z = 5
		dtol_p = 50
		ep = 1-1e-6
		def calc_no_poses_fun_1(minima_types,ind):
			def calc_no_poses_fun_2(ind_fix,ind):
				def is_same(ind_fix):
					ind_fix = ind_fix.at[1].set(0)
					ind_fix = ind_fix.at[2].set(ind+1)
					return ind_fix
				def is_not_same(ind_fix):
					return ind_fix
				distance = jnp.array([10.0,10.0,10.0])
				def go(distance):
					spos = sposs[ind]
					sposf = sposs[ind_fix[0]]
					distance = distance.at[0].set(jnp.arccos(ep*jnp.dot(res_spos[ind][0],res_spos[ind_fix[0]][0])))
					distance = distance.at[1].set(jnp.abs(spos[1]-sposf[1]))
					distance = distance.at[2].set(jnp.abs(results[ind][4]-results[ind_fix[0]][4]))
					return distance
				def stop(distance):
					return distance
				distance = jax.lax.cond(ind_fix[1] > 0.5,go,stop,distance)
				ind_fix = jax.lax.cond((distance[0] < dtol_a)*(distance[1] < dtol_z)*(distance[2] < dtol_p),is_same,is_not_same,ind_fix)
				return ind_fix,ind
			ind_fix,_ = jax.lax.scan(calc_no_poses_fun_2,jnp.array([ind,1,0],dtype = "int"),jnp.arange(grid_size))
			ind_type = ind_fix[2]
			def is_prev(ind_type):
				ind_type = minima_types[ind_type-1]
				return ind_type
			def is_not_prev(ind_type):
				return ind_type
			ind_type = jax.lax.cond(ind_type-1<ind,is_prev,is_not_prev,ind_type)
			def is_big(ind_type,minima_types):
				minima_types = minima_types.at[-1].set(minima_types[-1]+1)
				ind_type = minima_types[-1]
				return ind_type,minima_types
			def is_not_big(ind_type,minima_types):
				return ind_type,minima_types
			ind_type,minima_types = jax.lax.cond(ind_type > minima_types[-1],is_big,is_not_big,ind_type,minima_types)
			minima_types = minima_types.at[ind].set(ind_type)
			return minima_types,ind
			
		minima_types,_= jax.lax.scan(calc_no_poses_fun_1,minima_types,jnp.arange(grid_size))
		
		min_res_spos = jnp.zeros((minima_types[-1],8))
		def av_min_res_spos(min_res_spos,ind):
			spos = sposs[ind]
			correct_positon = position_point_jit(0,spos[2],spos[3],jnp.array([[0,0,1]]))
			correct_positon = position_point_jit(0,-spos[2],0,correct_positon)[0]
			min_res_spos = min_res_spos.at[minima_types[ind]-1,:3].set(min_res_spos[minima_types[ind]-1,:3]+correct_positon)
			min_res_spos = min_res_spos.at[minima_types[ind]-1,3].set(min_res_spos[minima_types[ind]-1,3]+spos[1])
			min_res_spos = min_res_spos.at[minima_types[ind]-1,4:].set(min_res_spos[minima_types[ind]-1,4:]+1)
			return min_res_spos,ind
		min_res_spos,_ = jax.lax.scan(av_min_res_spos,min_res_spos,np.arange(grid_size))
		min_res_spos = min_res_spos.at[:,:4].set(min_res_spos[:,:4]/min_res_spos[:,4:])
		
		
			
			

		minima = jnp.zeros(((minima_types[-1])*4,4))
		#Averaging collected minmia
		def av_minima_fun_1(minima,ind):
			spos = sposs[ind]
			minima = minima.at[minima_types[ind]-1].set(minima[minima_types[ind]-1]+spos[:4])
			minima = minima.at[minima_types[ind]-1+minima_types[-1]].set(minima[minima_types[ind]-1+minima_types[-1]]+1)
			minima = minima.at[minima_types[ind]-1+2*minima_types[-1]].set(minima[minima_types[ind]-1+2*minima_types[-1]]+results[ind][-1])
			minima = minima.at[minima_types[ind]-1+3*minima_types[-1]].set(minima[minima_types[ind]-1+3*minima_types[-1]]+spos[0]*jnp.cos(spos[3]))
			return minima, ind
		minima,_ = jax.lax.scan(av_minima_fun_1,minima,jnp.arange(minima_types.shape[0]-1))
		minima = minima.at[:minima_types[-1]].set(minima[:minima_types[-1]]/minima[minima_types[-1]:2*minima_types[-1]])
		minima = minima.at[2*minima_types[-1]:3*minima_types[-1]].set(minima[2*minima_types[-1]:3*minima_types[-1]]/minima[minima_types[-1]:2*minima_types[-1]])
		minima = minima.at[3*minima_types[-1]:].set(minima[3*minima_types[-1]:]/minima[minima_types[-1]:2*minima_types[-1]])
		minima_ind = jnp.argsort(minima[minima_types[-1]:2*minima_types[-1],0])[::-1]
		minima = minima.at[:minima_types[-1]].set(minima[minima_ind])
		minima = minima.at[minima_types[-1]:2*minima_types[-1]].set(minima[minima_ind+minima_types[-1]])
		minima = minima.at[2*minima_types[-1]:3*minima_types[-1]].set(minima[minima_ind+2*minima_types[-1]])
		minima = minima.at[3*minima_types[-1]:].set(minima[minima_ind+3*minima_types[-1]])
		
		devs = jnp.zeros(minima_types[-1])
		
		#Getting deviations from rank 1
		def get_min_devs(devs,ind):
			dev1 = jnp.arccos(jnp.dot(min_res_spos[minima_ind[ind+1],:3],min_res_spos[minima_ind[0],:3]))
			dev2 = jnp.abs(min_res_spos[minima_ind[ind+1],3]-min_res_spos[minima_ind[0],3])/100
			devs = devs.at[minima_ind[ind+1]].set(dev1+dev2)
			return devs,ind
		devs,_ = jax.lax.scan(get_min_devs,devs,jnp.arange(minima_types[-1]-1))
		
		#Getting color associated with the deviations
		def color_grid_fun_2(color_grid,ind):
			spos = sposs[ind]
			colr = 1.0
			colg = 1-devs[minima_types[ind]-1]/3
			colb = 1-devs[minima_types[ind]-1]/3

			def gone(col):
				col = 1.0
				return col
			def lone(col):
				return col
			
			def lzero(col):
				col=0.0
				return col
			def gzero(col):
				return col
				
			colr = jax.lax.cond(colr > 1, gone,lone,colr)
			colg = jax.lax.cond(colg > 1, gone,lone,colg)
			colb = jax.lax.cond(colb > 1, gone,lone,colb)
	 
			colr = jax.lax.cond(colr < 0, lzero,gzero,colr)
			colg = jax.lax.cond(colg < 0, lzero,gzero,colg)
			colb = jax.lax.cond(colb < 0, lzero,gzero,colb)
			color_grid = color_grid.at[ind].set(jnp.array([colr,colg,colb]))
			return color_grid,ind
			
		color_grid,_ = jax.lax.scan(color_grid_fun_2,color_grid,jnp.arange(color_grid.shape[0]))
		
		self.min_mem = jnp.zeros((minima_types[-1],4*self.numa+4*self.numb+2))
		self.min_mem = self.min_mem.at[:,-2].set(self.cut_a)
		self.min_mem = self.min_mem.at[:,-1].set(self.cut_b)
		self.minima = jnp.array(minima)
		self.minima_types = minima_types[:-1]
		self.minima_ind = minima_ind
		self.no_mins = minima_types[-1]
		self.new_memt_im = jnp.zeros(minima_types[-1])+self.memt_tails
		self.new_memt_om = jnp.zeros(minima_types[-1])+self.memt_tails_om
		return jnp.array(color_grid),pot_grid
		
	#This function optimises the membrane thickness (Should probably be JAXED)
	@partial(jax.jit,static_argnums=2)
	def optimise_mem_thickness(self,ind,fine,outer):
		mins = jnp.array(self.minima)
		no_mins = (mins.shape[0]//4)
		min_poses = mins[:no_mins]
		#print(min_poses)
		min_hits = mins[no_mins:no_mins*2,0]
		min_hits = 100*min_hits/(np.sum(min_hits))
		min_pots = mins[no_mins*2:no_mins*3,0]
		min_zdist = jnp.abs(mins[no_mins*3:,0])
		numa = self.numa
		numb = self.numb
		
		zdir = position_point_jit(0,min_poses[ind][2],min_poses[ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
		xydir =  position_point_jit(0,min_poses[ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
		position = jnp.concatenate([jnp.array([min_poses[ind][0],min_poses[ind][1]]),zdir,xydir])
		
		
		insd_grid = jnp.linspace(-5,5,fine)
		def is_outer3():
			return self.memt_tails_om
		def is_inner3():
			return self.memt_tails
		memt_start = jax.lax.cond(outer,is_outer3,is_inner3)
		memt_grid = jnp.linspace(memt_start-20,memt_start+20,fine)
		vals = jnp.zeros((fine,fine))
		def memt_fun1(vals,ind):
			ind_fix = ind
			def memt_fun2(vals,ind):
				
				def zero(change):
					return change,0.0
				def not_zero(change):
					def is_outer2(change):
						return -change/2
					def is_inner2(change):
						return change/2
					return change/2,jax.lax.cond(outer,is_outer2,is_inner2,change)
				chg1,chg2 = jax.lax.cond(position[0] < 1e-5,zero,not_zero,insd_grid[ind])
				position_test = position.at[1].set(position[1]+chg1)
				position_test = position_test.at[0].set(position[0]+chg2)
				
				heads = self.memt_heads
				h1_w = heads/6.0
				h2_w = heads/6.0
				h3_w = heads/6.0
				
				l_w = memt_grid[ind_fix]
				meml = -l_w/2.0 -h1_w -h2_w-h3_w
				def is_outer(mem_structure_im,mem_structure_om):
					return jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w]),mem_structure_im.copy()
				def is_inner(mem_structure_im,mem_structure_om):
					return mem_structure_om.copy(),jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
				mem_structure_om,mem_structure_im = jax.lax.cond(outer,is_outer,is_inner,self.mem_structure_im,self.mem_structure_om)
				#jax.debug.print("{x}",x=self.charge_mult)
				vals = vals.at[ind,ind_fix].set(self.calc_pot_jit(position_test,mem_structure_im,mem_structure_om))
				return vals, ind
			vals,_=jax.lax.scan(memt_fun2,vals,jnp.arange(fine))
			return vals,ind
		vals,_=jax.lax.scan(memt_fun1,vals,jnp.arange(fine))
		#plt.imshow(vals)
		#plt.show()
		#jax.debug.print("{x}",x=vals)
		best = jnp.unravel_index(jnp.argmin(vals),(fine,fine))
		best_pos = insd_grid[best[0]]+position[1]
		best_memt = memt_grid[best[1]]
		return best_pos,best_memt
		
	#This function evaluates the optimal membrane thickness for all minima
	def optimise_memt_all(self):
		#self.charge_mult = 0
		#self.charge_mult_om = 0
		mins = jnp.array(self.minima)
		no_mins = (mins.shape[0]//4)
		min_zdist = jnp.abs(mins[no_mins*3:,0])
		carry = jnp.zeros(5*no_mins)
		new_poses = jnp.zeros(no_mins)
		new_zdists = jnp.zeros(no_mins)
		min_poses = mins[:no_mins]
		
		def allmem_fun1(carry,ind):
			def zero(carry):
				bpos,bmem_im = self.optimise_mem_thickness(ind,10,False)
				carry = carry.at[ind].set(bpos)
				carry = carry.at[ind+no_mins*2].set(bmem_im)
				
				zdir = position_point_jit(0,min_poses[ind][2],min_poses[ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
				xydir =  position_point_jit(0,min_poses[ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
				position = jnp.concatenate([jnp.array([min_poses[ind][0],min_poses[ind][1]]),zdir,xydir])
				
				position_test = position.at[1].set(bpos)
				
				
				heads = self.memt_heads
				h1_w = heads/6.0
				h2_w = heads/6.0
				h3_w = heads/6.0
				
				l_w = bmem_im
				meml = -l_w/2.0 -h1_w -h2_w-h3_w
				mem_structure_im = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
					
				
				carry = carry.at[ind+no_mins*4].set(self.calc_pot_jit(position_test,mem_structure_im,self.mem_structure_om))
				return carry
			def not_zero(carry):
				bpos,bmem_im = self.optimise_mem_thickness(ind,10,False)
				bpos2,bmem_om = self.optimise_mem_thickness(ind,10,True)
				
				carry = carry.at[ind].set((bpos+bpos2)/2.0)
				carry = carry.at[ind+no_mins].set(min_zdist[ind]+(bpos-bpos2)/2.0)
				
				carry = carry.at[ind+no_mins*2].set(bmem_im)
				carry = carry.at[ind+no_mins*3].set(bmem_om)
				
				zdir = position_point_jit(0,min_poses[ind][2],min_poses[ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
				xydir =  position_point_jit(0,min_poses[ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
				position = jnp.concatenate([jnp.array([min_poses[ind][0],min_poses[ind][1]]),zdir,xydir])
				
				position_test = position.at[1].set((bpos+bpos2)/2.0)
				position_test = position_test.at[0].set(min_zdist[ind]+(bpos-bpos2)/2.0)
				heads = self.memt_heads
				h1_w = heads/6.0
				h2_w = heads/6.0
				h3_w = heads/6.0
				
				l_w = bmem_im
				meml = -l_w/2.0 -h1_w -h2_w-h3_w
				mem_structure_im = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
				
				l_w = bmem_om
				meml = -l_w/2.0 -h1_w -h2_w-h3_w
				mem_structure_om = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
				
				carry = carry.at[ind+no_mins*4].set(self.calc_pot_jit(position_test,mem_structure_im,mem_structure_om))
				
				return carry
			carry = jax.lax.cond(not self.dbmem,zero,not_zero,carry)
			return carry,ind
		carry,_ = jax.lax.scan(allmem_fun1,carry,jnp.arange(no_mins))
		self.new_memt_im = carry[no_mins*2:no_mins*3]
		self.new_memt_om = carry[no_mins*3:no_mins*4]
		new_poses = carry[:no_mins]
		new_zdists = carry[no_mins:no_mins*2]
		self.minima = self.minima.at[no_mins*3:,0].set(new_zdists)
		self.minima = self.minima.at[:no_mins,1].set(new_poses)
		#self.minima = self.minima.at[no_mins*2:no_mins*3,0].set(carry[no_mins*4:no_mins*5])
		#jax.debug.print("{x}",x=self.new_memt_om)
		
	#Builds a CG system using insane
	def build_oriented(self,orient_dir,build_args):
		if(self.curva):
			mins = np.array(self.minima_c)
		else:
			mins = np.array(self.minima)
		no_mins = (mins.shape[0]//4)
		min_zdist = mins[no_mins*3:,0]
		no_build = min(self.build_no,no_mins)
		for i in range(no_build):
			rank_dir = orient_dir+"Rank_"+str(i+1)+"/"
			zdist = min_zdist[i]/10
			print("Building rank "+str(i+1)+" with insane:")
			cg_sys_dir = rank_dir+"CG_System_rank_"+str(i+1)+"/"
			run_str = "python2.7 "+PATH_TO_INSANE+" "+build_args.strip()+" -o "+cg_sys_dir+"CG-system.gro -p "+cg_sys_dir+"topol.top -f "+cg_sys_dir+"protein-cg.pdb"
			if(zdist > 1e-6):
				run_str += " -ps "+str(zdist)
			err_val = os.system(run_str)
			if(err_val != 0):
				print("WARNING: There was an error when trying to build the system. Check -bd_args are correct.")
	#Writes a pdb using a given file as a template and replacing positions
	def write_oriented(self,temp_fn,orient_dir):
		lfile = open(temp_fn,"r")
		content = lfile.read()
		lfile.close()
		content = content.split("\n")
		
		if(self.curva):
			mins = np.array(self.minima_c)
		else:
			mins = np.array(self.minima)
		no_mins = (mins.shape[0]//4)
		min_poses = mins[:no_mins]
		#print(min_poses)
		min_hits = mins[no_mins:no_mins*2,0]
		min_hits = 100*min_hits/(np.sum(min_hits))
		min_pots = mins[no_mins*2:no_mins*3,0]
		min_zdist = np.abs(mins[no_mins*3:,0])
		numa = self.numa
		numb = self.numb
		
		#start_zdir = position_point_jit(0,min_poses[0][2],min_poses[0][3],jnp.array([[0.0,0.0,1.0]]))[0]
		#start_xydir =  position_point_jit(0,min_poses[0][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
		#self.z_force_plot(200,30,min_poses[0][1],start_zdir,start_xydir)
		
		
		ranks = open(os.path.join(orient_dir,"orientation.txt"),"w")
		ranks.write("Rank \t Relative potential \t % hits \t force max + \t force max -\n")
		for i in range(min_poses.shape[0]):
			if(not os.path.exists(orient_dir+"Rank_"+str(i+1)+"/")):
				os.mkdir(orient_dir+"Rank_"+str(i+1)+"/")
			rank_dir = orient_dir+"Rank_"+str(i+1)+"/"
			mem_info = self.min_mem[i]
			lss_a = mem_info[:numa]
			lss_b = mem_info[numa:numb+numa]
			centers_a = mem_info[numa+numb:3*numa+numb].reshape((numa,2))
			centers_b = mem_info[3*numa+numb:3*numa+3*numb].reshape((numb,2))
			heights_a = mem_info[3*numa+3*numb:4*numa+3*numb]
			heights_b = mem_info[4*numa+3*numb:4*numa+4*numb]
			cuta = int(mem_info[-2])
			cutb = int(mem_info[-1])
			infos = open(os.path.join(rank_dir,"info_rank_"+str(i+1)+".txt"),"w")
			infos.write("Iter-Membrane distance (for -dm only): "+str(form(min_zdist[i]))+"\n")
			infos.write("Potential Energy (rel): "+str(form(min_pots[i]))+"\n")
			infos.write("(Inner) Membrane Thickness: "+str(form(self.new_memt_im[i]))+"\n")
			infos.write("Outer Membrane Thickness (for -dm only): "+str(form(self.new_memt_om[i]))+"\n")
			
			start_zdir = position_point_jit(0,min_poses[i][2],min_poses[i][3],jnp.array([[0.0,0.0,1.0]]))[0]
			start_xydir =  position_point_jit(0,min_poses[i][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
			
			if self.force_calc > i:
				if self.curva:
					pm,mm = self.z_force_plot_c(50,30,min_poses[i][1],start_zdir,start_xydir,centers_a,centers_b,heights_a,heights_b)
				else:
					pm,mm = self.z_force_plot(50,30,min_poses[i][1],start_zdir,start_xydir)
			else:
				pm = "Not calculated (use -fc n)"
				mm = "Not calculated (use -fc n)"
			
			orient_poses = np.array(position_point_jit(min_poses[i][1],min_poses[i][2],min_poses[i][3],self.all_poses))
			ranks.write(str(i+1)+"\t "+str(form(min_pots[i]))+"\t"+str(form(min_hits[i]))+"\t"+str(pm)+"\t"+str(mm)+"\n")
			new_file = open(os.path.join(rank_dir,"oriented_rank_"+str(i+1)+".pdb"),"w")
			if(self.build_no > i):
				cg_sys_dir = rank_dir+"CG_System_rank_"+str(i+1)
				if(not os.path.exists(cg_sys_dir)):
					os.mkdir(cg_sys_dir)
				new_file_b = open(os.path.join(cg_sys_dir,"protein-cg.pdb"),"w")
			count = 0
			count2 = 0
			neww = False
			prev_num = -1
			strr = ""
			strc = ""
			for c in content:
				if(len(c) > 46):
					if("[" not in c and c[:4]=="ATOM"):
						zpos = c[46:54]
						ypos = c[38:46]
						xpos = c[30:38]
						res = c[17:20].strip()
						atom_num = int(c[22:26].strip())
						b_val = float(c[60:66].strip())
						pos = np.array([float(xpos.strip()),float(ypos.strip()),float(zpos.strip())])
						if(not np.any(np.isnan(pos))):
							xp = np.format_float_positional(orient_poses[count][0],precision=3)
							yp = np.format_float_positional(orient_poses[count][1],precision=3)
							zp = np.format_float_positional(orient_poses[count][2],precision=3)
							xp += "0"*(3-len((xp.split(".")[1])))
							yp += "0"*(3-len((yp.split(".")[1])))
							zp += "0"*(3-len((zp.split(".")[1])))
							if(atom_num != prev_num):
								if(neww):
									strr += three2one(res)
									strc += "T"
								else:
									strr += three2one(res)
									strc += "N"
								neww = False
							bbp = np.format_float_positional(b_val,precision=3)
							if(self.new_memt_im[i]/2 > orient_poses[count][2] >-self.new_memt_im[i]/2):
								#bbp = np.format_float_positional(100,precision=3)
								neww = True
							else:
								pass
								#bbp = np.format_float_positional(0,precision=3)
							bbp += "0"*(3-len((bbp.split(".")[1])))
							new_c = c[:30]+(" "*(8-len(xp)))+xp+(" "*(8-len(yp)))+yp+(" "*(8-len(zp))) +zp+c[54:60]+(" "*(8-len(bbp)))+bbp+c[66:]+"\n"						
							new_file.write(new_c)
							if(self.build_no > i):
								new_file_b.write(new_c)
							count += 1
						prev_num = atom_num
			infos.write("Transmembrane residues (T):\n")
			infos.write(strr+"\n")
			infos.write(strc+"\n")
			grid_num = 50
			xs = jnp.linspace(-100,100,grid_num)
			ys = jnp.linspace(-100,100,grid_num)
			grid = []
			grid2 = []
			grid_norm = []
			if(abs(min_zdist[i]) > 1e-5):
				grid_hu,grid_h2u,grid_normsu = make_membrane_jit(lss_a[:cuta],lss_b[:cutb],centers_a[:cuta],centers_b[:cutb],heights_a[:cuta],heights_b[:cutb],grid_num,xs,ys)
				grid_hl,grid_h2l,grid_normsl = make_membrane_jit(lss_a[cuta:],lss_b[cutb:],centers_a[cuta:],centers_b[cutb:],heights_a[cuta:],heights_b[cutb:],grid_num,xs,ys)
				grid = [np.array(grid_hu),np.array(grid_hl)]
				grid2= [np.array(grid_h2u),np.array(grid_h2l)]
				grid_norm = [np.array(grid_normsu),np.array(grid_normsl)]
			else:
				grid_h,grid_h2,grid_norms = make_membrane_jit(lss_a,lss_b,centers_a,centers_b,heights_a,heights_b,grid_num,xs,ys)
				grid = [np.array(grid_h)]
				grid2= [np.array(grid_h2)]
				grid_norm = [np.array(grid_norms)]
			count = 0
			xs = np.array(xs)
			ys = np.array(ys)
			for zind in range(2):
				zz = (2*zind)-1
				mem_totals = [self.memt_heads_om+self.new_memt_om[i],self.memt_heads+self.new_memt_im[i]]
				mem_tails = [self.new_memt_om[i],self.new_memt_im[i]]
				if(zz == 1 or abs(min_zdist[i]) > 1e-5):
					for xi in range(grid_num):
						for xj in range(grid_num):
							zs = [mem_totals[zind]/2.0+grid2[1-zind][xi][xj]/2+zz*min_zdist[i],mem_tails[zind]/2.0+grid2[1-zind][xi][xj]/2+zz*min_zdist[i],-mem_tails[zind]/2.0-grid2[1-zind][xi][xj]/2+zz*min_zdist[i],-mem_totals[zind]/2.0-grid2[1-zind][xi][xj]/2+zz*min_zdist[i]]
							adder = grid[1-zind][xi][xj]
							norma = grid_norm[1-zind][xi][xj]
							for xk in zs:
								count += 1
								count_str = (6-len(str(count)))*" "+str(count)
								c = "ATOM "+count_str+" BB   DUM     1       0.000   0.000  15.000  1.00  0.00" 
								xp = np.format_float_positional(xs[xi]+xk*norma[0],precision=3)
								yp = np.format_float_positional(ys[xj]+xk*norma[1],precision=3)
								zp = np.format_float_positional(adder+xk*norma[2],precision=3)
								xp += "0"*(3-len((xp.split(".")[1])))
								yp += "0"*(3-len((yp.split(".")[1])))
								zp += "0"*(3-len((zp.split(".")[1])))
								new_c = c[:30]+(" "*(8-len(xp)))+xp+(" "*(8-len(yp)))+yp+(" "*(8-len(zp))) +zp+c[54:]+"\n"	
								new_file.write(new_c)
			
			if(abs(min_zdist[i])>1e-5 and self.pg_layer_pred): 
				heads = self.memt_heads
				h1_w = heads/6.0
				h2_w = heads/6.0
				h3_w = heads/6.0
				
				
				l_w = self.new_memt_im[i]
				meml = -l_w/2.0 -h1_w -h2_w-h3_w
				mem_structure_im = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
				
				l_w = self.new_memt_om[i]
				meml = -l_w/2.0 -h1_w -h2_w-h3_w
				mem_structure_om = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
				
				pg_pos = self.get_pg_pos(400,i,mem_structure_im,mem_structure_om,rank_dir)
				#self.pg_pot_plot(400,i,mem_structure_im,mem_structure_om)
				#self.pg_pot_plot(400,i)
				for xi in range(grid_num):
					for xj in range(grid_num):
						count += 1
						count_str = (6-len(str(count)))*" "+str(count)
						c = "ATOM "+count_str+" BB   DUM     1       0.000   0.000  15.000  1.00  0.00" 
						xp = np.format_float_positional(xs[xi],precision=3)
						yp = np.format_float_positional(ys[xj],precision=3)
						zpa = np.format_float_positional(-pg_pos+self.pg_thickness/2,precision=3)
						zpb = np.format_float_positional(-pg_pos-self.pg_thickness/2,precision=3)
						xp += "0"*(3-len((xp.split(".")[1])))
						yp += "0"*(3-len((yp.split(".")[1])))
						zpa += "0"*(3-len((zpa.split(".")[1])))
						zpb += "0"*(3-len((zpb.split(".")[1])))
						new_c = c[:30]+(" "*(8-len(xp)))+xp+(" "*(8-len(yp)))+yp+(" "*(8-len(zpa))) +zpa+c[54:]+"\n"	
						new_file.write(new_c)
						new_c = c[:30]+(" "*(8-len(xp)))+xp+(" "*(8-len(yp)))+yp+(" "*(8-len(zpb))) +zpb+c[54:]+"\n"	
						new_file.write(new_c)
			new_file.close()
			infos.close()
		ranks.close()

	#We define some new potential functions which take into account curvature
	def mem_pot(self,point,grad,bead_num,lss_a,lss_b,centers_a,centers_b,heights_a,heights_b):
		dist,min_point = mem_pos(point,lss_a,centers_a,heights_a)
		h1_w = self.memt_heads/6.0
		h2_w = self.memt_heads/6.0
		h3_w = self.memt_heads/6.0
		l_w = self.memt_tails+gaussian_sum(lss_b,heights_b,centers_b,min_point[0],min_point[1])
		meml = -l_w/2.0 -h1_w -h2_w-h3_w
		mem_structure = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
		pot_at_point = self.smj(jnp.abs(dist),grad,bead_num,mem_structure)#+self.smcj(dist,grad,bead_num)
		return pot_at_point

	def in_mem(self,point,lss_a,lss_b,centers_a,centers_b,heights_a,heights_b):
		dist,min_point = mem_pos(point,lss_a,centers_a,heights_a)
		h1_w = self.memt_heads/6.0
		h2_w = self.memt_heads/6.0
		h3_w = self.memt_heads/6.0
		l_w = self.memt_tails+gaussian_sum(lss_b,heights_b,centers_b,min_point[0],min_point[1])
		meml = -l_w/2.0 -h1_w -h2_w-h3_w
		mem_structure = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
		retval = self.sibj(dist,2,mem_structure)
		return retval
		
	@jax.jit
	def in_memd_jit(self,point,zdist,centers_a,centers_b,heights_a,heights_b):
		retval = False
		def zero(retval):
			retval = self.in_mem(point,self.lss_a,self.lss_b,centers_a,centers_b,heights_a,heights_b)
			return retval
		def not_zero(retval):
			point_l = point.at[2].set(point[2]-zdist)
			point_u = point.at[2].set(point[2]+zdist)
			retval = self.in_mem(point_l,self.lss_a[:self.cut_a],self.lss_b[:self.cut_b],
				centers_a[:self.cut_a],centers_b[:self.cut_b],heights_a[:self.cut_a],
				heights_b[:self.cut_b])+self.in_mem(point_u,self.lss_a[self.cut_a:],
				self.lss_b[self.cut_b:],centers_a[self.cut_a:],centers_b[self.cut_b:],
				heights_a[self.cut_a:],heights_b[self.cut_b:])
			return retval
		retval = jax.lax.cond(not self.dbmem,zero,not_zero,retval)
		return retval
	
	@jax.jit
	def dmpcj(self,point,grad,zdist,bead_num,centers_a,centers_b,heights_a,heights_b):
		retval = 0.0
		def zero(retval):
			retval = self.mem_pot(point,grad,bead_num,self.lss_a,self.lss_b,centers_a,centers_b,heights_a,heights_b)
			return retval
		def not_zero(retval):
			point_l = point.at[2].set(point[2]-zdist)
			point_u = point.at[2].set(point[2]+zdist)
			retval = self.mem_pot(point_l,grad,bead_num,self.lss_a[:self.cut_a],
				self.lss_b[:self.cut_b],centers_a[:self.cut_a],centers_b[:self.cut_b],
				heights_a[:self.cut_a],heights_b[:self.cut_b])+self.mem_pot(point_u,
				grad,bead_num,self.lss_a[self.cut_a:],self.lss_b[self.cut_b:],
				centers_a[self.cut_a:],centers_b[self.cut_b:],heights_a[self.cut_a:],
				heights_b[self.cut_b:])
			return retval
		retval = jax.lax.cond(not self.dbmem,zero,not_zero,retval)
		return retval

	#Calculates potential of a single bead in a curved membrane
	@jax.checkpoint
	def calc_pot_at_p_check(self,p,zdist,ball,bead_type,centers_a,centers_b,heights_a,heights_b):
		smoothness = 2
		tot_pot_count = jnp.zeros(2)
		def calc_pot_fun_1(tot_pot_count,ind):
			def pot_cond_0(tot_pot_count):
				return tot_pot_count
			def not_pot_cond_0(tot_pot_count):
				pos = ball[ind]+p
				tot_pot_count = tot_pot_count.at[1].set(tot_pot_count[1]+1)
				tot_pot_count = tot_pot_count.at[0].set(tot_pot_count[0]+self.dmpcj(pos,smoothness,zdist,bead_type,centers_a,centers_b,heights_a,heights_b))
				return tot_pot_count
			tot_pot_count = jax.lax.cond(jnp.linalg.norm(ball[ind]) < 1e-5,pot_cond_0,not_pot_cond_0,tot_pot_count)
			return tot_pot_count,ind
		tot_pot_count,_ = jax.lax.scan(calc_pot_fun_1,tot_pot_count,jnp.arange(ball.shape[0]))
		tot_pot = tot_pot_count[0]/(tot_pot_count[1]+1e-5)
		return tot_pot	
	
	#Counter part of the new charge function above
	@jax.jit
	def new_chargep1_c(self,p,zdist,bead_type,lss_a,lss_b,centers_a,centers_b,heights_a,heights_b):
		charge_const = -(1.6*1.6*10000)/(4*jnp.pi*15*8.86)
		charge_const = charge_const*(self.charge_mult)*self.Charge_B_mins[bead_type]
		tot_charge = 0
		ppos = p+jnp.array([0.0,0.0,zdist])
		zpos,min_point = mem_pos(ppos,lss_a,centers_a,heights_a)
		h1_w = self.memt_heads/6.0
		h2_w = self.memt_heads/6.0
		h3_w = self.memt_heads/6.0
		l_w = self.memt_tails+gaussian_sum(lss_b,heights_b,centers_b,min_point[0],min_point[1])
		meml = -l_w/2.0 -h1_w -h2_w-h3_w
		mem_structure = jnp.array([meml,meml+h1_w,meml+h2_w+h1_w,meml+h2_w+h1_w+h3_w,meml+h2_w+h1_w+h3_w+l_w,meml+h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+h1_w+2*h3_w+l_w,meml+2*h2_w+2*h1_w+2*h3_w+l_w])
		grid_ex = 10
		grid_nums = 10
		dx = (2*(grid_ex)/grid_nums)*(2*(grid_ex)/grid_nums)
		grid = jnp.linspace(-grid_ex,grid_ex,grid_nums)
		def in_charge(x,msa,msb):
			return x
		def not_in_charge(x,msa,msb):
			def above(x,msa,msb):
				retval = msa
				return retval
			def below(x,msa,msb):
				retval = msb
				return retval
			retval = jax.lax.cond(x>msa,above,below,x,msa,msb)
			return retval
		zposa = jax.lax.cond((zpos<mem_structure[3])*(zpos>mem_structure[0]),in_charge,not_in_charge,zpos,mem_structure[3],mem_structure[0])
		zposb = jax.lax.cond((zpos<mem_structure[7])*(zpos>mem_structure[4]),in_charge,not_in_charge,zpos,mem_structure[7],mem_structure[4])
		charge_val = 0
		grid_h,grid_h2,grid_norms = make_membrane(lss_a,lss_b,centers_a,centers_b,heights_a,heights_b,grid_nums,grid+p[0],grid+p[1])
		def calc_charge_fun(tot_charge,ind):
			def calc_charge_fun2(tot_charge,ind):
				sgn =tot_charge[1]/jnp.abs(tot_charge[1])
				zz = tot_charge[1]+(grid_h2[ind][ind_fix]-l_w+self.memt_tails)
				#jax.debug.print("{x}",x=(grid_h2[ind][ind_fix]-l_w+30))
				adder = grid_h[ind][ind_fix]
				norma = grid_norms[ind][ind_fix]
				point_a = ppos
				#jax.debug.print("{x}",x=norma)
				point_b = jnp.array([p[0]+grid[ind]+zz*norma[0],p[1]+grid[ind_fix]+zz*norma[1],adder+1e-5+zz*norma[2]])
				#jax.debug.print("{x}",x=tot_charge[1])
				dist = jnp.linalg.norm(point_a-point_b)
				#jax.debug.print("{x}",x=dist)
				def not_cutoff(tot_charge):
					def lfive(tot_charge):
						tot_charge = tot_charge.at[0].set(tot_charge[0]+dx*(charge_const/5.0-charge_const/10.0))
						return tot_charge
					def gfive(tot_charge):
						tot_charge = tot_charge.at[0].set(tot_charge[0]+dx*(charge_const/(jnp.abs(dist))-charge_const/10.0))
						return tot_charge
					tot_charge = jax.lax.cond(dist < 5, lfive,gfive,tot_charge)
					#tot_charge += charge_const/(jnp.abs(dist)+1e-5)
					return tot_charge
				def cutoff(tot_charge):
					return tot_charge
				tot_charge = jax.lax.cond(dist > 10.0,cutoff,not_cutoff,tot_charge)
				#tot_charge += charge_const/(jnp.abs(dist)+1e-5)
				return tot_charge,ind
			ind_fix=ind
			tot_charge,_=jax.lax.scan(calc_charge_fun2,tot_charge,jnp.arange(grid_nums))
			return tot_charge,ind
		retval,_=jax.lax.scan(calc_charge_fun,jnp.array([0,zposa]),jnp.arange(grid_nums))
		retvalb,_=jax.lax.scan(calc_charge_fun,jnp.array([0,zposb]),jnp.arange(grid_nums))
		charge_val = retval[0]*1.01+retvalb[0]
		return charge_val
		
	@jax.jit
	def new_charge_c(self,p,zdist,bead_type,centers_a,centers_b,heights_a,heights_b):
		def zero():
			return self.new_chargep1_c(p,zdist,bead_type,self.lss_a,self.lss_b,centers_a,centers_b,heights_a,heights_b)
		def not_zero():
			return self.new_chargep1_c(p,-zdist,bead_type,self.lss_a[:self.cut_a],self.lss_b[:self.cut_b],centers_a[:self.cut_a],centers_b[:self.cut_b],heights_a[:self.cut_a],heights_b[:self.cut_b])+self.new_chargep1_c(p,zdist,bead_type,self.lss_a[self.cut_a:],self.lss_b[self.cut_b:],centers_a[self.cut_a:],centers_b[self.cut_b:],heights_a[self.cut_a:],heights_b[self.cut_b:])
		retval = jax.lax.cond(not self.dbmem,zero,not_zero)
		return retval
	#Calculates lipid displacemnt in a curved membrane
	def get_lipid_disp_c(self,filled_poses,disp_penalty,zdist,centers_a,centers_b,heights_a,heights_b):
		tot_pot_disp = jnp.zeros(2)
		def get_lipd_fun_1(tot_pot_disp,ind):
			def in_bilayer(tot_pot_disp,memval):
				tot_pot_disp = tot_pot_disp.at[0].set(tot_pot_disp[0]+disp_penalty*self.b_vals[ind]*memval)
				tot_pot_disp = tot_pot_disp.at[1].set(tot_pot_disp[1]+1)
				return tot_pot_disp
			def not_in_bilayer(tot_pot_disp,memval):
				return tot_pot_disp
			memval = self.in_memd_jit(filled_poses[ind],zdist,centers_a,centers_b,heights_a,heights_b)
			tot_pot_disp = jax.lax.cond(memval>0,in_bilayer,not_in_bilayer,tot_pot_disp,memval)
			return tot_pot_disp,ind
		tot_pot_disp,_ = jax.lax.scan(get_lipd_fun_1,tot_pot_disp,jnp.arange(filled_poses.shape[0]))
		return tot_pot_disp
				
	#Due to the large amount of data involved we have to use checkpoint (This forces recalculations rather than storing all data)		
	def get_lipid_disp_c_check(self,filled_poses,disp_penalty,zdist,centers_a,centers_b,heights_a,heights_b):
		return jax.checkpoint(self.get_lipid_disp_c)(filled_poses,disp_penalty,zdist,centers_a,centers_b,heights_a,heights_b)

	#A function for calculating the potential of a protein in a position with a curved membrane
	def calc_pot_c(self,position,centers_a,centers_b,heights_a,heights_b):
		disp_penalty = 0.025#0.025
		#thin_penalty = 0.005#0.01?
		def is_memopt():
			return 1e-6
		def is_not_memopt():
			return 0.005
		thin_penalty = jax.lax.cond(self.memopt,is_memopt,is_not_memopt)
		curv_penalty = 500.0#?500
		zdist_temp = position[0]
		in_depth = position[1]
		ang1 = position[2:5]
		ang2 = position[5:7]
		ang1 /= jnp.linalg.norm(ang1)
		zdist = jnp.abs(zdist_temp*jnp.dot(ang1,jnp.array([0.0,0.0,1.0])))
		tester_poses = position_pointv2_jit(in_depth,ang1,ang2,self.surface_poses)
		tester_fposes = position_pointv2_jit(in_depth,ang1,ang2,self.poses)
		tester_centers_a = centers_a.copy()
		tester_centers_b = centers_b.copy()
		def rot_spheres(carry,ind):
			new_spheres = position_pointv2_jit(0,ang1,ang2,self.spheres[ind])
			return carry,new_spheres
		_,test_spheres = jax.lax.scan(rot_spheres,0,jnp.arange(self.spheres.shape[0]))
		tot_pot = 0
		def calc_pot_fun_1(tot_pot,ind):
			tot_pot += self.surf_b_vals[ind]*self.calc_pot_at_p_check(tester_poses[ind],zdist,test_spheres[ind],self.bead_types[ind],tester_centers_a,tester_centers_b,heights_a,heights_b)
			tot_pot += self.surf_b_vals[ind]*self.new_charge_c(tester_poses[ind],zdist,self.bead_types[ind],tester_centers_a,tester_centers_b,heights_a,heights_b)
			
			return tot_pot,ind
		tot_pot,_ = jax.lax.scan(calc_pot_fun_1,tot_pot,jnp.arange(tester_poses.shape[0]))
		ldisp_p = self.get_lipid_disp_c_check(tester_fposes,disp_penalty,zdist,tester_centers_a,tester_centers_b,heights_a,heights_b)
		tot_pot += ldisp_p[0]
		def zero(tot_pot):
			tot_pot += gauss_sum_cost_jit(self.lss_b,heights_b,tester_centers_b)*thin_penalty
			tot_pot += cc_jit(self.lss_a,heights_a,tester_centers_a)*curv_penalty
			return tot_pot
		def not_zero(tot_pot):
			tot_pot += (gauss_sum_cost_jit(self.lss_b[self.cut_b:],heights_b[self.cut_b:],tester_centers_b[self.cut_b:])+gauss_sum_cost_jit(self.lss_b[:self.cut_b],heights_b[:self.cut_b],tester_centers_b[:self.cut_b]))*thin_penalty
			tot_pot += (cc_jit(self.lss_a[self.cut_a:],heights_a[self.cut_a:],tester_centers_a[self.cut_a:])+cc_jit(self.lss_a[:self.cut_a],heights_a[:self.cut_a],tester_centers_a[:self.cut_a]))*curv_penalty
			return tot_pot
		tot_pot = jax.lax.cond(not self.dbmem,zero,not_zero,tot_pot)
		return tot_pot
		
	#Jitting and checkpointing
	def calc_pot_c_jit(self,position,centers_a,centers_b,heights_a,heights_b):
		return jax.jit(self.calc_pot_c)(position,centers_a,centers_b,heights_a,heights_b)
	def calc_pot_c_check(self,position,centers_a,centers_b,heights_a,heights_b):
		return jax.checkpoint(self.calc_pot_c)(position,centers_a,centers_b,heights_a,heights_b)
	

	#A function that calculates if a protein has been fully ejected from a curved membrane
	@jax.jit
	def calc_in_water_c(self,position,centers_a,centers_b,heights_a,heights_b,tol):
		zdist_temp = position[0]
		in_depth = position[1]
		ang1 = position[2:5]
		ang2 = position[5:7]
		ang1 /= jnp.linalg.norm(ang1)
		zdist = jnp.abs(zdist_temp*jnp.dot(ang1,jnp.array([0.0,0.0,1.0])))
		tester_poses = position_pointv2_jit(in_depth,ang1,ang2,self.surface_poses)
		def rot_spheres(carry,ind):
			new_spheres = position_pointv2_jit(0,ang1,ang2,self.spheres[ind])
			return carry,new_spheres
		_,test_spheres = jax.lax.scan(rot_spheres,0,jnp.arange(self.spheres.shape[0]))
		tot_pot = 0
		def calc_pot_fun_1(tot_pot,ind):
			tot_pot += jnp.abs(self.surf_b_vals[ind]*self.calc_pot_at_p_check(tester_poses[ind],zdist,test_spheres[ind],self.bead_types[ind],centers_a,centers_b,heights_a,heights_b))
			tot_pot += jnp.abs(self.surf_b_vals[ind]*self.new_charge_c(tester_poses[ind],zdist,self.bead_types[ind],centers_a,centers_b,heights_a,heights_b))
			
			return tot_pot,ind
		tot_pot,_ = jax.lax.scan(calc_pot_fun_1,tot_pot,jnp.arange(tester_poses.shape[0]))
		
		return tot_pot<tol
	



	#Differentiating the potential to get gradients for minimisation
	@jax.jit
	def pot_c_grad_mem(self,position,centers_a,centers_b,heights_a,heights_b):
		return jax.grad(self.calc_pot_c_check,argnums=(0,1,2,3,4))(position,centers_a,centers_b,heights_a,heights_b)			
	
	#This almost certainly doesnt work anymore
	@partial(jax.jit,static_argnums = 1)
	def z_force_plot_c(self,max_iter,ranger,cen,zdir,xydir,centers_a,centers_b,heights_a,heights_b):
		zs = jnp.linspace(-ranger,ranger,max_iter)#20
		pots = jnp.zeros(max_iter)
		def calc_pot_fun(pots,ind):
			posi = jnp.zeros(7)
			posi = posi.at[1].set(zs[ind]+cen)
			posi = posi.at[2:5].set(zdir)
			posi = posi.at[5:7].set(xydir)
			pots = pots.at[ind].set(-self.pot_c_grad_mem(posi,centers_a,centers_b,heights_a,heights_b)[0][1])
			return pots,ind
		pots,_ = jax.lax.scan(calc_pot_fun,pots,jnp.arange(max_iter))
		plus_max = jnp.min(pots[max_iter//2:])
		minus_max = jnp.max(pots[:max_iter//2])
		return plus_max, minus_max
	
	
	#A function that minimises potential of a positions and curvature definition		
	def minimise_c(self,starting_pos,max_iter):	
		num = 3*self.numa+3*self.numb+7	
		numga = self.numa
		numgb = self.numb
		pot = 0
		data = jnp.zeros(num*4)
		gamma = jnp.array([self.gamma_val]*num)
		
		ep = 1e-8
		tol = 1e-10
		data = data.at[:num].set(starting_pos[:num])
		decend = 0.0
		def zero(decend):
			return decend
		def not_zero(decend):
			decend = 1.0
			return decend
		decend = jax.lax.cond(not self.dbmem,zero,not_zero,decend)
		decend_in = jnp.zeros(num)
		decend_in = decend_in.at[:].set(jnp.concatenate((jnp.array([0]),jnp.array([1.0]*6),jnp.array([1.0]*3*(numga+numgb)))))
		def is_memopt(decend_in):
			decend_in = decend_in.at[7+2*(numga+numgb):7+2*(numga+numgb)+numga].set(0.0)
			return decend_in
		def is_not_memopt(decend_in):
			return decend_in
		decend_in = jax.lax.cond(self.memopt,is_memopt,is_not_memopt,decend_in)
		data = data.at[num*2:num*3].set(self.lr_heights/(numga))#0.25
		data = data.at[num*2:num*2+7].set(self.lr_pos)#????
		data = data.at[num*2+7:num*2+2*(numga+numgb)+7].set(self.lr_cens)#0.05
		decend_in = decend_in.at[0].set(decend)
		def min_fun_1(data,ind):
			cena = data[7:7+2*numga].reshape((numga,2))
			cenb = data[7+2*numga:7+2*numga+2*numgb].reshape((numgb,2))
			grads = self.pot_c_grad_mem(data[:7],cena,cenb,data[7+2*numga+2*numgb:7+3*numga+2*numgb],data[7+3*numga+2*numgb:7+3*(numga+numgb)])	
			pos_g = grads[0]
			cena_g = grads[1].ravel()
			cenb_g = grads[2].ravel()
			hag = grads[3]
			hbg = grads[4]
			grad = jnp.concatenate((pos_g,cena_g,cenb_g,hag,hbg))
			data = data.at[num:num*2].set(data[num:num*2]*gamma + (1-gamma)*grad*grad)
			rms_grad = jnp.sqrt(ep + data[num:num*2])
			rms_change = jnp.sqrt(ep+data[num*2:num*3])
			change = -(rms_change/rms_grad)*grad
			change = change*decend_in
			data = data.at[:num].set(data[:num]+change)
			data = data.at[num*2:num*3].set(gamma*data[num*2:num*3]+(1-gamma)*change*change)
			return data,ind
		final_out = jnp.zeros(8+3*(numga+numgb))
		final_data,_ = jax.lax.scan(min_fun_1,data,None,length = max_iter)
		cena = final_data[7:7+2*numga].reshape((numga,2))
		cenb = final_data[7+2*numga:7+2*numga+2*numgb].reshape((numgb,2))
		
		final_pot = self.calc_pot_c_jit(final_data[:7],cena,cenb,final_data[7+2*numga+2*numgb:7+3*numga+2*numgb],final_data[7+3*numga+2*numgb:7+3*(numga+numgb)])
		flipped_data = final_data.copy()
		### As in the previous minimise
		normed = jnp.array([flipped_data[5],flipped_data[6],0.0])
		normed2 = jnp.array([flipped_data[2],flipped_data[3],flipped_data[4]])
		
		normed2 /= jnp.linalg.norm(normed2)
		
		direc1 = jnp.cross(normed2,jnp.array([0.0,0.0,1.0]))
		ang1 = (1-1e-8)*jnp.dot(normed2,jnp.array([0.0,0.0,1.0]))
		
		rot1qi = jnp.array([ang1+1,direc1[0],direc1[1],direc1[2]])
		rot1qi /= jnp.linalg.norm(rot1qi)
		rot1q = qcong_jit(rot1qi)

		
		xydir_cor = jnp.array([flipped_data[5],flipped_data[6],0.0])
		xydir_cor /= jnp.linalg.norm(xydir_cor)
		
		rotated_zdir = position_point_jit(0.0,0.0,jnp.pi,jnp.array([normed2]))[0]
		xycqp = jnp.array([0.0,xydir_cor[0],xydir_cor[1],xydir_cor[2]])
		xycrot_qp = qmul_jit(rot1q,qmul_jit(xycqp,rot1qi))
		xycrot_p = xycrot_qp[1:]
		
		rotated_xydir = position_point_jit(0.0,0.0,jnp.pi,jnp.array([xycrot_p]))[0]
		
		direc1 = jnp.cross(rotated_zdir,jnp.array([0.0,0.0,1.0]))
		ang1 = (1-1e-8)*jnp.dot(rotated_zdir,jnp.array([0.0,0.0,1.0]))
		
		rot1q = jnp.array([ang1+1,direc1[0],direc1[1],direc1[2]])
		rot1q /= jnp.linalg.norm(rot1q)
		rot1qi = qcong_jit(rot1q)
		
		xycqp = jnp.array([0.0,rotated_xydir[0],rotated_xydir[1],rotated_xydir[2]])
		xycrot_qp = qmul_jit(rot1q,qmul_jit(xycqp,rot1qi))
		xycrot_p = xycrot_qp[1:]

		flipped_data = flipped_data.at[2:5].set(rotated_zdir)
		flipped_data = flipped_data.at[5:7].set(xycrot_p[:2])
		
		###
		
		to_flip_cena = jnp.zeros((numga,3))
		to_flip_cena = to_flip_cena.at[:,:2].set(cena)
		to_flip_cenb = jnp.zeros((numgb,3))
		to_flip_cenb = to_flip_cenb.at[:,:2].set(cenb)
		
		flipped_cena_n = position_point_jit(0,0,jnp.pi,to_flip_cena)[:,:2]
		flipped_cenb_n = position_point_jit(0,0,jnp.pi,to_flip_cenb)[:,:2]
		
		flipped_cena = jnp.zeros_like(flipped_cena_n)
		flipped_cena = flipped_cena.at[:self.cut_a].set(flipped_cena_n[self.cut_a:])
		flipped_cena = flipped_cena.at[self.cut_a:].set(flipped_cena_n[:self.cut_a])
		
		flipped_cenb = jnp.zeros_like(flipped_cenb_n)
		flipped_cenb = flipped_cenb.at[:self.cut_b].set(flipped_cenb_n[self.cut_b:])
		flipped_cenb = flipped_cenb.at[self.cut_b:].set(flipped_cenb_n[:self.cut_b])
		
		ha = -flipped_data[7+2*numga+2*numgb:7+3*numga+2*numgb]
		hb = flipped_data[7+3*numga+2*numgb:7+3*numga+3*numgb]
		
		flipped_ha = jnp.zeros_like(ha)
		flipped_hb = jnp.zeros_like(hb)
		
		flipped_ha = flipped_ha.at[:self.cut_a].set(ha[self.cut_a:])
		flipped_ha = flipped_ha.at[self.cut_a:].set(ha[:self.cut_a])
		
		flipped_hb = flipped_hb.at[:self.cut_b].set(hb[self.cut_b:])
		flipped_hb = flipped_hb.at[self.cut_b:].set(hb[:self.cut_b])
		
		
		flipped_data = flipped_data.at[7:7+2*numga].set(flipped_cena.ravel())
		flipped_data = flipped_data.at[7+2*numga:7+2*numga+2*numgb].set(flipped_cenb.ravel())
		flipped_data = flipped_data.at[1].set(-flipped_data[1])
		
		flipped_data = flipped_data.at[7+2*numga+2*numgb:7+3*numga+2*numgb].set(flipped_ha)
		flipped_data = flipped_data.at[7+3*numga+2*numgb:7+3*numga+3*numgb].set(flipped_hb)
		#flipped_data = flipped_data.at[7+2*numga+2*numgb:7+3*numga+2*numgb].set(-1*flipped_data[7+2*numga+2*numgb:7+3*numga+2*numgb])
		final_flipped = self.calc_pot_c_jit(flipped_data[:7],flipped_cena,flipped_cenb,flipped_data[7+2*numga+2*numgb:7+3*numga+2*numgb],flipped_data[7+3*numga+2*numgb:7+3*(numga+numgb)])
		
		def no_flip(final_out):
			final_out = final_out.at[:7+3*(numga+numgb)].set(final_data[:7+3*(numga+numgb)])
			final_out = final_out.at[-1].set(final_pot)
			return final_out
		def flip(final_out):
			final_out = final_out.at[:7+3*(numga+numgb)].set(flipped_data[:7+3*(numga+numgb)])
			final_out = final_out.at[-1].set(final_flipped)
			return final_out
		final_out = jax.lax.cond(final_pot > final_flipped,flip,no_flip,final_out)	
		#final_out = final_out.at[-1].set(final_pot-final_flipped)
		#jax.debug.print("{x},{y}",x=final_pot,y=final_flipped)
		return final_out
			
	#We again turn minimse into a paralelised version
	@partial(jax.jit,static_argnums=2)
	def minimise_c_p(self,starting_pos,max_iter):
		return jax.pmap(self.minimise_c,static_broadcasted_argnums=1,in_axes=(0,None))(starting_pos,max_iter)
		
	@partial(jax.jit,static_argnums=(3))
	def get_centers_JAX(self,key,points,num):
		no_points = points.shape[0]
		#points = jax.random.permutation(key,points,independent=False,axis=0)
		ind_points = jnp.zeros(no_points)
		centers = jnp.zeros((no_points*(1+num)+1,2))
		centers = centers.at[no_points*num:-1].set(1)
		key,subkey = jax.random.split(key)
		perturb = jax.random.normal(subkey,(no_points*num,2))*5
		centers = centers.at[:no_points*num].set(perturb)
		def cen_fun_1(carry,ind):
			ind = ind%no_points
			def round1(carry):
				carry = carry.at[no_points*num:-1].set(carry[no_points*num:-1]-1)
				return carry
			def not_round1(carry):
				return carry
			carry = jax.lax.cond(ind == 0,round1,not_round1,carry)
			def zero(carry):
				indexer = jnp.array(carry[-1,0],dtype=int)
				carry = carry.at[indexer].set(carry[indexer]+points[ind])
				carry = carry.at[-1].set(carry[-1]+ 1)
				carry = carry.at[no_points*num+ind].set(1+carry[no_points*num+ind])
				fix_ind = ind
				def cen_fun_2(carry,ind):
					def close(carry):
						carry = carry.at[no_points*num+ind].set(1+carry[no_points*num+ind])
						return carry
					def not_close(carry):
						return carry
					carry = jax.lax.cond(jnp.linalg.norm(points[ind]-points[fix_ind])<30,close,not_close,carry)
					return carry,ind
				carry,_ = jax.lax.scan(cen_fun_2,carry,jnp.arange(no_points))
				return carry
			def not_zero(carry):
				return carry
			carry = jax.lax.cond(carry[no_points*num+ind,0] == 0 ,zero,not_zero,carry)
			return carry,ind
		centers,_=jax.lax.scan(cen_fun_1,centers,jnp.arange(no_points*num))
		return centers[:no_points],jnp.array(centers[-1,0],dtype=int)
		
	#This function gets the number of gaussians to use for the curvature minimisations	
	def get_no_gauss(self):
		key = jax.random.PRNGKey(234)
			
		min_poses = self.minima
		no_mins = (min_poses.shape[0]//4)
		def less():
			return no_mins
		def more():
			return self.only_top
		only_top = jax.lax.cond(self.only_top > no_mins,less,more)
		
		
		positioned_points_u = self.surface_poses[self.surface_poses[:,2]>0]
		positioned_points_l = self.surface_poses[self.surface_poses[:,2]<0]
		def is_memopt(key):
			return 2,2,1,1,0.00001
		def is_not_memopt(key):
			def start_pos_fun(carry,ind):
				start_zdir = position_point_jit(0,min_poses[ind][2],min_poses[ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
				start_xydir =  position_point_jit(0,min_poses[ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
				new_starting_pos = jnp.concatenate((min_poses[ind,:2],start_zdir,start_xydir))
				
				def not_double():
					new_key,subkey = jax.random.split(key)
					_,num_a = self.get_centers_JAX(subkey,position_pointv2_jit(new_starting_pos[1],new_starting_pos[2:5],new_starting_pos[5:7],self.surface_poses)[:,:2],1)
					def odd(num_a):
						return num_a+1
					def even(num_a):
						return num_a
					num_a = jax.lax.cond(num_a%2==0,even,odd,num_a)
					retval = jnp.array([jnp.array(num_a/2,dtype=int),jnp.array(num_a/2,dtype=int)])
					return retval
				def double():
					new_key,subkey = jax.random.split(key)
					_,num_a = self.get_centers_JAX(subkey,position_pointv2_jit(new_starting_pos[1],new_starting_pos[2:5],new_starting_pos[5:7],positioned_points_u)[:,:2],1)
					new_key,subkey = jax.random.split(key)
					_,num_b = self.get_centers_JAX(subkey,position_pointv2_jit(new_starting_pos[1],new_starting_pos[2:5],new_starting_pos[5:7],positioned_points_l)[:,:2],1)
					retval = jnp.array([jnp.max(jnp.array([num_a,num_b])),jnp.max(jnp.array([num_a,num_b]))])
					return retval
				retval = jax.lax.cond(min_poses[ind][0] > 1e-5,double,not_double)
				return carry,retval
			_,all_nums = jax.lax.scan(start_pos_fun,0,jnp.arange(only_top))
			return jnp.array(jnp.max(all_nums[:,0])+jnp.max(all_nums[:,1]),dtype=int),jnp.array(jnp.max(all_nums[:,0])+jnp.max(all_nums[:,1]),dtype=int),jnp.array(jnp.max(all_nums[:,0]),dtype=int),jnp.array(jnp.max(all_nums[:,0]),dtype=int),0.03
		
		
		self.numa,self.numb,self.cut_a,self.cut_b,lss_val = jax.lax.cond(self.memopt,is_memopt,is_not_memopt,key)
		self.cut_a = int(self.cut_a)
		self.cut_b = int(self.cut_b)
		self.numa = int(self.numa)
		self.numb = int(self.numb)
		#self.numa = int(jnp.max(all_nums[:,0])+jnp.max(all_nums[:,1]))
		#self.numb = int(jnp.max(all_nums[:,0])+jnp.max(all_nums[:,1]))
		#self.cut_a = int(jnp.max(all_nums[:,0]))
		#self.cut_b = int(jnp.max(all_nums[:,0]))
		
		
		self.lss_a = jnp.zeros((self.numa))+lss_val
		self.lss_b = jnp.zeros((self.numb))+lss_val
		
			

	#A funcion for formating data into a grid of starting positions for the minimse (WIP)
	def format_data(self):
		
		key = jax.random.PRNGKey(234)
		
		min_poses = self.minima
		no_mins = (min_poses.shape[0]//4)
		def less():
			return no_mins
		def more():
			return self.only_top
		only_top = jax.lax.cond(self.only_top > no_mins,less,more)
		pos_grid = jnp.zeros((only_top,3*(self.numa+self.numb)+8))
		
		starting_poses = jnp.zeros((only_top,7))
		
		centers_a = jnp.zeros((self.numa,2))
		centers_b = jnp.zeros((self.numb,2))
		heights_a = jnp.zeros(self.numa)
		heights_b = jnp.zeros(self.numb)

		positioned_points_u = self.surface_poses[self.surface_poses[:,2]>0]
		positioned_points_l = self.surface_poses[self.surface_poses[:,2]<0]
		def start_pos_fun(pos_grid,ind):
			start_zdir = position_point_jit(0,min_poses[ind][2],min_poses[ind][3],jnp.array([[0.0,0.0,1.0]]))[0]
			start_xydir =  position_point_jit(0,min_poses[ind][2],0.0,jnp.array([[0.0,-1.0,0.0]]))[0,:2]
			new_starting_pos = jnp.concatenate((min_poses[ind,:2],start_zdir,start_xydir))
			positioned_points = position_pointv2_jit(new_starting_pos[1],new_starting_pos[2:5],new_starting_pos[5:7],self.surface_poses)
			def is_memopt(positioned_points_u,positioned_points_l):
				return jnp.zeros((self.numa,2)),jnp.zeros((self.numb,2))
			def is_not_memopt(positioned_points_u,positioned_points_l):
				def not_double(positioned_points_u,positioned_points_l):
					new_key,subkey = jax.random.split(key)
					cens_a,_ = self.get_centers_JAX(subkey,positioned_points[:,:2],self.numa)
					centers_a = cens_a[:self.numa]
					
					new_key,subkey = jax.random.split(new_key)
					cens_b,_ = self.get_centers_JAX(subkey,positioned_points[:,:2],self.numb)
					centers_b = cens_b[:self.numb]
					return centers_a,centers_b
				def double(positioned_points_u,positioned_points_l):
					positioned_points_u = position_pointv2_jit(new_starting_pos[1],new_starting_pos[2:5],new_starting_pos[5:7],positioned_points_u)
					positioned_points_l = position_pointv2_jit(new_starting_pos[1],new_starting_pos[2:5],new_starting_pos[5:7],positioned_points_l)
					new_key,subkey = jax.random.split(key)
					cens_a,_ = self.get_centers_JAX(subkey,positioned_points_u[:,:2],self.cut_a)
					centers_a1 = cens_a[:self.cut_a]
					
					new_key,subkey = jax.random.split(new_key)
					cens_b,_ = self.get_centers_JAX(subkey,positioned_points_u[:,:2],self.cut_b)
					centers_b1 = cens_b[:self.cut_b]
					
					new_key,subkey = jax.random.split(key)
					cens_a,_ = self.get_centers_JAX(subkey,positioned_points_l[:,:2],self.cut_a)
					centers_a2 = cens_a[:self.cut_a]
					
					new_key,subkey = jax.random.split(new_key)
					cens_b,_ = self.get_centers_JAX(subkey,positioned_points_l[:,:2],self.cut_b)
					centers_b2 = cens_b[:self.cut_b]
					
					centers_a = jnp.concatenate((centers_a1,centers_a2))
					centers_b = jnp.concatenate((centers_b1,centers_b2))
					return centers_a,centers_b
				#jax.debug.print("{x},{y},{z},{d}",x=self.numa,y=self.numb,z=self.cut_a,d=self.cut_b)
				centers_a,centers_b = jax.lax.cond(min_poses[ind][0] > 1e-5, double, not_double,positioned_points_u,positioned_points_l)
				return centers_a,centers_b
			centers_a,centers_b = jax.lax.cond(self.memopt, is_memopt, is_not_memopt,positioned_points_u,positioned_points_l)
			starting_pos = jnp.concatenate((new_starting_pos,centers_a.ravel(),centers_b.ravel(),heights_a,heights_b,jnp.zeros(1)))
			pos_grid = pos_grid.at[ind].set(starting_pos)
			return pos_grid,ind
		self.pos_grid,_ = jax.lax.scan(start_pos_fun,pos_grid,jnp.arange(only_top))


	#A function that minimises on a grid of starting positions
	def minimise_on_grid_c(self,max_iter):
		self.curva = True
		no_mins = self.pos_grid.shape[0]
		result_grid = jnp.zeros_like(self.pos_grid)
		no_runs = jnp.ceil(no_mins/self.red_cpus).astype(int)
		print("Batch size:",self.red_cpus)
		print("Number of batches:",no_runs)
		times_taken = jnp.zeros(no_runs)
		#This non JAX for loop is needed beacuse pmap doesn't interact well with scan
		for i in range(no_runs):
			timers = time.time()
			print(str(i+1)+"/"+str(no_runs))
			result_grid = result_grid.at[i::no_runs].set(self.minimise_c_p(self.pos_grid[i::no_runs],max_iter).block_until_ready())
			print("Time per single minimisation: "+str(np.round((time.time()-timers)/self.red_cpus,3))+" s")
			times_taken = times_taken.at[i].set((time.time()-timers))
			if(i > 5):
				time_rem = jnp.array((no_runs-(i+1))*jnp.mean(times_taken[i-5:i+1]),dtype=int)
			elif(i>1):
				time_rem = jnp.array((no_runs-(i+1))*jnp.mean(times_taken[1:i+1]),dtype=int)
			else:
				time_rem = jnp.array((no_runs-(i+1))*jnp.mean(times_taken[:i+1]),dtype=int)
			print("Estimated remaining time: {:02}h {:02}m {:02}s".format(time_rem//3600,(time_rem%3600)//60,time_rem%60))
		self.result_grid_c = result_grid

	#A function that normalises position and curvature definitions (to allow comparisont)
	@jax.jit
	def normalise_pos_c_jit(self,position):
		new_pos = position.copy()
		
		new_pos = jnp.zeros(4+3*(self.numa+self.numb))
		new_pos = new_pos.at[4:].set(position[7:])
		new_pos = new_pos.at[:2].set(position[:2])
		zdir_test_a = position[2:5]
		zdir_test_a /= jnp.linalg.norm(zdir_test_a)
		new_pos = new_pos.at[3].set(jnp.arccos(jnp.dot(zdir_test_a,jnp.array([0.0,0.0,1.0]))))
		
		xydir_test = jnp.array([position[5],position[6],0])
		zdir_test = jnp.array([position[2],position[3],0])
		anga = 0.0
		angb = 0.0
		def z_dir_zero(anga,xydir_test,zdir_test):
			return 0.0,0.0
		def z_dir_not_zero(anga,xydir_test,zdir_test):
			xydir_test /= jnp.linalg.norm(xydir_test)
			zdir_test /= jnp.linalg.norm(zdir_test)
			anga = jnp.arctan2(jnp.dot(jnp.cross(zdir_test,xydir_test),jnp.array([0.0,0.0,1.0])),jnp.dot(xydir_test,zdir_test))
			angb = jnp.arctan2(jnp.dot(jnp.cross(zdir_test,jnp.array([0.0,-1.0,0.0])),jnp.array([0.0,0.0,1.0])),jnp.dot(jnp.array([0.0,-1.0,0.0]),zdir_test))
			return anga,angb
		anga,angb = jax.lax.cond(jnp.linalg.norm(zdir_test)<1e-5,z_dir_zero,z_dir_not_zero,anga,xydir_test,zdir_test)
		new_pos = new_pos.at[2].set(anga)
		
		
		centers_a = new_pos[4:4+2*self.numa].reshape((self.numa,2))
		to_pos_a = jnp.zeros((self.numa,3))
		to_pos_a = to_pos_a.at[:,:2].set(centers_a)
		centers_a = position_point_jit(0,angb,0,to_pos_a)[:,:2]
		
		
		centers_b = new_pos[4+2*self.numa:4+2*self.numa+2*self.numb].reshape((self.numb,2))
		to_pos_b = jnp.zeros((self.numa,3))
		to_pos_b = to_pos_b.at[:,:2].set(centers_b)
		centers_b = position_point_jit(0,angb,0,to_pos_b)[:,:2]
		
		new_pos = new_pos.at[4:4+2*self.numa].set(centers_a.ravel())
		new_pos = new_pos.at[4+2*self.numa:4+2*self.numa+2*self.numb].set(centers_b.ravel())

		
		heights_a = new_pos[4+2*self.numa+2*self.numb:4+3*self.numa+2*self.numb]
		heights_b = new_pos[4+3*self.numa+2*self.numb:4+3*(self.numa+self.numb)]
		
		def norm_fun_1(new_pos):
			new_pos = new_pos.at[3].set(new_pos[3]-2*jnp.pi)
			return new_pos
		def norm_cond_1(new_pos):
			return new_pos[3] > jnp.pi
		
		def norm_fun_2(new_pos):
			new_pos = new_pos.at[3].set(new_pos[3]+2*jnp.pi)
			return new_pos
		def norm_cond_2(new_pos):
			return new_pos[3] < -jnp.pi
			
		new_pos = jax.lax.while_loop(norm_cond_1,norm_fun_1,new_pos)
		new_pos = jax.lax.while_loop(norm_cond_2,norm_fun_2,new_pos)
		
		def is_neg(new_pos):
			new_pos = new_pos.at[3].set(-new_pos[3])
			new_pos = new_pos.at[2].set(new_pos[2]+jnp.pi)
			
			centers_a = new_pos[4:4+2*self.numa].reshape((self.numa,2))
			to_pos_a = jnp.zeros((self.numa,3))
			to_pos_a = to_pos_a.at[:,:2].set(centers_a)
			centers_a = position_point_jit(0,jnp.pi,0,to_pos_a)[:,:2]
			
			
			centers_b = new_pos[4+2*self.numa:4+2*self.numa+2*self.numb].reshape((self.numb,2))
			to_pos_b = jnp.zeros((self.numa,3))
			to_pos_b = to_pos_b.at[:,:2].set(centers_b)
			centers_b = position_point_jit(0,jnp.pi,0,to_pos_b)[:,:2]
			
			new_pos = new_pos.at[4:4+2*self.numa].set(centers_a.ravel())
			new_pos = new_pos.at[4+2*self.numa:4+2*self.numa+2*self.numb].set(centers_b.ravel())
			return new_pos
		def is_pos(new_pos):
			return new_pos
		new_pos = jax.lax.cond(new_pos[3]< 0,is_neg,is_pos,new_pos)
		
		def norm_fun_3(ang):
			ang += jnp.pi*2
			return ang
		def norm_cond_3(ang):
			return ang < 0
			
		def norm_fun_4(ang):
			ang -= jnp.pi*2
			return ang
		def norm_cond_4(ang):
			return ang > 2*jnp.pi
			
		new_pos = new_pos.at[2].set(jax.lax.while_loop(norm_cond_3,norm_fun_3,new_pos[2]))
		new_pos = new_pos.at[2].set(jax.lax.while_loop(norm_cond_4,norm_fun_4,new_pos[2]))
		
		centers_a = new_pos[4:4+2*self.numa].reshape((self.numa,2))
		centers_b = new_pos[4+2*self.numa:4+2*self.numa+2*self.numb].reshape((self.numb,2))

		def in_water(new_pos):
			new_pos = jnp.zeros_like(new_pos)
			new_pos = new_pos.at[1].set(-300)
			return new_pos
		def not_in_water(new_pos):
			return new_pos
		new_pos = jax.lax.cond(self.calc_in_water_c(position[:7],centers_a,centers_b,heights_a,heights_b,1e-5),in_water,not_in_water,new_pos)
		return new_pos


	#A function that collects all positions and normalises them
	@jax.jit
	def get_all_normal_spos_jit(self):
		grid_size = self.no_mins
		normal_spos = jnp.zeros((self.result_grid_c[:,:-1].shape[0],self.result_grid_c[:,:-1].shape[1]-3))
		def calc_poses_fun_1(normal_spos,ind):
			spos = self.normalise_pos_c_jit(self.result_grid_c[ind,:-1])
			normal_spos = normal_spos.at[ind].set(spos)
			return normal_spos,ind
			
		normal_spos,_ = jax.lax.scan(calc_poses_fun_1,normal_spos,jnp.arange(grid_size))
		return normal_spos
	
	#A function that collects all final positions and groups them into local minima
	def recollect_minima_info_c(self):
		old_minima = self.minima[self.no_mins:self.no_mins*2,0]
		normal_spos = self.normal_spos
		result_grid = self.result_grid_c
		grid_size = self.no_mins
		pos_tracker = jnp.array([[0,0,1]],dtype="float64")
		res_spos = jnp.zeros((grid_size,1,3))
		color_grid = jnp.zeros((grid_size,3),dtype="float64")

		#Calculating a simplified version of the oriented position
		def calc_poses_fun_1(res_spos,ind):
			spos = normal_spos[ind]
			part_pos = position_point(0,spos[2],spos[3],pos_tracker)
			res_spos = res_spos.at[ind].set(position_point(0,-spos[2],0,part_pos))
			return res_spos,ind
			
		res_spos,_ = jax.lax.scan(calc_poses_fun_1,res_spos,jnp.arange(grid_size))

		#Collecting minima which are equivilant
		minima_types = jnp.zeros(grid_size+1,dtype = "int")
		dtol_a = jnp.pi/24.0
		dtol_z = 5
		dtol_p = 100
		dtol_c = 1e6
		ep = 1-1e-6
		def calc_no_poses_fun_1(minima_types,ind):
			def calc_no_poses_fun_2(ind_fix,ind):
				def is_same(ind_fix):
					ind_fix = ind_fix.at[1].set(0)
					ind_fix = ind_fix.at[2].set(ind+1)
					return ind_fix
				def is_not_same(ind_fix):
					return ind_fix
				distance = jnp.array([10.0,10.0,10.0,10.0])
				def go(distance):
					spos = normal_spos[ind]
					sposf = normal_spos[ind_fix[0]]
					distance = distance.at[0].set(jnp.arccos(ep*jnp.dot(res_spos[ind][0],res_spos[ind_fix[0]][0])))
					distance = distance.at[1].set(jnp.abs(spos[1]-sposf[1]))
					distance = distance.at[2].set(jnp.abs(result_grid[ind][-1]-result_grid[ind_fix[0]][-1]))
					distance = distance.at[3].set(jnp.linalg.norm(spos[4:-1]-sposf[4:-1]))
					return distance
				def stop(distance):
					return distance
				distance = jax.lax.cond(ind_fix[1] > 0.5,go,stop,distance)
				ind_fix = jax.lax.cond((distance[0] < dtol_a)*(distance[1] < dtol_z)*(distance[2] < dtol_p)*(distance[3] < dtol_c),is_same,is_not_same,ind_fix)
				return ind_fix,ind
			ind_fix,_ = jax.lax.scan(calc_no_poses_fun_2,jnp.array([ind,1,0],dtype = "int"),jnp.arange(grid_size))
			ind_type = ind_fix[2]
			def is_prev(ind_type):
				ind_type = minima_types[ind_type-1]
				return ind_type
			def is_not_prev(ind_type):
				return ind_type
			ind_type = jax.lax.cond(ind_type-1<ind,is_prev,is_not_prev,ind_type)
			def is_big(ind_type,minima_types):
				minima_types = minima_types.at[-1].set(minima_types[-1]+1)
				ind_type = minima_types[-1]
				return ind_type,minima_types
			def is_not_big(ind_type,minima_types):
				return ind_type,minima_types
			ind_type,minima_types = jax.lax.cond(ind_type > minima_types[-1],is_big,is_not_big,ind_type,minima_types)
			minima_types = minima_types.at[ind].set(ind_type)
			return minima_types,ind
			
		minima_types,_= jax.lax.scan(calc_no_poses_fun_1,minima_types,jnp.arange(grid_size))
		no_mins = minima_types[-1]
		minima = jnp.zeros((no_mins*4,4))
		#Averaging collected minmia
		def av_minima_fun_1(minima,ind):
			spos = normal_spos[ind]
			minima = minima.at[minima_types[ind]-1].set(minima[minima_types[ind]-1]+spos[:4]*old_minima[ind])
			minima = minima.at[minima_types[ind]-1+no_mins].set(minima[minima_types[ind]-1+no_mins]+old_minima[ind])
			minima = minima.at[minima_types[ind]-1+2*no_mins].set(minima[minima_types[ind]-1+2*no_mins]+result_grid[ind][-1]*old_minima[ind])
			minima = minima.at[minima_types[ind]-1+3*no_mins].set(minima[minima_types[ind]-1+3*no_mins]+spos[0]*jnp.cos(spos[3])*old_minima[ind])
			return minima, ind
		minima,_ = jax.lax.scan(av_minima_fun_1,minima,jnp.arange(minima_types.shape[0]-1))
		minima = minima.at[:no_mins].set(minima[:minima_types[-1]]/minima[no_mins:2*no_mins])
		minima = minima.at[2*no_mins:3*no_mins].set(minima[2*no_mins:3*no_mins]/minima[no_mins:2*no_mins])
		minima = minima.at[3*no_mins:].set(minima[3*no_mins:]/minima[no_mins:2*no_mins])
		minima_ind = jnp.argsort(minima[no_mins:2*no_mins,0])[::-1]
		minima = minima.at[:no_mins].set(minima[minima_ind])
		minima = minima.at[no_mins:2*no_mins].set(minima[minima_ind+no_mins])
		minima = minima.at[2*no_mins:3*no_mins].set(minima[minima_ind+2*no_mins])
		minima = minima.at[3*no_mins:].set(minima[minima_ind+3*no_mins])
		
		min_mems = jnp.zeros((no_mins*2,result_grid.shape[1]-8))	
		
		def av_min_mem_fun_1(min_mem,ind):
			spos = normal_spos[ind]
			min_mem = min_mem.at[minima_types[ind]-1].set(min_mem[minima_types[ind]-1]+spos[4:])
			min_mem = min_mem.at[minima_types[ind]-1+no_mins].set(min_mem[minima_types[ind]-1+no_mins]+1)
			return min_mem, ind
		min_mems,_ = jax.lax.scan(av_min_mem_fun_1,min_mems,jnp.arange(grid_size))
		
		min_mem = min_mems.at[:no_mins].set(min_mems[:no_mins]/min_mems[no_mins:no_mins*2])[:no_mins]
		min_mem = min_mem[minima_ind]
		final_min_mem = jnp.zeros((no_mins,min_mem.shape[1]+self.lss_a.shape[0]+self.lss_b.shape[0]+2))
		def fill_min_mem(final_min_mem,ind):
			final_min_mem = final_min_mem.at[ind].set(jnp.concatenate((self.lss_a,self.lss_b,min_mem[ind],jnp.array([self.cut_a]),jnp.array([self.cut_b]))))
			return final_min_mem,ind
		final_min_mem,_ = jax.lax.scan(fill_min_mem,final_min_mem,jnp.arange(no_mins))
		self.minima_c = minima
		self.min_mem = final_min_mem
		self.minima_types_c = minima_types[:-1]
		self.minima_ind_c = minima_ind
		self.no_mins_c = minima_types[-1]
		
	#A function that makes grids for the output graphs
	def make_graph_grids(self,angs,grid_size):
		new_minima_types = jnp.zeros_like(self.minima_types)
		no_mins = self.minima_ind_c.shape[0]
		no_mins2 = self.minima_ind.shape[0]
		
		minima_ind_inv = jnp.array([jnp.arange(no_mins2)[self.minima_ind==i][0] for i in range(no_mins2)])
		minima_ind_c_inv = jnp.array([jnp.arange(no_mins)[self.minima_ind_c==i][0] for i in range(no_mins)])
		
		def new_minima_types_fun(new_minima_types,ind):
			new_minima_types = new_minima_types.at[ind].set(minima_ind_c_inv[self.minima_types_c[minima_ind_inv[self.minima_types[ind]-1]]-1])
			return new_minima_types,ind
		
		new_minima_types,_ = jax.lax.scan(new_minima_types_fun,new_minima_types,jnp.arange(grid_size))
		pot_grid = jnp.zeros(grid_size)
		def set_pot_grid(pot_grid,ind):
			pot_grid = pot_grid.at[ind].set(self.minima_c[no_mins*2+new_minima_types[ind],0])
			return pot_grid,ind
		pot_grid,_ = jax.lax.scan(set_pot_grid,pot_grid,jnp.arange(grid_size))
		devs = jnp.zeros(no_mins)
		ep = 1-1e-6
		def get_min_devs(devs,ind):
			sposind_p = position_point_jit(0,self.minima_c[ind+1][2],self.minima_c[ind+1][3],jnp.array([[0,0,1]]))
			sposind = position_point_jit(0,-self.minima_c[ind+1][2],0,sposind_p)[0]
			sposo_p = position_point_jit(0,self.minima_c[0][2],self.minima_c[0][3],jnp.array([[0,0,1]]))
			sposo = position_point_jit(0,-self.minima_c[0][2],0,sposo_p)[0]
			dev1 = jnp.arccos(jnp.dot(sposind,sposo)*ep)
			dev2 = jnp.abs(self.minima_c[ind+1][0]-self.minima_c[0][0])/100
			dev3 = jnp.linalg.norm(self.min_mem[ind+1]-self.min_mem[0])/100
			devs = devs.at[ind+1].set(dev1+dev2+dev3)
			return devs,ind
		devs,_ = jax.lax.scan(get_min_devs,devs,jnp.arange(no_mins))
		
		color_grid = jnp.zeros((grid_size,3),dtype="float64")
		def color_grid_fun_2(color_grid,ind):
			colr = 1.0
			colg = 1-devs[new_minima_types[ind]]/3
			colb = 1-devs[new_minima_types[ind]]/3

			def gone(col):
				col = 1.0
				return col
			def lone(col):
				return col
			
			def lzero(col):
				col=0.0
				return col
			def gzero(col):
				return col
				
			colr = jax.lax.cond(colr > 1, gone,lone,colr)
			colg = jax.lax.cond(colg > 1, gone,lone,colg)
			colb = jax.lax.cond(colb > 1, gone,lone,colb)
	 
			colr = jax.lax.cond(colr < 0, lzero,gzero,colr)
			colg = jax.lax.cond(colg < 0, lzero,gzero,colg)
			colb = jax.lax.cond(colb < 0, lzero,gzero,colb)
			color_grid = color_grid.at[ind].set(jnp.array([colr,colg,colb]))
			return color_grid,ind
			
		color_grid,_ = jax.lax.scan(color_grid_fun_2,color_grid,jnp.arange(color_grid.shape[0]))
		
		return pot_grid,color_grid
	
	#A function that makes graphs of curvature and thinning of the membrane for each minima
	def make_mem_graphs(self,orient_dir,grid_num):
		min_zdist = self.minima_c[self.no_mins_c*3:,0]
		for i in range(self.min_mem.shape[0]):
			if(not os.path.exists(orient_dir+"Rank_"+str(i+1)+"/")):
				os.mkdir(orient_dir+"Rank_"+str(i+1)+"/")
			rank_dir = orient_dir+"Rank_"+str(i+1)+"/"
			xs = jnp.linspace(-100,100,grid_num)
			ys = jnp.linspace(-100,100,grid_num)
			mem_info = self.min_mem[i]
			lss_a = mem_info[:self.numa]
			lss_b = mem_info[self.numa:self.numb+self.numa]
			centers_a = mem_info[self.numa+self.numb:3*self.numa+self.numb].reshape((self.numa,2))
			centers_b = mem_info[3*self.numa+self.numb:3*self.numa+3*self.numb].reshape((self.numb,2))
			heights_a = mem_info[3*self.numa+3*self.numb:4*self.numa+3*self.numb]
			heights_b = mem_info[4*self.numa+3*self.numb:4*self.numa+4*self.numb]
			cuta = int(mem_info[-2])
			cutb = int(mem_info[-1])
			grid = []
			grid2 = []
			lss = []
			heights = []
			centers = []
			if(abs(min_zdist[i]) > 1e-5):
				grid_hu,grid_h2u,_ = make_membrane_jit(lss_a[:cuta],lss_b[:cutb],centers_a[:cuta],centers_b[:cutb],heights_a[:cuta],heights_b[:cutb],grid_num,xs,ys)
				grid_hl,grid_h2l,_ = make_membrane_jit(lss_a[cuta:],lss_b[cutb:],centers_a[cuta:],centers_b[cutb:],heights_a[cuta:],heights_b[cutb:],grid_num,xs,ys)
				grid = [np.array(grid_hu),np.array(grid_hl)]
				grid2= [np.array(grid_h2u),np.array(grid_h2l)]
				lss = [lss_a[:cuta],lss_a[cuta:]]
				heights = [heights_a[:cuta],heights_a[cuta:]]
				centers = [centers_a[:cuta],centers_a[cuta:]]
			else:
				grid_h,grid_h2,_ = make_membrane_jit(lss_a,lss_b,centers_a,centers_b,heights_a,heights_b,grid_num,xs,ys)
				grid = [np.array(grid_h)]
				grid2= [np.array(grid_h2)]
				lss = [lss_a]
				heights = [heights_a]
				centers = [centers_a]
			for j in range(len(grid2)):
				plt.imshow(grid2[j],extent=[-100,100,-100,100])
				plt.xlabel("x")
				plt.ylabel("y")
				plt.title("Membrane thickness")
				plt.tight_layout()
				plt.savefig(rank_dir+"mem_thick_rank"+str(i+1)+"_"+str(j)+".png")
				plt.clf()
				gcurv = jnp.zeros((grid_num,grid_num))
				def calc_fun_1(gcurv,ind):
					def calc_fun_2(ind_fix,ind):
						row = gc_jit(lss[j],heights[j],centers[j],xs[ind_fix],ys[ind])
						return ind_fix,row
					_,row = jax.lax.scan(calc_fun_2,ind,jnp.arange(grid_num))
					gcurv =gcurv.at[ind].set(row)
					return gcurv,ind
				gcurv,_ = jax.lax.scan(calc_fun_1,gcurv,jnp.arange(grid_num))
				plt.imshow(gcurv,extent=[-100,100,-100,100])
				plt.xlabel("x")
				plt.ylabel("y")
				plt.title("Membrane curvature")
				plt.tight_layout()
				plt.savefig(rank_dir+"mem_curv_rank"+str(i+1)+"_"+str(j)+".png")
				plt.clf()

#Registering the class as a pytree node for JAX
tree_util.register_pytree_node(MemBrain,MemBrain._tree_flatten,MemBrain._tree_unflatten)

#A helper function the writes and arbritrary PDB file
def write_point(points,fn,orient_dir):
	new_file = open(os.path.join(orient_dir,fn),"w")
	count = 0
	for i in points:
		count += 1
		count_str = (6-len(str(count)))*" "+str(count)
		c = "ATOM "+count_str+" BB   DUM     1       0.000   0.000  15.000  1.00  0.00" 
		xp = np.format_float_positional(i[0],precision=3)
		yp = np.format_float_positional(i[1],precision=3)
		zp = np.format_float_positional(i[2],precision=3)
		xp += "0"*(3-len((xp.split(".")[1])))
		yp += "0"*(3-len((yp.split(".")[1])))
		zp += "0"*(3-len((zp.split(".")[1])))
		new_c = c[:30]+(" "*(8-len(xp)))+xp+(" "*(8-len(yp)))+yp+(" "*(8-len(zp))) +zp+c[54:]+"\n"	
		new_file.write(new_c)
	new_file.close()
	
	
#A sigmoid function
def sigmoid(x,grad):
	ret_val = -x*grad
	def overflow_pos(ret_val):
		ret_val = 100.0
		return ret_val
	def overflow_neg(ret_val):
		ret_val = -100.0
		return ret_val
	def not_overflow(ret_val):
		return ret_val
	
	ret_val = jax.lax.cond(-x*grad>100,overflow_pos,not_overflow,ret_val)
	ret_val = jax.lax.cond(-x*grad<-100,overflow_neg,not_overflow,ret_val)
	return 1.0/(1.0+jnp.exp(ret_val))
	
sj = jax.jit(sigmoid)


#Functions for calculating sums of gaussians
def gaussian(ls,height,cx,cy,x,y):
	return height*jnp.exp(-(ls*(x-cx))**2-(ls*(y-cy))**2)
	
def gaussian_sum(lss,heights,centers,x,y):
	total = 0
	def sum_fun_1(total,ind):
		total = total +gaussian(lss[ind],heights[ind],centers[ind][0],centers[ind][1],x,y)
		return total,ind
	total,_ = jax.lax.scan(sum_fun_1,total,jnp.arange(heights.shape[0]))
	return total
	
guassian_sum_dx = jax.grad(gaussian_sum,argnums=3)
guassian_sum_dy = jax.grad(gaussian_sum,argnums=4)	
guassian_sum_d2yi = jax.grad(guassian_sum_dy,argnums=(3,4))
guassian_sum_d2xi = jax.grad(guassian_sum_dx,argnums=(3,4))	

#A function for calculating the curvature
def gaussian_curvature(lss,heights,centers,x,y):
	gxx = 	guassian_sum_d2xi(lss,heights,centers,x,y)[0]
	gxy = 	guassian_sum_d2xi(lss,heights,centers,x,y)[1]
	gyy = 	guassian_sum_d2yi(lss,heights,centers,x,y)[1]
	gx = guassian_sum_dx(lss,heights,centers,x,y)
	gy = guassian_sum_dy(lss,heights,centers,x,y)
	gauss_curv = (gxx*gyy-gxy*gxy)/((1+gx*gx+gy*gy)*(1+gx*gx+gy*gy))
	return gauss_curv
	
gc_jit = jax.jit(gaussian_curvature)

#A function that gets a cost associated with curvature
def curv_cost(lss,heights,centers):
	gcnum = 100	
	xx_min = -200.0
	xx_step = 400.0/(gcnum-1)
	yy_min = -200.0
	yy_step = 400.0/(gcnum-1)
	gcurv = 0.0
	def calc_fun_1(gcurv,ind):
		def calc_fun_2(ind_fix,ind):
			ifix = jnp.array(ind_fix[1],dtype=int)
			ind_fix = ind_fix.at[0].set(ind_fix[0]+jnp.abs(jnp.power(gc_jit(lss,heights,centers,xx_min+ifix*xx_step,yy_min+ind*yy_step),2)))
			return ind_fix,ind
		ind_fix = jnp.array([gcurv,ind],dtype=float)
		ind_fix,_ = jax.lax.scan(calc_fun_2,ind_fix,jnp.arange(gcnum))
		gcurv = ind_fix[0]
		return gcurv,ind
	gcurv,_ = jax.lax.scan(calc_fun_1,gcurv,jnp.arange(gcnum))
	return gcurv*1e4
	
cc_jit = jax.jit(curv_cost)

#A function that calculates a cost associated to the area under a gaussian sum (kind of)
def gaussian_sum_cost(lss,heights,centers):
	gcnum = 100	
	xx_min = -200.0
	xx_step = 400.0/(gcnum-1)
	yy_min = -200.0
	yy_step = 400.0/(gcnum-1)
	tcurv = 0.0
	def calc_fun_1(tcurv,ind):
		def calc_fun_2(ind_fix,ind):
			ifix = jnp.array(ind_fix[1],dtype=int)
			ind_fix = ind_fix.at[0].set(ind_fix[0]+jnp.abs(jnp.power(gaussian_sum(lss,heights,centers,xx_min+ifix*xx_step,yy_min+ind*yy_step),2)))
			return ind_fix,ind
		ind_fix = jnp.array([tcurv,ind],dtype=float)
		ind_fix,_ = jax.lax.scan(calc_fun_2,ind_fix,jnp.arange(gcnum))
		tcurv = ind_fix[0]
		return tcurv,ind
	tcurv,_ = jax.lax.scan(calc_fun_1,tcurv,jnp.arange(gcnum))
	return tcurv
	
gauss_sum_cost_jit = jax.jit(gaussian_sum_cost)

#A function that gets the normal at every point on a surface of a gaussian sum
def normal_vec(lss,heights,centers,x,y):
	total = jnp.zeros(3)
	total = total.at[2].set(-1)
	def sum_fun_2(total,ind):
		total = total.at[0].set(total[0] + -2*lss[ind]**2*(x-centers[ind][0])*gaussian(lss[ind],heights[ind],centers[ind][0],centers[ind][1],x,y))
		total = total.at[1].set(total[1] + -2*lss[ind]**2*(y-centers[ind][1])*gaussian(lss[ind],heights[ind],centers[ind][0],centers[ind][1],x,y))
		return total,ind
	total,_ = jax.lax.scan(sum_fun_2,total,jnp.arange(heights.shape[0]))
	total = total/jnp.linalg.norm(total)
	return total
	
	
#A function that creates a series of grid used for plotting a membrane
def make_membrane(lss_a,lss_b,centers_a,centers_b,heights_a,heights_b,grid_num,xs,ys):
	grid_h = jnp.zeros((grid_num,grid_num))
	grid_h2 = jnp.zeros((grid_num,grid_num))
	grid_norms = jnp.zeros((grid_num,grid_num,3))
	def calc_fun_1(grid_h,ind):
		def calc_fun_2(ind_fix,ind):
				return ind_fix,gaussian_sum(lss_a,heights_a,centers_a,xs[ind_fix],ys[ind])
		_,row = jax.lax.scan(calc_fun_2,ind,jnp.arange(grid_num))
		grid_h = grid_h.at[ind].set(row)
		return grid_h,ind
	grid_h,_ = jax.lax.scan(calc_fun_1,grid_h,jnp.arange(grid_num))

	def calc_fun_1(grid_h2,ind):
		def calc_fun_2(ind_fix,ind):
				return ind_fix,gaussian_sum(lss_b,heights_b,centers_b,xs[ind_fix],ys[ind])
		_,row = jax.lax.scan(calc_fun_2,ind,jnp.arange(grid_num))
		grid_h2 = grid_h2.at[ind].set(row)
		return grid_h2,ind
	grid_h2,_ = jax.lax.scan(calc_fun_1,grid_h2,jnp.arange(grid_num))
	def calc_fun_1(grid_norms,ind):
		def calc_fun_2(ind_fix,ind):
			return ind_fix,normal_vec(lss_a,heights_a,centers_a,xs[ind_fix],ys[ind])
		_,row = jax.lax.scan(calc_fun_2,ind,jnp.arange(grid_num))
		grid_norms= grid_norms.at[ind].set(row)
		return grid_norms,ind
	grid_norms,_ = jax.lax.scan(calc_fun_1,grid_norms,jnp.arange(grid_num))	
	return grid_h,grid_h2,grid_norms

make_membrane_jit = jax.jit(make_membrane,static_argnums = 6)

#A function that gets the distance between two points on adjacent surfaces
def distance_to_point(xy,point,lss_a,centers_a,heights_a):
	return jnp.linalg.norm(point-jnp.array([xy[0],xy[1],gaussian_sum(lss_a,heights_a,centers_a,xy[0],xy[1])]))
	
grad_dtp = jax.grad(distance_to_point,argnums=0)

#This function uses to above to find the point on the surface whose normal contains the point
def mem_pos(point,lss_a,centers_a,heights_a):
	def minimise(starting_pos,max_iter):		
		data = jnp.zeros(6)
		gamma = 0.75
		ep = 1e-8
		tol = 1e-10
		data = data.at[:2].set(starting_pos[:2])
		data = data.at[4:6].set(jnp.array([0.01,0.01]))
		def min_fun_1(data,ind):
			grad = grad_dtp(data[:2],starting_pos,lss_a,centers_a,heights_a)
			data = data.at[2:4].set(data[2:4]*gamma + (1-gamma)*grad*grad)
			rms_grad = jnp.sqrt(ep + data[2:4])
			rms_change = jnp.sqrt(ep+data[4:6])
			change = -(rms_change/rms_grad)*grad
			data = data.at[:2].set(data[:2]+change)
			data = data.at[4:6].set(gamma*data[4:6]+(1-gamma)*change*change)
			return data,ind
		final_data,_ = jax.lax.scan(min_fun_1,data,None,length = max_iter)
		return final_data[:2]
	min_point = minimise(point,10)
	normal = normal_vec(lss_a,heights_a,centers_a,min_point[0],min_point[1])
	final_point = jnp.concatenate((min_point,jnp.array([gaussian_sum(lss_a,heights_a,centers_a,min_point[0],min_point[1])])))
	direc = final_point-point
	dist = jnp.dot(direc,normal)
	return dist,min_point
	
mem_pos = jax.checkpoint(mem_pos)


#Setting up quaternions for rotations later
def qmul(q1,q2):
	part1 = q1[0]*q2
	part2 = jnp.array([-q1[1]*q2[1],q1[1]*q2[0],-q1[1]*q2[3],q1[1]*q2[2]])
	part3 = jnp.array([-q1[2]*q2[2],q1[2]*q2[3],q1[2]*q2[0],-q1[2]*q2[1]])
	part4 = jnp.array([-q1[3]*q2[3],-q1[3]*q2[2],q1[3]*q2[1],q1[3]*q2[0]])
	return part1+part2+part3+part4
qmul_jit = jax.jit(qmul)	


def qcong(q1):
	q1 = q1.at[1:].set(-1*q1[1:])
	return q1
qcong_jit = jax.jit(qcong)

#function for calculating potential between two charged sheets
@partial(jax.jit,static_argnums=(1,2,3))
def potential_between_sheets(rdist,grid_size,extent,cc):
	xs = jnp.linspace(-extent,extent,grid_size)
	tot_pot = 0
	def pbs1(tot_pot,ind):
		ind_fix1=ind
		def pbs2(tot_pot,ind):
			point_a = jnp.array([xs[ind],xs[ind_fix1],0.0])
			ind_fix2 = ind
			def pbs3(tot_pot,ind):
				ind_fix3 = ind
				def pbs4(tot_pot,ind):
					point_b = jnp.array([xs[ind_fix3],xs[ind],rdist])
					tot_pot += (extent/grid_size)*(extent/grid_size)*(extent/grid_size)*(extent/grid_size)/jnp.linalg.norm(point_a-point_b)
					return tot_pot,ind
				tot_pot,_=jax.lax.scan(pbs4,tot_pot,jnp.arange(grid_size))
				return tot_pot,ind
			tot_pot,_=jax.lax.scan(pbs3,tot_pot,jnp.arange(grid_size))
			return tot_pot,ind
		tot_pot,_=jax.lax.scan(pbs2,tot_pot,jnp.arange(grid_size))
		return tot_pot,ind
	tot_pot,_=jax.lax.scan(pbs1,tot_pot,jnp.arange(grid_size))
	return tot_pot*cc
	
#A function for positioning a set of points
def position_point(in_depth,ang1,ang2,poses):
	direc1 = jnp.array([0,0,1],dtype="float64")
	direc2 = jnp.array([1,0,0],dtype="float64")
	rot1q = jnp.array([jnp.cos(ang1/2),direc1[0]*jnp.sin(ang1/2),direc1[1]*jnp.sin(ang1/2),direc1[2]*jnp.sin(ang1/2)])
	rot2q = jnp.array([jnp.cos(ang2/2),direc2[0]*jnp.sin(ang2/2),direc2[1]*jnp.sin(ang2/2),direc2[2]*jnp.sin(ang2/2)])
	rot1qi = qcong_jit(rot1q)
	rot2qi = qcong_jit(rot2q)
	rotated_poses = poses.copy()
	def postion_fun_1(carry,ind):
		p_to_rot = rotated_poses[ind]
		qp = jnp.array([0,p_to_rot[0],p_to_rot[1],p_to_rot[2]])
		rot_qp1 = qmul_jit(rot1q,qmul_jit(qp,rot1qi))
		rot_qp = qmul_jit(rot2q,qmul_jit(rot_qp1,rot2qi))
		rot_p = rot_qp[1:]
		return carry,rot_p
	_,rotated_poses = jax.lax.scan(postion_fun_1,0,jnp.arange(poses.shape[0]))
	rotated_poses = rotated_poses.at[:,2].set(rotated_poses[:,2]-in_depth)
	
	return rotated_poses
position_point_jit = jax.jit(position_point)


#A function for positioning a set of points (in a smoother manner as this is better for minimisation)
def position_pointv2(in_depth,zdir,xydir,poses):
	ep = 1-1e-8
	zdir /= jnp.linalg.norm(zdir)
	
	
	direc1 = jnp.cross(zdir,jnp.array([0.0,0.0,1.0]))
	ang1 = ep*jnp.dot(zdir,jnp.array([0.0,0.0,1.0]))
	
	rot1qi = jnp.array([ang1+1,direc1[0],direc1[1],direc1[2]])
	rot1qi /= jnp.linalg.norm(rot1qi)
	rot1q = qcong_jit(rot1qi)
	
	
	
	xydir_cor = jnp.array([xydir[0],xydir[1],0.0])
	xydir_cor /= jnp.linalg.norm(xydir_cor)
	
	xycqp = jnp.array([0.0,xydir_cor[0],xydir_cor[1],xydir_cor[2]])
	xycrot_qp = qmul_jit(rot1q,qmul_jit(xycqp,rot1qi))
	xycrot_p = xycrot_qp[1:]
	
	xyqp = jnp.array([0.0,0.0,-1.0,0.0])
	xyrot_qp = qmul_jit(rot1q,qmul_jit(xyqp,rot1qi))
	xyrot_p = xyrot_qp[1:]
	
	direc2 = jnp.cross(xyrot_p,xycrot_p)
	ang2 = ep*jnp.dot(xyrot_p,xycrot_p)
	
	rot2q = jnp.array([ang2+1,direc2[0],direc2[1],direc2[2]])
	rot2q /= jnp.linalg.norm(rot2q)
	rot2qi = qcong_jit(rot2q)
	
	
	rotated_poses = poses.copy()
	def postion_fun_1(carry,ind):
		p_to_rot = rotated_poses[ind]
		qp = jnp.array([0,p_to_rot[0],p_to_rot[1],p_to_rot[2]])
		rot_qp1 = qmul_jit(rot1q,qmul_jit(qp,rot1qi))
		rot_qp = qmul_jit(rot2q,qmul_jit(rot_qp1,rot2qi))
		rot_p = rot_qp[1:]
		return carry,rot_p
	_,rotated_poses = jax.lax.scan(postion_fun_1,0,jnp.arange(poses.shape[0]))
	rotated_poses = rotated_poses.at[:,2].set(rotated_poses[:,2]-in_depth)
	
	return rotated_poses
position_pointv2_jit = jax.jit(position_pointv2)

#A function that returns an array of points on a sphere using a fibbonacci spiral lattice
def create_sph_grid(bsize):
	sgrid = jnp.zeros((bsize,2))
	gr = (1+jnp.sqrt(5))/2
	def ball_fun_3(sgrid,ind):
		phi = jnp.arccos(1-2*(ind+0.5)/(bsize*2))
		theta = jnp.pi*(ind+0.5)*(gr)*2
		def norm_fun_1(theta):
			theta = theta - 2*jnp.pi
			return theta
		def norm_cond_1(theta):
			return theta > jnp.pi*2
		
		def norm_fun_2(theta):
			theta = theta + 2*jnp.pi
			return theta
		def norm_cond_2(theta):
			return theta < 0
			
		theta = jax.lax.while_loop(norm_cond_1,norm_fun_1,theta)
		theta = jax.lax.while_loop(norm_cond_2,norm_fun_2,theta)
		sgrid = sgrid.at[ind].set(jnp.array([phi,theta]))
		return sgrid, ind
		
	sgrid,_ = jax.lax.scan(ball_fun_3,sgrid,np.arange(bsize))
	return sgrid

#A fucntion that reads a martini file to get interaction strengths
def get_int_strength(bead_1,bead_2,martini_file):
	string = " "*(6-len(bead_1))+bead_1+" "*(6-len(bead_2))+bead_2
	string2 = " "*(6-len(bead_2))+bead_2+" "*(6-len(bead_1))+bead_1
	mfile = open(martini_file,"r")
	content = mfile.readlines()
	for i,line in enumerate(content):
		if(string in line or string2 in line):
			return -float(line[32:45])
			
def get_mem_def(martini_file):
	#We use interactions strengths from martini using a POPE(Q4p)/POPG(P4)/POPC(Q1)? lipid as a template
	W_B_mins = jnp.array([get_int_strength("W",Beadtype(i).name,martini_file) for i in range(22)])
	LH1_B_mins = jnp.array([get_int_strength("P4",Beadtype(i).name,martini_file) for i in range(22)])
	LH2_B_mins = jnp.array([get_int_strength("Q5",Beadtype(i).name,martini_file) for i in range(22)])
	LH3_B_mins = jnp.array([get_int_strength("SN4a",Beadtype(i).name,martini_file) for i in range(22)])
	LH4_B_mins = jnp.array([get_int_strength("N4a",Beadtype(i).name,martini_file) for i in range(22)])
	LT1_B_mins = jnp.array([get_int_strength("C1",Beadtype(i).name,martini_file) for i in range(22)])
	LT2_B_mins = jnp.array([get_int_strength("C4h",Beadtype(i).name,martini_file) for i in range(22)])
	Charge_B_mins =jnp.array([0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,-1,-1,0,0,0,0],dtype="float64")
															   # #
	return (W_B_mins,LH1_B_mins,LH2_B_mins,LH3_B_mins,LH4_B_mins,LT1_B_mins,LT2_B_mins,Charge_B_mins)	

#Ploting some graphs of local minima. This is more complex than a simple plot as
#The information is on a spiral lattice.
def create_graphs(orient_dir,col_grid,pot_grid,angs,resa):
	graph_mesh_cols = jnp.zeros((resa,resa,4))
	graph_mesh_vals = jnp.zeros((resa,resa,2))
	
	
	def get_ind(resa,angs):
		ang1 = angs[0]
		ang2 = angs[1]
		ind1 = resa*ang1/(jnp.pi/2)
		ind2 = resa*ang2/(2*jnp.pi)
		ind1 = jnp.floor(ind1)
		ind2 = jnp.floor(ind2)
		ind1 = jnp.array(ind1,dtype=int)
		ind2 = jnp.array(ind2,dtype=int)
		return ind1,ind2
	def set_vals(graph_mesh,vals,leng):
		graph_mesh = graph_mesh.at[:,:,-1].set(1e-5)
		def set_vals_fun_1(graph_mesh,ind):
			ind1,ind2 = get_ind(resa,angs[ind])
			graph_mesh = graph_mesh.at[ind1,ind2,:-1].set(graph_mesh[ind1,ind2,:-1]+vals[ind])
			graph_mesh = graph_mesh.at[ind1,ind2,-1].set(graph_mesh[ind1,ind2,-1]+1)
			return graph_mesh,ind
		graph_mesh,_ = jax.lax.scan(set_vals_fun_1,graph_mesh,jnp.arange(angs.shape[0]))
		def div_fun_1(d_graph_mesh,ind):
			def div_fun_2(ind_fix,ind):
				divided = graph_mesh[ind_fix,ind,:-1]/graph_mesh[ind_fix,ind,-1]
				def empty(divided):
					return jnp.zeros(leng)+jnp.nan
				def nempty(divided):
					return divided
				divided = jax.lax.cond(graph_mesh[ind_fix,ind,-1] > 0.5, nempty,empty,divided)
				return ind_fix,divided
			_,row = jax.lax.scan(div_fun_2,ind,jnp.arange(resa))
			d_graph_mesh = d_graph_mesh.at[ind].set(row)
			return d_graph_mesh,ind
		
		final_mesh,_ = jax.lax.scan(div_fun_1,jnp.zeros((resa,resa,leng)),jnp.arange(resa))
		return final_mesh
	graph_mesh_cols = set_vals(graph_mesh_cols,col_grid,3)
	graph_mesh_vals = set_vals(graph_mesh_vals,pot_grid,1)
	plt.imshow(np.array(graph_mesh_cols),extent=[0,jnp.pi*2,0,jnp.pi/2],aspect=4)
	plt.xlabel("Theta")
	plt.ylabel("Phi")
	plt.title("Orientation of minima given starting position (z,theta,phi)")
	plt.tight_layout()
	plt.savefig(orient_dir+"local_minima_orientation.png")
	plt.imshow(graph_mesh_vals,extent=[0,jnp.pi*2,0,jnp.pi/2],aspect=4)
	plt.xlabel("Theta")
	plt.ylabel("Phi")
	plt.title("Relative potential energy of minima given starting position (z,theta,phi)")
	plt.tight_layout()
	plt.savefig(orient_dir+"local_minima_potential.png")

#concerts three letter codes to one for transmembrane residue output
def three2one(s):
	if(s == "ALA"):
		return "A"
	if(s == "ARG"):
		return "R"
	if(s=="LYS"):
		return "K"
	if(s == "ASN"):
		return "N"
	if(s == "ASP"):
		return "D"
	if(s == "CYS"):
		return "C"
	if(s == "GLU"):
		return "E"
	if(s == "GLN"):
		return "Q"
	if(s == "GLY"):
		return "G"
	if(s == "HIS"):
		return "H"
	if(s == "ILE"):
		return "I"
	if(s == "LEU"):
		return "L"
	if(s == "MET"):
		return "M"
	if(s == "PHE"):
		return "F"
	if(s == "PRO"):
		return "P"
	if(s == "SER"):
		return "S"
	if(s == "THR"):
		return "T"
	if(s == "TRP"):
		return "W"
	if(s == "TYR"):
		return "Y"
	if(s == "VAL"):
		return "V"
#formatting function	
def form(val):
	new_val = np.format_float_positional(val,precision=3)
	return new_val
