import sys,os
import numpy 
import MDAnalysis
import MDAnalysis.analysis.leaflet
import matplotlib.pyplot as plt
import numpy.ma
import matplotlib.cm



## This script needs a cenetered fitted trajectory 
## use trjconv -fit transxy+rotxy
##

####################################################
##### Variables you can change #####################
####################################################

#### frames from the trajectory to analyse
start = 1
finish = 5000

#### species to use for bilayer thickness calculations - for atomistic change to P8
n = "No longer needed"

thickstring = "name PO4 or name PO1 or name PO2 or name PO3"

#### these are the species (atoms or CG particles) for which we find densities

### Uncomment the line below for an atomistic system  #####
#Species = ["P8", "C13",  "CA2",  "C50" , "C43", "C3", "N4", "O4", "O1", "C3", "C39" , "O16", "O35" , "C44" , "C20", "C26",  "C42" , "C41" , "C45" , "C46" , "C47", "C48" ,"C49", "C50", "C4*" , "C3*" ]

### Uncomment the line below for a coarse grained system   #####
Species = ["PO4","NC3","GLH","NH3","GL1","GL2", "C1A", "PO1", "PO2","NCO","AM1","AM2","PO3","ROH" ]

#### Flags for ploting residues on 2D plots
plot_basic = True
plot_acidic = True
plot_side = False
plot_CA = True

### Colorscheme see matplotlib.cm ?? for option or create your own thick is for thickness CM is fro densities
#CM = matplotlib.cm.gray
#CM = matplotlib.cm.summer
### define your own colormap
cdict = {'red': (#(0.0, 0.0, 0.5),
                 (0.0, 1.0, 1.0),
                 (0.25, 1.0, 1.0),
                 (0.35, 0.0, 0.0),
                 (0.6, 0.0, 0.0),
                 (1.0, 1.0, 0.0)),
         'green': (#(0.0, 0.0, 0.0),
                 (0.0, 0.75, 0.75),
                 (0.25, 1.0, 1.0),
                 (0.35, 1.0, 1.0),
                 (0.6, 0.5, 0.5),
                 (1.0, 1.0, 0.0)),
         'blue': (#(0.0, 0.0, 0.5),
                 (0.0, 0.79, 0.79),
                 (0.25, 1.0, 1.0),
                 (0.35, 1.0, 1.0),
                 (0.6, 0.0, 0.0),
                 (1.0, 0.0, 0.0))}
CM = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

####################################################
##### Variables you shouldn't need to change #######
####################################################

##### only find leaflets for the first 'leafletts' frames
leafletts = 1

### selection strings to select different types of residues
sel_basic = "(resname ARG or resname LYS or resname HIS)"   ## basic residues
sel_acidic = "(resname ASP or resname GLU)"                 ## acidic residues

####################################################
##### Load the trajectory into the universe ########
####################################################

#### load the trajectory
M = MDAnalysis.Universe(sys.argv[1],sys.argv[2])
print ("loaded trajectory\n")
print (M.trajectory.dt, "ps between frames") 


##### parameters for discretising the data
xrange = [-M.dimensions[0]/2,M.dimensions[0]/2] ### Angstroms around protein
yrange = [-M.dimensions[1]/2,M.dimensions[1]/2] ### Angstroms around protein 
nbins=(int(M.dimensions[0]),int(M.dimensions[1]))   #### no. of bins in which to discretise data
center=[0,0,0]

##### Define seperate atomgropups for protein, calphas and side chains
Protein = M.select_atoms("protein")
CA = M.select_atoms("name BB")
Side = M.select_atoms("protein and not (name BB)")

####################################################
##### Initialise the system and directories ########
####################################################

### create directory for files/images
dir = 'analysis/'
if not os.path.exists(dir):
            os.makedirs(dir)

#### create the dictionaries to store the data
species={} ##dictionary containing species atomgroups both leaflet
species0={} ##dictionary containing species atomgroups outer leaflet
species1={} ##dictionary containing species atomgroups inner leaflet
speciesPop0={} ##dictionary containing species populations outer leaflet
speciesPop1={} ##dictionary containing species populations inner leaflet
speciesZ0={} ##dictionary containing species average height of protein in proximity of lipids
speciesZ1={} ##dictionary containing species  average height of protein in proximity of lipids
for s in Species:
    species[s] = M.select_atoms("name " + s) ## initializ(s)e the atom group dict arrays
    speciesPop0[s] = numpy.zeros([int(x) for x in nbins]) ## initializ(s)e the dict arrays
    speciesPop1[s] = numpy.zeros([int(x) for x in nbins]) ## initializ(s)e the dict arrays
    speciesZ0[s] = () ### list of minimum corrdinates of each species
    speciesZ1[s] = ()

Species.append("head")
speciesPop0["head"] = numpy.zeros([int(x) for x in nbins]) ## initializ(s)e the dict arrays
speciesPop1["head"] = numpy.zeros([int(x) for x in nbins]) ## initializ(s)e the dict arrays
speciesZ0["head"] = () ### list of minimum corrdinates of each species
speciesZ1["head"] = ()



##### initialize the arrays for the data and histogram
####leaflet 0
thick0 = numpy.zeros([int(x) for x in nbins])   #### bilayer thickness based on minimum PO4 distance outer
thick1 = numpy.zeros([int(x) for x in nbins])   #### bilayer thickness based on minimum PO4 distance  inner
z0 = numpy.zeros([int(x) for x in nbins])   #### bilayer zcoordinate based on PO4 distance outer
z1 = numpy.zeros([int(x) for x in nbins])   #### bilayer zcoordinate based on PO4 distance inner



####################################################
#### functions                              ########
####################################################


def get_dist(atomgroup0, atomgroup1, cutoff = 5):
    """distance array between atomgroups"""
    dist = MDAnalysis.analysis.distances.distance_array(atomgroup0.positions, atomgroup1.positions)
    min0 = numpy.sort(dist,1)[:,:cutoff]
    min1 = numpy.transpose(numpy.sort(dist,0)[:cutoff])
    mean0 = numpy.mean(min0,1)
    mean1 = numpy.mean(min1,1)
    dist0 = numpy.hstack((atomgroup0.positions , mean0.reshape(-1,1) , min0[:,0].reshape(-1,1)))
    dist1 = numpy.hstack((atomgroup1.positions , mean1.reshape(-1,1) , min1[:,0].reshape(-1,1)))
    return dist0, dist1


def get_leaflets():
    """populates species dictionaries with atomgroups separated into upper and lower leaflets based on the z coordinates """
    #### Run the leaflet finder using species[n] (PO4 for coarse grained sims).
    L = MDAnalysis.analysis.leaflet.LeafletFinder(M,"name PO4",cutoff=15)
    ###### Determine which leaflet is upper from the C.O.M 	
    if (L.groups(0).center_of_mass()[2] > L.groups(1).center_of_mass()[2]):
        species0["head"] = L.groups(0) #### upper   
        species1["head"] = L.groups(1) #### lower
    else:
        species0["head"] = L.groups(1) #### upper
        species1["head"] = L.groups(0) #### lower
    #### and create the appropriate atom groups
    for s in Species:
        if s != "head":
            species0[s] = species0["head"].residues.atoms.select_atoms("name " + s)
            species1[s] = species1["head"].residues.atoms.select_atoms("name " + s)

def update_2Dhistogram(coordinates, histogram, center, weights=None):
    """ Updates histograms of species populations """
    temp = coordinates - center
    #print center
    ### special case for z-coordinates....
    if weights == "Z":
        weights = temp[:,2]
        #print weights
    #### bin the results
    histogram = histogram + numpy.histogram2d(temp[:,0], temp[:,1] , bins = nbins , range=[xrange, yrange] , weights = weights)[0]
    return histogram


def update_zrange(atomgroup, zrange, center):
    """Appends the maximum and minimum z coordinates of each species to the appropriate dictionary """
    try:
        closeAtoms = (Protein + atomgroup).select_atoms("protein and around 7 name " + s)
        zrange = numpy.append(zrange , (numpy.min(closeAtoms.positions[:,2] - center[2]) , numpy.max(closeAtoms.positions[:,2] -center[2])))
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        pass
    return zrange

def normalise_histogram(histogram, frequency):
    """Normalises histograms based on species populations/frequencies"""
    zeros = frequency < 1
    histogram = histogram / (frequency + zeros)
    return histogram

def write_z_pdb(pdbfile, leaflet, populations, data, data1):
    """Writes out pdb of z coordinates data"""
    ### dummy histogram
    dummy = numpy.histogram2d([0,0], [0,0] , bins = nbins, range=[xrange, yrange], weights = [0,0])
    #### Correctly formatted 1st part of each line in pdb
    head = "ATOM     01  SS  SUR     1    "
    fname = (dir + pdbfile)
    with open(fname, 'w') as outfile:
        outfile.write("TITLE  bilayer z-coordinates as b-factors\n" )
        outfile.write("COMMENT lower leaflet has  negative z-coordinate as b-factors so coloring is consistent\n" )
    for i in numpy.arange(len(dummy[1])):
        for j in numpy.arange(len(dummy[2])):
            if populations[i-1][j-1] > 3:
                oneline =  head + "%8.3f" *3  %(dummy[1][i] , dummy[2][j] , data[i-1][j-1]) + " %5.2f" *2 %(1.00, data1[i-1][j-1])
                with open(fname, 'a') as outfile:
                    outfile.write(oneline +"\n")

####################################################
##### Loop through the trajectory and       ########
##### collect data                          ########
####################################################
for ts in M.trajectory[start:finish]:
    print (ts)
    #### for the first "leafletts" frames return the atomgroup selections for each leaflet for all species
    if ((ts.frame - start) <= leafletts):
        get_leaflets()

    ##### Calculation of Blayer thickness for each lipid
    ##### and density of NH3, PO4 and GLH     
    dist0, dist1 = get_dist(species0["head"],species1["head"])
    ##### center data around protein
    center = Protein.centroid()
    #center[0] = M.dimensions[0]/2 
    #center[1] = M.dimensions[1]/2
    #center[2] = M.dimensions[2]/2
    #print center
    #### loop through all the species in the dictionary
    ### update population histograms
    for s in Species:
        if len(species0[s]) > 0:
            speciesPop0[s] = update_2Dhistogram(species0[s].positions, speciesPop0[s], center)
            speciesZ0[s] = update_zrange(species0[s], speciesZ0[s], center)
        if len(species1[s]) > 0:       
            speciesPop1[s] = update_2Dhistogram(species1[s].positions, speciesPop1[s], center)
            speciesZ1[s] = update_zrange(species1[s], speciesZ1[s], center)
    ### update thickness histograms
    thick0 = update_2Dhistogram(dist0[:,0:3], thick0, center, weights = dist0[:,4])
    thick1 = update_2Dhistogram(dist1[:,0:3], thick1, center, weights = dist1[:,4])
    ### update zcoordinate histograms
    z0 = update_2Dhistogram(species0["head"].positions, z0, center, weights = "Z")
    z1 = update_2Dhistogram(species1["head"].positions, z1, center, weights = "Z")

####################################################
##### Normalize the histograms and center   ########
##### the z coordinates about the bilayer   ########
##### center                                ########
####################################################
thick0 = normalise_histogram(thick0, speciesPop0["head"])
z0 = normalise_histogram(z0, speciesPop0["head"])
thick1 = normalise_histogram(thick1, speciesPop1["head"])
z1 = normalise_histogram(z1, speciesPop1["head"])

### center z coordinates histograms so the center of the bilayer is at z=0  
meanz =  0.5 * (z0.mean() + z1.mean())
z1 = z1 - meanz
z0 = z0 - meanz

####################################################
##### Write out the pdb zcoordinate data    ########
##### and protein                           ########
####################################################
write_z_pdb("z0xyz.pdb", "upper", speciesPop0["head"], z0, z0 )
write_z_pdb("z1xyz.pdb", "lower", speciesPop1["head"], z1,-z1 ) #### NEEEDS changing....
##### Use the protein from the final frame and translate it relative to bilayer in z0,z1
Protein.positions = (Protein.positions - center - [0,0,meanz])
##### Write out as pdb using MDA writer
Protein.write( dir + "protein.pdb")
