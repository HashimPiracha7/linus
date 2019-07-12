import numpy as np
import crystal_graph as graph
import sph_projection_utils as sph
import symmetry_finding as sf
import plotly
import pymatgen
import random
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.groups import SpaceGroup,SYMM_DATA , sg_symbol_from_int_number
import plotly
from plotly.graph_objs import *
plotly.offline.init_notebook_mode(connected=False)
import matplotlib.pyplot as plt

def get_coords_of_sites_in_sphere(structure, center_site, radial_cutoff=5., center_on_site=True, include_center=False):
	# This function takes care of figuring out which image 
	# (copy across periodic boundaries) of a given atom 
	# is within the sphere.
	atoms_in_sphere = structure.get_sites_in_sphere(center_site.coords, radial_cutoff, include_index=True, include_image=True)

	atom_sites, distances, indices, images = zip(*atoms_in_sphere)

	if include_center:
		cart_coords = np.array([atom.coords for atom in atom_sites])
	else:
		cart_coords = np.array([atom.coords for atom in atom_sites if atom != center_site])

	if center_on_site:
		cart_coords -= center_site.coords.reshape(-1, 3)

	return cart_coords


def central_atom_positions(unit_cell, element):
    positions = []
    for counter in range(len(unit_cell["structure"])):
        if element in unit_cell["structure"][counter]:
            positions.append(unit_cell["structure"][counter])
    return positions


def get_sph(unit_cell, element= "Si"):
    print(unit_cell["material_id"])
    central_atom = central_atom_positions(unit_cell,element)
    coordiates_xyz =[get_coords_of_sites_in_sphere(unit_cell["structure"],i,radial_cutoff=2) for i in central_atom]
    angular = [sph.xyz_to_phi_theta(i) for i in coordiates_xyz]
    phi = [x for x,y in angular]
    theta = [y for x,y in angular]
    spherical_harmonic = [sph.get_Ylm_coeffs(phi[i],theta[i],sum = True) for i in range(len(angular))]
    return spherical_harmonic

def norm(arr):
    if type(arr) == list:
        return[sf.norm_sph(i.reshape(1, -1))[0] for i in arr]
    else:
        return sf.norm_sph(arr)

tp = graph.TensorProduct(5, 5, 5)

def tensor_product_set_1(sph, index):
    # set 1 is when a single atom's sph is Tensor producted to all the other central atoms
    temporary = sph[:index]+ sph[index+1:]
    tensor_products= [tp.compute(sph[index],i) for i in temporary] 
    return tensor_products

def vector_from_central_atom(unit_cell, central_element, atom_index):
	positions = central_atom_positions(unit_cell, central_element)
	temporary = positions[:atom_index] +positions[atom_index+1:]
	return [(positions[atom_index].coords - i.coords) for i in temporary]


def vector_from_central_atom_sph(unit_cell, element, central_atom_index):
    vectors = vector_from_central_atom(unit_cell, element, central_atom_index)
    angular_vectors = [sph.xyz_to_phi_theta(i) for i in vectors]
    phi = [x for x,y in angular_vectors]
    theta = [y for x,y in angular_vectors]
    spherical_harmonic = [sph.get_Ylm_coeffs(phi[i],theta[i],sum = False) for i in range(len(angular_vectors))]
    return spherical_harmonic

def tensor_product_set_2(sph, unit_cell,element, central_atom_index):
    #Set 2 is atom producted with the vector to other atoms
    unit_cell_sph = sph
    vectos_sph = vector_from_central_atom_sph(unit_cell, element, central_atom_index)
    tensor_product = [tp.compute(unit_cell_sph[central_atom_index],i) for i in vectos_sph]
    return tensor_product

def vector_to_central_atom(unit_cell,central_element, atom_index):
	positions = central_atom_positions(unit_cell, central_element)
	temporary = positions[:atom_index] +positions[atom_index+1:]
	return [(i.coords- positions[atom_index].coords) for i in temporary]

def vector_to_central_atom_sph(unit_cell, central_element,central_atom_index):
    vectors = vector_to_central_atom(unit_cell,central_element, central_atom_index)
    angular_vectors = [sph.xyz_to_phi_theta(i) for i in vectors]
    phi = [x for x,y in angular_vectors]
    theta = [y for x,y in angular_vectors]
    spherical_harmonic = [sph.get_Ylm_coeffs(phi[i],theta[i],sum = False) for i in range(len(angular_vectors))]
    return spherical_harmonic

def tensor_product_set_3(sph, unit_cell, central_element, central_atom_index):
	unit_cell_sph = sph
	vectos_sph = vector_to_central_atom_sph(unit_cell,central_element, central_atom_index)
	tensor_product = [tp.compute(unit_cell_sph[central_atom_index],i) for i in vectos_sph]
	return tensor_product

def tensor_product_set_4(sph, unit_cell, central_element, central_atom_index):
	tensor_product_from = tensor_product_set_2(sph, unit_cell,central_element, central_atom_index)
	tensor_product_to= tensor_product_set_3(sph, unit_cell, central_element, central_atom_index)
	return [tp.compute(tensor_product_to[i],tensor_product_from[i]) for i in range(len(tensor_product_from))]




    





