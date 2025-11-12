import os

from chgnet.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from pymatgen.core import Structure
from ase import Atoms
from ase.optimize import FIRE

def find_poscar_file():
    for file in os.listdir("."):
        if file == "POSCAR":
            return file
    raise FileNotFoundError("No POSCAR file found in the current directory.")

try:
    poscar_file = find_poscar_file()
    print(f"Found POSCAR file: {poscar_file}")
except FileNotFoundError as e:
    print(e)
    exit(1)

path = os.path.join(os.getcwd(), poscar_file)
print(f"POSCAR file path: {path}")


chgnet = CHGNet.load()
chgnet.to("cpu")
print("CHGNet model is running on CPU.")


from chgnet.model import StructOptimizer
relaxer = StructOptimizer() #fmax
#Predict energy, force, stress, magmom
structure = Structure.from_file(path)
print(structure)
prediction = chgnet.predict_structure(structure)
print('unrelax magmom\n' ,prediction)

for key, unit in [
    ("energy", "eV/atom"),
    ("forces", "eV/A"),
    ("stress", "GPa"),
    ("magmom", "mu_B"),
]:
    print(f"CHGNet-predicted {key} ({unit}):\n{prediction[key[0]]}\n")
#Structure Optimization
#structure.perturb(0.1)# Perturb the structure


positions = structure.cart_coords 
numbers = [site.specie.number for site in structure]  
cell = structure.lattice.matrix  

atoms = Atoms(positions=positions, numbers=numbers, cell=cell, pbc=True)

calculator = CHGNetCalculator(model=chgnet)
atoms.calc = calculator  

optimizer = FIRE(atoms, trajectory="FIRE.traj")
optimizer.run(fmax=0.05, steps=200)

atoms.write("relaxed_fire_POSCAR", format="vasp")
print("Fire relaxation completed. Structure saved to relaxed_POSCAR.")

