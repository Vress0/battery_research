import os
from datetime import datetime
from chgnet.model import CHGNet
from chgnet.model.dynamics import CHGNetCalculator
from pymatgen.core import Structure
from ase import Atoms
from ase.optimize import FIRE
from poscar_utils import smart_load_poscar

# 建立輸出資料夾
job_id = os.environ.get('SLURM_JOB_ID', datetime.now().strftime('%Y%m%d_%H%M%S'))
output_dir = f"results_{job_id}"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

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
chgnet.to("cuda")
print("CHGNet model is running on GPU.")

from chgnet.model import StructOptimizer
relaxer = StructOptimizer()

# Predict energy, force, stress, magmom
structure = smart_load_poscar(path)
print(structure)
prediction = chgnet.predict_structure(structure)
print('unrelax magmom\n', prediction)

# 將預測結果寫入輸出資料夾
with open(os.path.join(output_dir, "prediction_results.txt"), "w") as f:
    f.write('Unrelaxed magmom\n')
    f.write(str(prediction) + '\n\n')
    for key, unit in [
        ("energy", "eV/atom"),
        ("forces", "eV/A"),
        ("stress", "GPa"),
        ("magmom", "mu_B"),
    ]:
        result = f"CHGNet-predicted {key} ({unit}):\n{prediction[key[0]]}\n"
        print(result)
        f.write(result + '\n')

# Structure Optimization
positions = structure.cart_coords 
numbers = [site.specie.number for site in structure]  
cell = structure.lattice.matrix  
atoms = Atoms(positions=positions, numbers=numbers, cell=cell, pbc=True)

calculator = CHGNetCalculator(model=chgnet)
atoms.calc = calculator  

# 將 trajectory 和輸出檔案路徑指定到輸出資料夾
optimizer = FIRE(atoms, trajectory=os.path.join(output_dir, "FIRE.traj"))
optimizer.run(fmax=0.05, steps=1000)

# 將鬆弛後的結構儲存到輸出資料夾
atoms.write(os.path.join(output_dir, "relaxed_fire_POSCAR"), format="vasp")
print(f"Fire relaxation completed. Structure saved to {output_dir}/relaxed_fire_POSCAR.")

# MD
from chgnet.model.dynamics import MolecularDynamics
import time

supercell = structure.make_supercell([2, 2, 2], in_place=False)
print(supercell.composition)

# 將 MD 輸出檔案路徑指定到輸出資料夾
md = MolecularDynamics(
    atoms=supercell,
    model=chgnet,
    ensemble="nvt",
    temperature=600,  # in K
    timestep=1,  # in fs
    trajectory=os.path.join(output_dir, "md_out.traj"),
    logfile=os.path.join(output_dir, "md_out.log"),
    loginterval=10,
)

# Start timing before MD starts
start_time = time.time()
md.run(60000)  # run a 60 ps MD simulation
# End timing after MD ends
end_time = time.time()

# Compute total runtime
total_time = end_time - start_time
print(f"MD simulation completed in {total_time:.2f} seconds.")

with open(os.path.join(output_dir, "md_out.log"), "a") as log_file:
    log_file.write(f"\nMD simulation completed in {total_time:.2f} seconds.\n")

print(f"\nAll output files have been saved to: {output_dir}")
