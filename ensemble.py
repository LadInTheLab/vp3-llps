import numpy as np


def write_ensemble(trajectory_array, num_timepoints: int, output_path, atom_names=None, residue_names=None, residue_numbers=None,
                   chain_ids=None):
    """
    Write an ensemble PDB file for atom positions at selected timesteps.

    Parameters:
    :param trajectory_array: numpy array of shape (K, N, 3), positions of N atoms over K timesteps.
    :param num_timepoints: int, the number of timepoints to extract
    :param output_path: str, the path to write the ensemble PDB file.
    :param atom_names: list of str, names of the atoms, length N (default to "ATOM").
    :param residue_names: list of str, names of the residues, length N (default to "RES").
    :param residue_numbers: List of int, the indices of the residue in its respective chain
    :param chain_ids: list of str, the ID of the chain that each residue belongs to
    """
    trajectory_array = trajectory_array.copy()
    N, K, _ = trajectory_array.shape
    # Extract X evenly spaced timepoints
    indices = np.linspace(0, K - 1, num_timepoints, dtype=int)
    trajectory_array = trajectory_array[:, indices, :].transpose(1, 0, 2)  # Shape (K, N, 3)


    K, N, _ = trajectory_array.shape
    if atom_names is None:
        atom_names = ["ATOM"] * N
    if residue_names is None:
        residue_names = ["RES"] * N
    if residue_numbers is None:
        residue_numbers = ["1"] * N
    if chain_ids is None:
        chain_ids = ["A"] * N  # Default to a single chain

    with open(output_path, "w") as pdb:
        for model_idx, structure in enumerate(trajectory_array, start=1):
            pdb.write(f"MODEL     {model_idx}\n")
            for atom_idx, (position, atom_name, residue_name, residue_number, chain_id) in enumerate(
                    zip(structure, atom_names, residue_names, residue_numbers, chain_ids), start=1
            ):
                x, y, z = position
                pdb.write(
                    f"ATOM  {atom_idx:5d} {atom_name:<4} {residue_name:<3} {chain_id:>1}   1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
                )
            pdb.write("ENDMDL\n")
    print(f"Ensemble PDB written to {output_path}")