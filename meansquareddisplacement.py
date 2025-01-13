import numpy as np
from scipy.stats import mode


def calculate_msd(structure_topology_df, corrected_trajectory_array, method='mean', verbose=True, reference_index=0,
                  **kwargs):
    if verbose:
        print(f"START: Calculating mean squared displacement...")

    # Get the indices of the backbone atoms with which we calculate the dihedrals
    calpha_indices = list(structure_topology_df.index[structure_topology_df['TOPOLOGY_Atom_Name'].isin(['CA'])])

    # Select the backbone atoms from the trajectory
    trajectory_array = corrected_trajectory_array[calpha_indices]

    if method == 'mean':
        mean_position = np.mean(trajectory_array, axis=1)
        msd = np.sum((trajectory_array - mean_position[:, np.newaxis, :]) ** 2, axis=(1, 2)) / trajectory_array.shape[1]
    elif method == 'reference':
        reference_pos = trajectory_array[:, reference_index, :]
        msd = (np.sum((trajectory_array - reference_pos[:, np.newaxis, :]) ** 2, axis=(1, 2)) /
               trajectory_array.shape[1])
    elif method == 'median':
        reference_pos = np.median(trajectory_array, axis=1)
        msd = np.sum((trajectory_array - reference_pos[:, np.newaxis, :]) ** 2, axis=(1, 2)) / trajectory_array.shape[1]
    elif method == 'mode':
        reference_pos, _ = mode(trajectory_array, axis=1)
        msd = np.sum((trajectory_array - reference_pos[:, np.newaxis, :]) ** 2, axis=(1, 2)) / trajectory_array.shape[1]
    elif method == 'timeresolved':
        def _calculate_tr_msd(trajectory_array, reference_pos):
            # Calculate differences between each frame and the reference frame
            diff = trajectory_array - reference_pos[:, np.newaxis, :]  # Shape: (atoms, frames, dimensions)
            # Compute squared differences
            diff_squared = np.sum(diff ** 2, axis=2)  # Shape: (atoms, frames)
            # Average over atoms and take square root
            msd = np.mean(diff_squared, axis=0)  # Shape: (frames,)

            return msd

        reference_pos = trajectory_array[:, 0, :]
        if 'ranges' in kwargs.keys():
            ranges = kwargs['ranges']
            msd_list = []
            for atomrange in ranges:
                sub_ref = reference_pos[atomrange[0]:atomrange[1], :]
                sub_trajectory = trajectory_array[atomrange[0]:atomrange[1], :, :]
                sub_msd = _calculate_tr_msd(sub_trajectory, sub_ref)
                msd_list.append(sub_msd)
            msd = msd_list
        else:
            msd = _calculate_tr_msd(trajectory_array, reference_pos)


    else:
        raise ValueError(f"method '{method}' is not valid, valid options are 'reference', 'mean', 'median', 'mode'.")

    if verbose:
        print(f"COMPLETE: Mean squared displacement calculated.")

    return msd