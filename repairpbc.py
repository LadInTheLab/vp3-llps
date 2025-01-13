import time
import numpy as np


def repair_pbc(selected_trajectory, box_dimensions, verbose=True):
    """
    'Lasciate ogni speranza, voi ch'entrate'

    This function is another that involves a great deal of manipulation of multidimensional arrays, and
    as such the code may be hard to follow. The naive approach to correcting periodic boundary conditions is to
    start with one atom as ground truth, and move down the chain, checking each inter-atomic distance and comparing
    it to a threshold value (based on some reasonable maximum possible inter-atomic distance), then attempting to
    correct each in sequence. This approach is slow, as it requires iteration over all atoms and all frames (easily
    millions of iterations for a short, small simulation). The array-based technique is conceptually similar, but is
    far more efficient (approx 70x faster in my testing) because it takes advantage of the fact that periodic
    boundary condition 'jumps' are always 0.5n multiples of box dimensions for a cubic box, so all displacements and
    all 'reasonable' corrections (including no change) can be tested simultaneously by vector operations, and array
    manipulations can allow the 'best' correction for each atom to be computed for all atoms and all frames at once.
    Each conceptual step is outlined in code comments, but following the actual computation requires familiarity
    with NumPy array manipulations and vectorized operations. It's worth noting that parts of this code
    *could* be achieved with 4 dimensional arrays instead of multiple 3-dimensional arrays, but in my testing this
    didn't yield significant enough improvements to computing time or memory usage to justify the less-readable code.
    """

    # Get the allowed offsets from the dimensions of the water box
    offsets = box_dimensions
    box_center = offsets / 2
    allowed_multipliers = np.arange(0, 5) * 0.5

    start_time = time.perf_counter()
    if verbose:
        print(f"START: Correcting periodic boundary conditions...")

    # Calculate allowed corrections, which are always multiples of 0.5 times a box dimension
    allowed_corrections_X = allowed_multipliers * offsets[0]
    allowed_corrections_Y = allowed_multipliers * offsets[1]
    allowed_corrections_Z = allowed_multipliers * offsets[2]

    # Shift the trajectory 'up' by one to allow calculation of displacement in all axes
    trajectory_upshifted = np.roll(selected_trajectory, shift=-1, axis=0)
    fwd_displacements = trajectory_upshifted - selected_trajectory

    # Find the sign of the corrections to be made, the opposite of the sign of the displacement
    fwd_displacements_signs = np.sign(fwd_displacements)
    fwd_correction_signs = fwd_displacements_signs * -1

    # Broadcast the allowed corrections arrays to match the shape of the data
    allowed_corrections_X_broadcast = np.tile(allowed_corrections_X,
                                              (fwd_displacements.shape[0], fwd_displacements.shape[1], 1))
    allowed_corrections_Y_broadcast = np.tile(allowed_corrections_Y,
                                              (fwd_displacements.shape[0], fwd_displacements.shape[1], 1))
    allowed_corrections_Z_broadcast = np.tile(allowed_corrections_Z,
                                              (fwd_displacements.shape[0], fwd_displacements.shape[1], 1))

    # Sign the allowed corrections
    allowed_corrections_X_signed = allowed_corrections_X_broadcast * fwd_correction_signs[:, :, 0][:, :, np.newaxis]
    allowed_corrections_Y_signed = allowed_corrections_Y_broadcast * fwd_correction_signs[:, :, 1][:, :, np.newaxis]
    allowed_corrections_Z_signed = allowed_corrections_Z_broadcast * fwd_correction_signs[:, :, 2][:, :, np.newaxis]

    # Apply the corrections to the displacements to each axis individually
    corrected_disp_X = fwd_displacements[:, :, 0][:, :, np.newaxis] + allowed_corrections_X_signed
    corrected_disp_Y = fwd_displacements[:, :, 1][:, :, np.newaxis] + allowed_corrections_Y_signed
    corrected_disp_Z = fwd_displacements[:, :, 2][:, :, np.newaxis] + allowed_corrections_Z_signed

    # Find the index of the minimum displacements
    idx_min_displacement_X = np.argmin(np.abs(corrected_disp_X), axis=2)
    idx_min_displacement_Y = np.argmin(np.abs(corrected_disp_Y), axis=2)
    idx_min_displacement_Z = np.argmin(np.abs(corrected_disp_Z), axis=2)

    # Build 3 2-D arrays with the same dimensions as axes 0 and 1 of the trajectory, containing the best signed disp
    best_X_corrections = allowed_corrections_X_signed[
        np.arange(allowed_corrections_X_signed.shape[0])[:, None],
        np.arange(allowed_corrections_X_signed.shape[1]), idx_min_displacement_X]
    best_Y_corrections = allowed_corrections_Y_signed[
        np.arange(allowed_corrections_Y_signed.shape[0])[:, None],
        np.arange(allowed_corrections_Y_signed.shape[1]), idx_min_displacement_Y]
    best_Z_corrections = allowed_corrections_Z_signed[
        np.arange(allowed_corrections_Z_signed.shape[0])[:, None],
        np.arange(allowed_corrections_Z_signed.shape[1]), idx_min_displacement_Z]

    # Merge these arrays into a 3D array
    best_corrections_XYZ = np.stack((best_X_corrections, best_Y_corrections, best_Z_corrections), axis=2)

    # Shift the corrections "down" to align them with the atoms they'll be correcting, correct first atom to origin
    best_corrections_shifted = np.roll(best_corrections_XYZ, shift=1, axis=0)

    # Calculate a running sum of the corrections to be made
    cumulative_corrections = np.cumsum(best_corrections_shifted, axis=0)

    # Calculate corrected positions
    corrected_trajectory = selected_trajectory + cumulative_corrections

    # Find the centroid for each frame, and make sure it's centered in the box
    centroids = np.mean(corrected_trajectory, axis=0)
    centroids_shift = box_center - centroids
    centroids_shift_broadcast = np.broadcast_to(centroids_shift, corrected_trajectory.shape)
    corrected_trajectory = corrected_trajectory + centroids_shift_broadcast

    end_time = time.perf_counter()
    time_elapsed = np.round((end_time - start_time), decimals=2)
    total_points = (corrected_trajectory.shape[0] * corrected_trajectory.shape[1] *
                    corrected_trajectory.shape[2]) + corrected_trajectory.shape[1]
    if verbose:
        print(
            f"COMPLETE: PBC correction. ({('{:,}'.format(total_points))} inter-atomic distances checked in {time_elapsed} seconds)")

    return corrected_trajectory
