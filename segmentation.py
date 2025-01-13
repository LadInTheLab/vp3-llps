import numpy as np
import pandas as pd
import scipy as sp
import plotly.graph_objects as go
from skimage import measure, filters, segmentation, morphology, draw, feature

def watershed_segment(single_channel_image: np.ndarray, background_maximum: int | float, foreground_minimum: int | float,
                      min_marker_size: int = 5, percentile: int | None = None):
    """
    Perform watershed segmentation on a single channel image.
    Only valid markers (connected components above a certain size) are used for segmentation.
    :param single_channel_image: A NumPy ndarray with shape (X, Y)
    :param background_maximum: An integer or float, the value of the brightest pixel that is certain to be background
    :param foreground_minimum: An integer or float, the value of the dimmest pixel that is certain to be foreground
    :param min_marker_size: The minimum size of a valid marker in pixels (connected component size)
    :param visualize: A boolean indicating whether or not to visualize the segmentation for parameter tuning
    :return: A NumPy ndarray with shape (X, Y) where background pixels are 0, and foreground objects have value 1
    """
    try:
        # Input validation
        if not isinstance(single_channel_image, np.ndarray): # The labeled image must be a NumPy array
            raise InvalidInputError(f"Labeled image must be a NumPy ndarray, not {type(single_channel_image)}.")
        if not isinstance(background_maximum, (int, float)): # The background threshold must be int or float
            raise InvalidInputError(f"Minimum size must be an integer or float, not {type(background_maximum)}.")
        if not isinstance(foreground_minimum, (int, float)): # The foreground threshold must be int or float
            raise InvalidInputError(f"Maximum size must be an integer or float, not {type(foreground_minimum)}.")
        if not isinstance(min_marker_size, int) or min_marker_size <= 0:
            raise InvalidInputError(f"Minimum marker size must be a positive integer, not {min_marker_size}.")
        if single_channel_image.ndim != 2: # The input image must be a 2D array
            raise InvalidInputError(f"Labeled image must be a 2D array, input has shape {single_channel_image.shape}.")

        # Use the sobel filter to get an elevation map
        elevation = filters.sobel(single_channel_image)

        # Get a dynamic foreground value based on percentile in the nonzero pixels
        if percentile is not None:
            clipped_image = np.clip(single_channel_image, 0, 100)
            nonzero_pixels = clipped_image[clipped_image > 0]
            foreground_minimum = np.percentile(nonzero_pixels, percentile)

        # Create a markers image where background pixels have value 1 and foreground has value 2, unknown is 0
        markers = np.zeros_like(single_channel_image, dtype=int)
        markers[single_channel_image < background_maximum] = 1
        markers[single_channel_image > foreground_minimum] = 2

        # Create a copy of the markers array to preserve the original labels
        filtered_markers = markers.copy()

        # Label connected components in the markers image for regions labeled as 2
        labeled_markers, num_labels = measure.label(markers == 2, connectivity=2, return_num=True)

        # Iterate through each labeled region
        for region in measure.regionprops(labeled_markers):
            region_label = region.label
            region_area = region.area
            # Get the pixel indices for the current region
            region_pixels = labeled_markers == region_label

            # Only remove regions labeled with 2 and smaller than min_marker_size
            if region_area < min_marker_size:
                # Set the pixels of the small region back to 0 (background)
                filtered_markers[region_pixels] = 0

        markers = filtered_markers

        # Perform watershed segmentation
        watershed_segmented_image = segmentation.watershed(image=elevation, markers=markers)

        # Fill holes
        holes_filled_image = sp.ndimage.binary_fill_holes(watershed_segmented_image - 1)

        # Remove noise
        denoised_image = sp.ndimage.binary_opening(holes_filled_image)

        return denoised_image.astype(int)

    except InvalidInputError as e:
        print(f"Invalid input error: {e}")
        return None


def watershed_distance_label(segmented_image: np.ndarray, min_distance: int = 0, visualize: bool = False):
    """
    Use the distance transform and watershed algorithm to label a segmented image, or relabel a labeled image. Useful
    for separating connected objects and is most effective for larger objects.
    :param segmented_image: A NumPy array representing a segmented image. Must be an array of integers, zero value
    pixels are treated as background.
    :param min_distance: Optional, minimum distance between markers - increase this value if single objects are being
    divided into multiple parts.
    :param visualize: Optional, if True will display a visual representation of the segmented image.
    :return: A NumPy array representing a segmented image, where different objects have different integer values.
    """
    try:
        # Input validation
        if not isinstance(segmented_image, np.ndarray):
            raise InvalidInputError(f"Input image must be a NumPy ndarray, not {type(segmented_image)}.")
        if not issubclass(segmented_image.dtype.type, np.integer):  # The label image must be an array of integers.
            raise InvalidInputError("The segmented image must be an array of integers.")

        # Make the image binary if it isn't already - replace all labeled pixels with 1, leave background as 0from
        label_mask = np.where(segmented_image == 0, 0, 1)

        # Calculate a distance transform for the array
        distance_transform = sp.ndimage.distance_transform_edt(label_mask)

        # Find the peak local maxima of the image, with min_distance enforcing separation
        markers = feature.peak_local_max(distance_transform, min_distance=min_distance)

        marker_mask = np.zeros_like(segmented_image)
        marker_idx = 0
        for mark in markers:
            x, y = mark[0], mark[1]
            marker_mask[x, y] = 2 + marker_idx
            marker_idx += 1

        # Perform watershed segmentation using the inverse distance transform to define basins
        watershed_seg = segmentation.watershed(-distance_transform, marker_mask, mask=segmented_image)

        if visualize:
            fig = go.Figure()
            fig.add_trace(go.Heatmap(z=watershed_seg))
            fig.show()


        return watershed_seg

    except InvalidInputError as e:
        print(f"Invalid input error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def measure_objects(segmented_image: np.ndarray, intensity_image: np.ndarray, properties: list | tuple = None):
    """
    Measure the objects in a segmented image and convert the measurements to a dataframe
    :param segmented_image: A segmented 2D integer array where pixels with the same value are part of the same object,
    and pixels with value 0 are treated as background. Must have the same X, Y shape as intensity_image.
    :param intensity_image: A NumPy array representing the intensity of the objects - the original image. Must have the
    same X, Y shape as segmented_image
    :param properties: Optional, a list or tuple of properties to measure. Default is to measure all properties.
    :return: A Pandas DataFrame containing the measured objects as rows.
    """
    try:
        # Input validation
        if not isinstance(segmented_image, np.ndarray): # Segmentation image must be a numpy array
            raise InvalidInputError("The segmented image must be a NumPy array")
        if not issubclass(segmented_image.dtype.type, np.integer):  # The label image must be an array of integers.
            raise InvalidInputError("The segmented image must be an array of integers.")
        if not isinstance(intensity_image, np.ndarray): # Intensity image must be a numpy array
            raise InvalidInputError("The intensity image must be a NumPy array")
        if segmented_image.ndim != 2: # Segmentation images are 2D arrays
            raise InvalidInputError("The segmented image must be a 2D array")
        if (segmented_image.shape[0], segmented_image.shape[1]) != (intensity_image.shape[0], intensity_image.shape[1]):
            raise InvalidInputError("The intensity and segmented images must have the same X, Y shape.")
        if properties is not None: # If properties are specified they must be list or tuple
            if not isinstance(properties, (list, tuple)):
                raise InvalidInputError("The properties must be a list or tuple")

        # List of properties to measure
        if properties is None:
            properties = ['label', 'area', 'area_bbox', 'area_convex', 'area_filled', 'axis_major_length',
                          'axis_minor_length', 'bbox', 'centroid', 'centroid_local', 'coords_scaled', 'coords',
                          'eccentricity', 'equivalent_diameter_area', 'image', 'image_convex', 'image_filled',
                          'intensity_max', 'intensity_mean', 'intensity_min', 'intensity_std', 'moments', 'num_pixels',
                          'orientation', 'perimeter', 'perimeter_crofton','slice', 'solidity']

        # Measure region properties
        object_properties_dict = measure.regionprops_table(label_image=segmented_image, intensity_image=intensity_image,
                                                           properties=properties)

        # Convert the region properties to a DataFrame
        object_properties_df = pd.DataFrame(object_properties_dict)

        return object_properties_df

    except InvalidInputError as e:
        print(f"Invalid input error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None