import pandas as pd
import numpy as np
import os

from scipy.optimize import least_squares
from scipy.stats import sem
from scipy.stats import t

import plotly.graph_objects as go



def fit_single_term(params, x, y):
    # The single exponential equation for FRAP recovery
    i0, a, b = params
    return i0 - a * np.exp(-b * x) - y


def fit_double_term(params, x, y):
    # The double exponential equation for FRAP recovery
    i0, a, b, l, r = params
    return (i0 - a * np.exp(-b * x)) - (l * np.exp(-r * x)) - y


def timestamp_dataframe(csv_path: str, index_scalar=1.0, roi_index=1, whole_obj_index=2, bg_index=3, timecode_index=None):
    # Convert time values to a standardized timecode - the 'index_scalar' parameter allows for indices or frames to
    # be converted to a meaningful quantity like seconds by applying a conversion factor.
    raw_dataframe = pd.read_csv(csv_path)
    timestamped_df = raw_dataframe
    if timecode_index is not None:
        timestamped_df['TIMESTAMP'] = raw_dataframe.loc[:, timecode_index]
    else:
        timestamped_df['TIMESTAMP'] = raw_dataframe.index * index_scalar
    timestamped_df['ROI_1'] = raw_dataframe.iloc[:, roi_index]
    timestamped_df['WHOLE'] = raw_dataframe.iloc[:, whole_obj_index]
    timestamped_df['BACKGROUND'] = raw_dataframe.iloc[:, bg_index]
    timestamped_df = timestamped_df[['TIMESTAMP', 'ROI_1', 'WHOLE', 'BACKGROUND']]
    return timestamped_df


def bg_correct(timestamped_dataframe):
    # Correct for the background fluorescence by subtracting it from the ROI and Whole values
    timestamped_dataframe['ROI_1_bgcorrected'] = timestamped_dataframe['ROI_1'] - timestamped_dataframe['BACKGROUND']
    timestamped_dataframe['WHOLE_bgcorrected'] = timestamped_dataframe['WHOLE'] - timestamped_dataframe['BACKGROUND']
    bgcorrected_dataframe = timestamped_dataframe[['TIMESTAMP', 'ROI_1_bgcorrected', 'WHOLE_bgcorrected']]
    return bgcorrected_dataframe


def calculate_depth(timestamped_dataframe, prebleach_frames: int, bleach_frames=1):
    prebleach_mean = np.mean(timestamped_dataframe['ROI_1_bgcorrected'][0: prebleach_frames - 1])
    bleach_intensity = timestamped_dataframe['ROI_1_bgcorrected'][((prebleach_frames + bleach_frames) - 1)]
    bleach_depth = (prebleach_mean - bleach_intensity) / prebleach_mean
    return bleach_depth


def normalize(timestamped_dataframe, prebleach_frames: int, bleach_frames=1):
    # Normalize the data by the double method
    roi_prebleach_mean = np.mean(timestamped_dataframe['ROI_1_bgcorrected'][0: prebleach_frames - 1])
    whole_mean_prebleach = np.mean(timestamped_dataframe['WHOLE_bgcorrected'][0: prebleach_frames - 1])
    first_term = whole_mean_prebleach / timestamped_dataframe['WHOLE_bgcorrected']
    second_term = timestamped_dataframe['ROI_1_bgcorrected'] / roi_prebleach_mean
    double_normalized = first_term * second_term
    postbleach_index = (prebleach_frames + bleach_frames - 1)
    postbleach_intensity = double_normalized[postbleach_index]
    # Modifies the double normalized values to give the fullscale normalized values (generally preferred)
    fullscale_normalized = ((double_normalized - postbleach_intensity) / (1 - postbleach_intensity))

    double_normalized = double_normalized.to_frame()
    fullscale_normalized = fullscale_normalized.to_frame()

    double_normalized['TIMESTAMP'] = timestamped_dataframe['TIMESTAMP']
    fullscale_normalized['TIMESTAMP'] = timestamped_dataframe['TIMESTAMP']

    double_normalized.columns = ['DATA', 'TIMESTAMP']
    fullscale_normalized.columns = ['DATA', 'TIMESTAMP']

    return double_normalized, fullscale_normalized


def process_frapdata(normal_method: str, csv_path, prebleach_frames, bleach_frames, index_scalar, roi_index=1,
                     whole_obj_index=2, bg_index=3, timecode_index=None):
    timestamped_dataframe = timestamp_dataframe(csv_path, index_scalar, roi_index, whole_obj_index, bg_index,
                                                timecode_index)
    corrected = bg_correct(timestamped_dataframe)
    depth = calculate_depth(corrected, prebleach_frames, bleach_frames)
    double_norm, fullscale_norm = normalize(corrected, prebleach_frames, bleach_frames)
    if normal_method == 'double':
        return double_norm, depth
    elif normal_method == 'fullscale':
        return fullscale_norm, depth
    else:
        print(f"Unknown method: {normal_method}")


def fit_curve(normalized_data: pd.DataFrame, equation_type: str, prebleach_frames: int):
    if 'TIMESTAMP' not in normalized_data.columns:
        print("Data must be timestamped and normalized. Use 'process_frapdata(args)' to do this automatically.")
        return
    else:
        # Establish initial guesses for curve parameters, thank you EasyFRAP!
        normalized_data = normalized_data[prebleach_frames + 1:]

        x_values = normalized_data['TIMESTAMP']
        y_values = normalized_data['DATA']
        initial_guess = [0.85, 0.5, 0.563]
        initial_guess_double = [0.85, 0.5, 0.563, 0.316, 0.36]

        # Use the least squares method to optimize curve parameters
        res_lsq = None
        if equation_type == 'single':
            res_lsq = least_squares(fit_single_term, initial_guess, args=(x_values, y_values))
            residuals = y_values - fit_single_term(res_lsq.x, x_values, 0)
        elif equation_type == 'double':
            res_lsq = least_squares(fit_double_term, initial_guess_double, args=(x_values, y_values))
            residuals = y_values - fit_double_term(res_lsq.x, x_values, 0)
        else:
            print('Unknown equation type')
            return

        # Calculate R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        curve_params = res_lsq.x
        mobile_fraction = curve_params[0]
        if mobile_fraction < 0 or mobile_fraction > 1:
            mobile_fraction = 'Bad Fit'

        # Calculate the recovery half life
        t_half = np.log(2) / curve_params[2]
        if t_half < 0:
            t_half = 'Bad Fit'

        fit_results = {'Curve Parameters: ': curve_params, 'R Squared: ': r_squared,
                       'Mobile Fraction: ': mobile_fraction, 'Half Life': t_half}

        return fit_results
class FrapExperiment:
    def __init__(self, csv_path_list: list, prebleach_frames: int, bleach_frames: int, index_scalar: float, roi_index=1,
                 whole_obj_index=2, bg_index=3, timecode_index=None, normal_method='fullscale', fit_method='double'):
        self.csv_path_list = csv_path_list
        self.prebleach_frames = prebleach_frames
        self.bleach_frames = bleach_frames
        self.index_scalar = index_scalar
        self.roi_index = roi_index
        self.whole_obj_index = whole_obj_index
        self.bg_index = bg_index
        self.timecode_index = timecode_index
        self.normal_method = normal_method

        self.frap_trials = []
        self.fit_method = fit_method
        self._prepare_trials()
        self.merged_array = self._get_merged_normalized()
        self.mean_intensity = np.mean(self.merged_array, axis=1)
        self.timestamps = list(getattr(self, 'Trial_0').normalized_data['TIMESTAMP'])
        self.fit_params = self._experiment_fit()

        self.trial_fits = self._trial_fits()

    def plot_mean(self, errorbars=True, show_fit=True, ci95=False, ci99=False, marker_color=None, marker_size=None,
                  fit_color=None, ci_color=None, data_mode='markers', figure_template='plotly', bgcolor=None,
                  fit_thickness=None):
        fig = go.Figure()
        fig_elements = []
        if errorbars:
            error_values = sem(self.merged_array, axis=1)
            data_trace = go.Scatter(x=self.timestamps, y=self.merged_array[:, 0],
                                    error_y=dict(type='data', array=error_values, thickness=1),
                                    mode=data_mode, name='Mean Data', marker=dict(color=marker_color, size=marker_size))
        else:
            data_trace = go.Scatter(x=self.timestamps, y=self.mean_intensity, mode=data_mode, name='Mean Data',
                                    marker=dict(color=marker_color, size=marker_size))
        fig_elements.append(data_trace)

        if show_fit:
            curve, fit_name, fit_type = None, None, None
            r_squared = np.round(self.fit_params['R Squared: '], 2)
            mobile_fraction = np.round(self.fit_params['Mobile Fraction: '], 3)
            t_half = np.round(self.fit_params['Half Life'], 3)
            if self.fit_method == 'single':
                curve = fit_single_term(self.fit_params['Curve Parameters: '], np.array(self.timestamps), 0)
                fit_type = 'Single Exponential Fit'
                if ci95 or ci99:
                    if ci_color is None:
                        ci_color = fit_color
                    residuals = self.mean_intensity - curve
                    std_dev_residuals = np.std(residuals, ddof=1)
                    t_value = None
                    name = None
                    if ci95:
                        t_value = t.ppf(0.975, df=len(self.mean_intensity) - 1)
                        name = '95% Confidence Interval'
                    elif ci99:
                        t_value = t.ppf(0.995, df=len(self.mean_intensity) - 1)
                        name = '99% Confidence Interval'
                    ci_width = t_value * std_dev_residuals / np.sqrt(len(self.mean_intensity))
                    upper_bound = curve + ci_width
                    lower_bound = curve - ci_width
                    fig_elements.append(go.Scatter(x=np.concatenate([self.timestamps, self.timestamps[::-1]]),
                                                   y=np.concatenate([upper_bound, lower_bound[::-1]]),
                                                   name=name,
                                                   fill='toself',
                                                   line=dict(color=ci_color)
                                                   ))
            elif self.fit_method == 'double':
                curve = fit_double_term(self.fit_params['Curve Parameters: '], np.array(self.timestamps), 0)
                fit_type = 'Double Exponential Fit'
                if ci95 or ci99:
                    if ci_color is None:
                        ci_color = fit_color
                    residuals = self.mean_intensity - curve
                    std_dev_residuals = np.std(residuals, ddof=1)
                    t_value = None
                    name = None
                    if ci95:
                        t_value = t.ppf(0.975, df=len(self.mean_intensity) - 1)
                        name = '95% Confidence Interval'
                    elif ci99:
                        t_value = t.ppf(0.995, df=len(self.mean_intensity) - 1)
                        name = '99% Confidence Interval'
                    ci_width = t_value * std_dev_residuals / np.sqrt(len(self.mean_intensity))
                    upper_bound = curve + ci_width
                    lower_bound = curve - ci_width
                    fig_elements.append(go.Scatter(x=np.concatenate([self.timestamps, self.timestamps[::-1]]),
                                                   y=np.concatenate([upper_bound, lower_bound[::-1]]),
                                                   name=name,
                                                   fill='toself',
                                                   line=dict(color=ci_color)
                                                   ))

            curve_trace = go.Scatter(x=self.timestamps, y=curve, mode='lines', name=fit_type,
                                     marker=dict(color=fit_color), line=dict(width=fit_thickness))
            fig_elements.append(curve_trace)
        layout = go.Layout(
            title=f'Mean FRAP Data',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Fluorescence Intensity', range=[0, 1.1], autorange=False),
            template=figure_template,
            plot_bgcolor=bgcolor
        )

        fig.layout = layout
        fig.add_traces(fig_elements)
        fig.show()
        return fig_elements

    def trial_quantities(self, param_name: str):
        output_dict = {}
        for trial, data in self.trial_fits.items():
            for key, value in data.items():
                if param_name.lower() in key.lower():
                    output_dict.update({f"{trial} {key.strip()}": value})
                else:
                    continue
        return output_dict

    def _trial_fits(self):
        parameters = {}
        for trial in self.frap_trials:
            trial_name = f"Trial_{str(trial.index)}"
            norm_data = trial.normalized_data.iloc[0:110]
            trial_params = fit_curve(norm_data, self.fit_method, self.prebleach_frames)
            parameters.update({trial_name: trial_params})
        return parameters

    def _experiment_fit(self):
        mean_list = list(self.mean_intensity)
        mean_df = pd.DataFrame({'TIMESTAMP': self.timestamps, 'DATA': mean_list})
        fit_params = fit_curve(mean_df, self.fit_method, self.prebleach_frames)
        return fit_params

    def _prepare_trials(self):
        for index, csv_path in enumerate(self.csv_path_list):
            name = str(os.path.basename(csv_path)).replace(" ", "_")[:-4]
            trial = self.FrapTrial(csv_path, prebleach_frames=self.prebleach_frames, bleach_frames=self.bleach_frames,
                                   index_scalar=self.index_scalar, roi_index=self.roi_index,
                                   whole_obj_index=self.whole_obj_index, bg_index=self.bg_index,
                                   timecode_index=self.timecode_index, normal_method=self.normal_method,
                                   fit_method=self.fit_method, name=name, index=index)

            setattr(self, f"Trial_{index}", trial)
            self.frap_trials.append(trial)

    def _get_merged_normalized(self):
        normalized_data_val_list = []
        for trial in self.frap_trials:
            normalized_data_df = trial.normalized_data
            normalized_data_values = normalized_data_df['DATA'].values
            normalized_data_val_list.append(normalized_data_values)
        data_array = np.vstack(normalized_data_val_list).transpose()

        return data_array

    class FrapTrial:
        def __init__(self, csv_path: str, prebleach_frames: int, bleach_frames: int, index_scalar: int, roi_index=1,
                     whole_obj_index=2, bg_index=3, timecode_index=None, normal_method='fullscale',
                     fit_method='double', name='', index=0):
            normalized, depth = process_frapdata(normal_method, csv_path, prebleach_frames, bleach_frames, index_scalar,
                                                 roi_index, whole_obj_index, bg_index, timecode_index)

            fit_results = fit_curve(normalized, fit_method, prebleach_frames)

            self.index = index
            self.fit_results = fit_results
            self.normalized_data = normalized
            self.bleach_depth = depth
            self.name = name