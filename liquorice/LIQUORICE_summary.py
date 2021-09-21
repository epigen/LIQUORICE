import argparse
import pandas as pd
import glob
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
matplotlib.rcParams.update({'font.size': 20})
import numpy as np
from pathlib import Path
from scipy import ndimage
from collections import defaultdict
import logging
import typing
import json
import seaborn as sns
import matplotlib.patches as mpatches
from scipy import stats
import pathlib


def plot_control_distributions_and_test_normality(summary_df: pd.DataFrame,col: str,alpha: float=0.05,
                                                  use_uncorrected_coverage: bool = False):
    """
    For each region-set in **summary_df**, plot the distribution of the control sample's score measured by metric
    **col** as histograms and probability plots
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html).
    Also run tests the Shapiro-Wilk test for normal distributions and prints a warning if this test detects significant
    deviations from the normal distribution. Plotting and testing is skipped for every region-set where n<4.

    :param summary_df: Input `pandas.DataFrame` with the columns "is_control" and **col**
    :param col: Name of the column that should be used for plotting
    :param use_uncorrected_coverage: If True, indicate in the file name that the uncorrected coverage has been used to
        generate the underlying data.
    :return:
    """

    control_df=summary_df[summary_df["is control"]=="yes"]

    regionset_list = list(set(summary_df["region-set"]))
    for regionnr, regionset in enumerate(regionset_list):

        outpath=f"{regionset}_control_distribution_by_{'_'.join(col.split(' ')[1:3]).replace(':','')}" \
                f"{'_using_uncorrected_coverage' if use_uncorrected_coverage else ''}.pdf"

        if control_df[control_df['region-set'] == regionset].shape[0]<2:
            print(f"WARNING: Only {control_df[control_df['region-set'] == regionset].shape[0]} control sample(s) are/is"
                  f" available for "
                  f"region-set {regionset}. Statistical comparisons for this region-set were skipped.")
            continue

        if control_df[control_df['region-set'] == regionset].shape[0]<4:
            print(f"WARNING: Only {control_df[control_df['region-set'] == regionset].shape[0]} control samples are "
                  f"available for "
                  f"region-set {regionset}. Statistical comparisons for this region-set are likely inaccurate. "
                  f"Tests for normal distributions and plotting of the control distributions will be skipped.")
            continue

        try:
            normality_rejected_shapiro = stats.shapiro(control_df[control_df["region-set"] == regionset][col])[1]<alpha
            if normality_rejected_shapiro:
                print(f"\nWARNING: The  test for normality (Shapiro-Wilk) indicated that the control samples are "
                      f"not normally distributed for "
                      f"region-set '{regionset}' and metric '{col}'."
                      f" Please beware that this may impact the validity of the statistical "
                      f"comparisons. You can check the distribution of the control samples' scores in the plot '"
                      f"{outpath}'.\n")
        except ValueError:
            print(f"WARNING: Only {control_df[control_df['region-set'] == regionset].shape[0]} control samples are "
                  f"available for "
                  f"region-set {regionset}. Statistical comparisons for this region-set may be inaccurate, and tests for"
                  f" normal distribution could not be performed. You can check the distribution of the control samples' "
                  f"scores in the plot '{outpath}'.\n")

            #return np.nan,np.nan

        plt.subplot(1,2,1)
        plt.hist(control_df[control_df["region-set"] == regionset][col],bins=10)
        plt.title(f"Distribution of control scores in region-set {regionset}", fontsize=15)
        plt.xlabel(col)
        plt.ylabel("Nr. of control samples")
        plt.subplot(1,2,2)
        stats.probplot(control_df[control_df["region-set"] == regionset][col], dist="norm",plot=plt)
        plt.gcf().set_size_inches(16,8)
        plt.savefig(outpath)
        plt.close()



def get_prediction_interval_per_row(row: pd.Series, control_df: pd.DataFrame, col: str, alpha: int =0.05) -> \
                                    typing.Tuple[float, float]:
    """
    For a row in a `pandas.DataFrame`, find control values for the same regionset and return the prediction interval.

    :param row: A `pandas.Series` with the columns 'region-set', 'is control', and **col**, corresponding to a single
        sample.
    :param control_df: A `pandas.DataFrame` with at least the columns 'region-set', and **col**. Should only
        contain data of control samples.
    :param col: Use this column as a metric to determine differences.
    :param alpha: Alpha level for the prediction interval. Default 0.05: 95% prediction interval
    :return: A tuple (lower prediction interval,higher prediction interval), or (np.nan,np.nan) if calculation is not
        possible.
    """

    std_ctrl = control_df[control_df["region-set"] == row["region-set"]].std()[col]

    if std_ctrl!=std_ctrl:
        return np.nan,np.nan

    pi_lower,pi_higher=get_prediction_interval(control_df[control_df["region-set"] == row["region-set"]][col].values)

    return pi_lower,pi_higher


def get_prediction_interval(values: typing.List,alpha: int=0.05) -> typing.Tuple[float, float]:
    """
    Returns the upper and lower prediction interval for a list of values.

    :param values: Scores for which the prediction interval shall be calculated
    :param alpha: Alpha level for the prediction interval. Default 0.05: 95% prediction interval
    :return: A tuple (lower_prediction_interval, upper_prediction_interval)
    """

    mean=np.mean(values)
    std=np.std(values)
    n=len(values)

    t_higher=stats.t.ppf(1-alpha/2, n-1)
    t_lower=stats.t.ppf(alpha/2, n-1)
    pi_lower = mean + t_lower*std*np.sqrt(1+1/n) # Lower prediction interval
    pi_higher = mean + t_higher*std*np.sqrt(1+1/n) # Upper prediction interval

    return (pi_lower,pi_higher)


def check_difference_to_control(row: pd.Series, control_df: pd.DataFrame, col: str,
                                prediction_interval_control_col: str, negative_is_strong: bool,
                                ) -> str:
    """
    For a row in a `pandas.DataFrame`, find control values for the same regionset and return if sample is significantly
    different compared to the controls, as measured by metric **col**. A sample is deemed significantly different if
     its score in metric **col** lies outside the prediction interval of the control group.

    :param row: A `pandas.Series` with the columns 'region-set', 'is control', and **col**, corresponding to a single
        sample.
    :param control_df: A `pandas.DataFrame` with at least the columns 'region-set', and **col**. Should only
        contain data of control samples.
    :param col: Use this column as a metric to determine differences.
    :param prediction_interval_control_col: Name of the column that contains the prediction interval of the control
        group.
    :param negative_is_strong:  Set to `True` if a very negative value of the metric is associated with a strong dip
        signal, such as for the dip area.
    :return: A string indicating the result of the comparison, or np.nan if row[col] is NaN.
    """

    if np.nan in row[prediction_interval_control_col]:
        return "N/A"

    pi_lower,pi_higher=row[prediction_interval_control_col]
    std_ctrl = control_df[control_df["region-set"] == row["region-set"]].std()[col]

    if row["is control"] == "yes" or std_ctrl!=std_ctrl:
        return "N/A"

    # If a strong sample should have a very negative value, a sign. stronger sample needs to have a more negative
    # value than the lower PI
    if (row[col] < pi_lower) and negative_is_strong:
        return "Significantly stronger coverage drop"
    # If a strong sample should have a very negative value, a sign. weaker sample needs to have a more positive value
    # than the upper PI
    elif (row[col] > pi_higher) and negative_is_strong:
        return "Significantly weaker coverage drop"

    # If a strong sample should have a very positive value, a sign. stronger sample needs to have a more positive value
    # than the upper PI
    elif (row[col] > pi_higher) and not negative_is_strong:
        return "Significantly stronger coverage drop"

    # If a strong sample should have a very positive value, a sign. weaker sample needs to have a more negative value
    # than the lower PI
    elif (row[col] < pi_lower) and not negative_is_strong:
        return "Significantly weaker coverage drop"

    elif row[col] != row[col]:
        return np.nan

    else:
        return "n.s."


def zscore_to_control(row: pd.Series, control_df: pd.DataFrame, col: str)  -> float:
    """
    For a row in a dataframe, find control values for the same regionset and return the zscore of the sample compared to
    these controls, as measured by metric "col".

    :param row: A pd.Series with the columns 'region-set', 'is control', and **col**, corresponding to a single sample.
    :param control_df: A `pandas.DataFrame` with at least the columns 'region-set', and **col**. Should only
        contain data of control samples.
    :param col: Calculate z-score by comparing this metric between the sample and controls.
    :return: z-score, rounded to 2 decimals
    """
    mean_ctrl = control_df[control_df["region-set"] == row["region-set"]].mean()[col]
    std_ctrl = control_df[control_df["region-set"] == row["region-set"]].std()[col]
    return round((row[col] - mean_ctrl) / std_ctrl, 2)


def get_list_of_coveragefiles(dirname: str, regionset: str, use_uncorrected_coverage:bool =False) -> typing.List[str]:
    """
    Return a list of 'corrected_coverage_mean_per_bin.csv' or 'uncorrected_coverage_mean_per_bin.csv' files, or
    return an error if no such files could be found.

    :param dirname: path to LIQUORICE output directory
    :param regionset: Name of the regionset of interest
    :param use_uncorrected_coverage: If True, load 'uncorrected_coverage_mean_per_bin.csv' files, else, load
        'corrected_coverage_mean_per_bin.csv files
    :return: A list of 'corrected_coverage_mean_per_bin.csv' or 'uncorrected_coverage_mean_per_bin.csv' file paths
    """

    if not use_uncorrected_coverage:
        coverage_filelist = sorted(glob.glob("%s/*/%s/corrected_coverage_mean_per_bin.csv" % (dirname, regionset))) + \
                            sorted(glob.glob("%s/%s/corr_coverageerage_mean_per_bin.csv" % (dirname, regionset)))
    else:
        coverage_filelist = sorted(
            glob.glob("%s/*/%s/uncorrected_coverage_mean_per_bin.csv" % (dirname, regionset))) + sorted(
            glob.glob("%s/%s/uncorrected_coverage_mean_per_bin.csv" % (dirname, regionset)))

    if len(coverage_filelist) == 0:
        filename = 'corrected_coverage_mean_per_bin.csv' if not use_uncorrected_coverage else 'coverage_mean_per_bin.csv'
        sys.exit(f"ERROR: could not find any files with the name {filename}. Please check that you specified the "
                 f"correct directory and check the --use_uncorrected_coverage setting.")
    return coverage_filelist


def verify_consistant_binning_settings(dirname: str, regionset:str):
    """
    Asserts that, for the given regionset, all samples have the same binning settings.
    Calls `sys.exit()` with an error message if settings are inconsistent or cannot be found.

    :param regionset_list: A list of region-sets to be analyzed.
    """

    binsize=None
    extend_to=None
    binning_settings_glob=glob.glob("%s/*/%s/binning_settings.json" % (dirname, regionset))

    if len(binning_settings_glob) != len(glob.glob(f"{dirname}/*/{regionset}")):
        sys.exit(f"Could not detect a binning_settings.json file for every sample-directory in {regionset}. Aborting."
                 f"You can specify --extend_to and --binsize manually, add the missing files, or remove the incomplete"
                 f"sample-directories.")
    sample=None
    try:
        for binning_setting in binning_settings_glob:
            sample=binning_setting.split("/")[-3]
            with open(binning_setting) as json_file:
                settings_dict = json.load(json_file)
            if not binsize:
                binsize=settings_dict["binsize"]
            if not extend_to:
                extend_to=settings_dict["extend_to"]
            assert binsize==settings_dict["binsize"]
            assert extend_to==settings_dict["extend_to"]

    except AssertionError:
        sys.exit(f"Found binning settings that are inconsistant between samples for regionset {regionset}. "
                 f"The conflicting samples include: {sample} and {binning_settings_glob[0]} (and maybe more). "
                 f"Seems like LIQUORICE has been called with different --extend_to and/or --binsize parameters in "
                 f"the same working directory and for the same region-set (or the same region-set-name). "
                 f"Re-run LIQUORICE with consistent "
                 f"settings or move the inconsistent result folders to a different directory. "
                 f"The summary table has been generated regardless, but summary plots cannot be generated.")


def get_binsize_and_extendto_from_saved_settings(dirname: str, regionset:str) -> typing.Tuple[int,int]:
    """
    Calls :func:`verify_consistant_binning_settings` and returns the binsize and extend_to settings used by LIQUORICE
    for a given regionset, if no error is raised.

    :param dirname: Path to LIQUORICE's working directory.
    :param regionset: Name of the regionset of interest.
    :return: A tuple with two integers: binsize and extend_to.
    """
    verify_consistant_binning_settings(dirname=dirname,regionset=regionset)
    binning_settings_glob=glob.glob("%s/*/%s/binning_settings.json" % (dirname, regionset))
    with open(binning_settings_glob[0]) as json_file:
        settings_dict = json.load(json_file)
        binsize=settings_dict["binsize"]
        extend_to=settings_dict["extend_to"]

    return binsize,extend_to


def plot_overlay(dirname: str, summary_df: pd.DataFrame, control_name_list: typing.List[str],
                 extend_to : int =None, binsize: int =None,
                 normalize_by_intercept: bool =True, smooth_sigma: float =3,
                 significance_col: str = "Dip area: interpretation vs controls in same region set",
                 use_uncorrected_coverage: bool =False,alpha: float =0.5, linewidth: float =3):
    """
    Summarize plots, create one plot per regionset. Case samples with significant differences to controls
    (based on **significance_col**) are marked by color.
    **smooth_sigma** can be used to make the plots smoother and easier to compare - this is a visual
    effect only.

    :param dirname: Path to LIQUORICE's working directory.
    :param summary_df: The output of the function :func:`create_summary_table_LIQUORICE`.
    :param control_name_list: List of names of samples that should be plotted as controls.  Can be empty.
    :param extend_to: *extend_to* setting that was used when running LIQUORICE. If None, infer from the
        binning_settings.json files that are saved by LIQUORICE by default.
    :param binsize: *binsize* setting that was used when running LIQUORICE. If None, infer from the
        binning_settings.json files that are saved by LIQUORICE by default.
    :param normalize_by_intercept: If True, extract the intercept from the fitted model to normalize the dips between
        samples (intercept of each sample is positioned at y=0). Otherwise, the mean coverage of each sample is
        positioned at y=0.
    :param smooth_sigma: Visually smooth the coverage signals, using this strength of smoothing. Higher values indicate
        stronger smoothing. Set to 0 for no smoothing.
    :param significance_col: Use this columns as an indicator for significant differences between case and control
        samples. Can contain the following values: "Significantly stronger coverage drop",
        "Significantly weaker coverage drop", "n.s.", and "N/A".
    :param use_uncorrected_coverage: If True, plot the coverage profile that is not corrected for biases by LIQUORICE.
    :param alpha: Alpha (transparency) parameter for plotting.
    :param linewidth: Linewidth parameter for plotting
    """
    regionset_list = list(set(summary_df["region-set"]))

    for regionset in regionset_list:

        if binsize is None or extend_to is None:
            inferred_binsize, inferred_extend_to = get_binsize_and_extendto_from_saved_settings(
                dirname=dirname,regionset=regionset)
        binsize = binsize if binsize is not None else inferred_binsize
        extend_to = extend_to if extend_to is not None else inferred_extend_to


        coverage_filelist=get_list_of_coveragefiles(dirname, regionset,
                                                    use_uncorrected_coverage=use_uncorrected_coverage)

        nr_samples = defaultdict(int)

        fig = plt.figure(frameon=False)
        ax1 = plt.subplot(1, 2, 1)
        ctrl_ax = plt.subplot(1, 2, 1, sharex=ax1, sharey=ax1)
        case_ax = plt.subplot(1, 2, 2, sharex=ax1, sharey=ax1)

        for coveragefile in coverage_filelist:

            samplename = str(Path(coveragefile).parent.parent).split("/")[-1]

            #avg_centersize=150
            regions_df=pd.read_csv("/".join(coveragefile.split("/")[:-1])+"/regions.bed",sep="\t",header=None)
            avg_centersize=int(round((regions_df[2]-regions_df[1]).mean()))

            summary_df_this_sample_and_region = summary_df[
                (summary_df["sample"] == samplename) & (summary_df["region-set"] == regionset)]
            if summary_df_this_sample_and_region.empty:
                sys.exit(f"ERROR: Could not find an entry corresponding to {samplename} in the "
                         f"summary_across_samples_and_ROIS.csv file. Please ensure that the folder names and the"
                         f" sample name entry in {samplename}/{regionset}/fitted_gaussians_parameter_summary.csv are"
                         f"consistant - i.e. do not rename the sample-wise output directories of LIQUORICE (or edit the"
                         f" fitted_gaussians_parameter_summary.csv file accordingly).")
            df_cov = pd.read_csv(coveragefile)
            cov_col=df_cov.columns[1]


            if not normalize_by_intercept:
                y_norm = [y - df_cov[cov_col].mean() for y in df_cov[cov_col].values]
            else:
                intercept = summary_df_this_sample_and_region["Intercept"]
                y_norm = [y - intercept for y in df_cov[cov_col].values]
            if smooth_sigma:
                y_norm = ndimage.gaussian_filter(y_norm, mode="nearest", sigma=smooth_sigma)


            central_bin_x_values=[x*avg_centersize-avg_centersize/2 for x in [0.05,0.175,0.5,0.825,0.95]]
            x=list(np.arange(-extend_to,extend_to+1, binsize)) # set x such that all bins have the same size
            if len(x)!=len(y_norm): # check if this fits the actual data
                # if not, assume that the core region has been split differently, as given in central_bin_x_values.
                x=list(np.arange(-extend_to+(binsize/2)-avg_centersize/2,(binsize/2)-avg_centersize/2,binsize))+central_bin_x_values+list(np.arange(avg_centersize/2+(binsize/2),extend_to+avg_centersize/2+(binsize/2),binsize))

            try:
                assert len(x) == len(y_norm)
            except AssertionError:
                print(len(x),len(y_norm))
                sys.exit(f"ERROR: --binsize and/or --extend_to of file {coveragefile} does not fit to the data from "
                         f"other samples/region-sets. "
                         f"Seems like LIQUORICE has been called with different --extend_to and/or --binsize parameters"
                         f" in "
                         f"the same working directory and for the same region-set (or the same region-set-name). "
                         f"Re-run LIQUORICE with consistent "
                         f"settings or move the inconsistent result folders to a different directory.")

            plt.sca(ctrl_ax)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.sca(case_ax)
            plt.gca().yaxis.set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_visible(False)

            if samplename in control_name_list:
                plt.sca(ctrl_ax)
                plt.plot(x, y_norm, alpha=alpha, color="seagreen", linewidth=linewidth, zorder=9)
                nr_samples["ctrl"] += 1
            else:
                plt.sca(case_ax)

                if summary_df_this_sample_and_region[significance_col].values[0]!= \
                                 summary_df_this_sample_and_region[significance_col].values[0]:
                    significance_status="n.s."
                else:
                    significance_status=summary_df_this_sample_and_region[significance_col].values[0]
                nr_samples[significance_status]+=1
                sign_to_color_dict={"Significantly stronger coverage drop":"firebrick",
                    "Significantly weaker coverage drop":"darkblue", "n.s.":"grey","N/A":"grey"}
                color=sign_to_color_dict[significance_status]

                plt.plot(x, [val_list[0] for val_list in y_norm] if normalize_by_intercept else y_norm,
                         alpha=alpha, color=color, linewidth=linewidth, zorder=1)
        legend_elems = [
            Line2D([0], [1], lw=1.5, color="firebrick", label='Case samples with significantly stronger coverage drop, '
                                                              f'n={nr_samples["Significantly stronger coverage drop"]}', markersize=1),
            Line2D([0], [1], lw=1.5, color="darkblue",label='Case samples with significantly weaker coverage drop, '
                                                            f'n={nr_samples["Significantly weaker coverage drop"]}', markersize=1),
            Line2D([0], [1], lw=1.5, color="silver", label='Case samples not sign. different or N/A, '
                                                           f'n={nr_samples["n.s."]+nr_samples["N/A"]}', markersize=1),
            Line2D([0], [1], lw=1.2, color="darkseagreen", label=f'Control samples, n={nr_samples["ctrl"]}',
                   markersize=1)]

        plt.suptitle(f"Overlay plot for region-set {regionset}")
        ttl = plt.gca().title
        ttl.set_position([.5, 1.05])
        plt.xlim([-extend_to, extend_to])
        plt.sca(ctrl_ax)
        plt.legend(handles=legend_elems, fontsize=13)
        plt.ylabel(f"Bias corrected coverage, normalized by {'intercept' if normalize_by_intercept else 'mean'}", labelpad=4)
        plt.xlabel("Relative distance to center of ROI [bp]")
        plt.sca(case_ax)
        plt.xlabel("Relative distance to center of ROI [bp]")

        fig = plt.gcf()
        fig.set_size_inches(16, 9)
        plt.subplots_adjust(left=0.1, right=0.965, bottom=0.14, top=0.9, hspace=0.25)
       # plt.locator_params(axis='y', nbins=5)
        fname=f"{regionset}_summary_plot_overlay_of_samples_by_{'_'.join(significance_col.split(' ')[:2]).replace(':','')+('_using_uncorrected_coverage' if use_uncorrected_coverage else '')}.pdf"
        plt.savefig(fname)
        plt.close()


def get_ymin_ymax_over_all_samples(coverage_filelist: typing.List[str],
                                   normalize_by_intercept: bool, regionset: str, summary_df: pd.DataFrame) \
        -> typing.Tuple[float,float]:
    """
    Get the the upper and lower limits for the y-axis for a given regionset, such that the same scale is used for all
    samples.

    :param coverage_filelist: List of 'corrected_coverage_mean_per_bin.csv' (or 'uncorrected_coverage_mean_per_bin.csv')
     files
    :param normalize_by_intercept: f True, extract the intercept from the fitted model to normalize the dips between
        samples (intercept of each sample is positioned at y=0). Otherwise, the mean coverage of each sample is
        positioned at y=0.
    :param regionset: Name of the regionset of intest as given in the summary_df
    :param summary_df: pd.DataFrame with columns 'sample', 'region-set', and 'Intercept'.
    :return: A tuple of floats: y_min,y_max
    """
    # Figure out the min and max over all samples such that they can be drawn to same scale
    y_min_list = []
    y_max_list = []
    for i, coveragefile in enumerate(coverage_filelist):
        df_cov = pd.read_csv(coveragefile)
        cov_col = df_cov.columns[1]
        if not normalize_by_intercept:
            y_max_list.append(max([y - df_cov[cov_col].mean() for y in df_cov[cov_col].values]))
            y_min_list.append(min([y - df_cov[cov_col].mean() for y in df_cov[cov_col].values]))
        else:
            samplename = str(Path(coveragefile).parent.parent).split("/")[-1]
            summary_df_this_sample_and_region = summary_df[
                (summary_df["sample"] == samplename) & (summary_df["region-set"] == regionset)]
            intercept = summary_df_this_sample_and_region["Intercept"].values[0]
            y_max_list.append(max([y - intercept for y in df_cov[cov_col].values]))
            y_min_list.append(min([y - intercept for y in df_cov[cov_col].values]))
    y_min = min(y_min_list)
    y_max = max(y_max_list)
    enlarge = (y_max - y_min) * 0.05
    y_min = y_min - enlarge
    y_max = y_max + enlarge
    return y_min, y_max


def plot_as_subplots(dirname: str, summary_df: pd.DataFrame, control_name_list: typing.List[str],
                     extend_to : int =None, binsize: int =None,
                     normalize_by_intercept: bool =True, smooth_sigma: float =0,
                     significance_col: str = "Dip area: interpretation vs controls in same region set",
                     use_uncorrected_coverage: bool =False,y_min_fixed: float=None, y_max_fixed:float =None):
    """
    Summarize plots, create one set of plots per regionset. Case samples with significant differences to controls
    (based on **significance_col**) are marked by color.
     **smooth_sigma** can be used to make the plots smoother and easier to compare, this is a visual effect only.

    :param dirname: Path to LIQUORICE's working directory.
    :param summary_df: The output of the function :func:`create_summary_table_LIQUORICE`.
    :param control_name_list: List of names of samples that should be plotted as controls.  Can be empty.
    :param extend_to: *extend_to* setting that was used when running LIQUORICE. If None, infer from the
        binning_settings.json files that are saved by LIQUORICE by default.
    :param binsize: *binsize* setting that was used when running LIQUORICE. If None, infer from the
        binning_settings.json files that are saved by LIQUORICE by default.
    :param normalize_by_intercept: If True, extract the intercept from the fitted model to normalize the dips between
        samples (intercept of each sample is positioned at y=0). Otherwise, the mean coverage of each sample is
        positioned at y=0.
    :param smooth_sigma: Visually smooth the coverage signals, using this strength of smoothing. Higher values indicate
        stronger smoothing. Set to 0 for no smoothing.
    :param significance_col: Use this columns as an indicator for significant differences between case and control
        samples. Can contain the following values: "Significantly stronger coverage drop",
        "Significantly weaker coverage drop", "n.s.", and "N/A".
    :param use_uncorrected_coverage: If True, plot the coverage profile that is not corrected for biases by LIQUORICE.
    :param y_min_fixed: If specified, use this value as the minimum value for the y axis.
    :param y_min_fixed: If specified, use this value as the maximum value for the y axis.
    """
    regionset_list = list(set(summary_df["region-set"]))
    for regionnr, regionset in enumerate(regionset_list):

        if binsize is None or extend_to is None:
            inferred_binsize, inferred_extend_to = get_binsize_and_extendto_from_saved_settings(
                dirname=dirname,regionset=regionset)
        binsize = binsize if binsize is not None else inferred_binsize
        extend_to = extend_to if extend_to is not None else inferred_extend_to

        coverage_filelist=get_list_of_coveragefiles(dirname, regionset, use_uncorrected_coverage=use_uncorrected_coverage)

        fig = plt.figure(frameon=False)

        y_min_auto, y_max_auto = get_ymin_ymax_over_all_samples(coverage_filelist, normalize_by_intercept, regionset,
                                                                summary_df)
        y_min = y_min_fixed if y_min_fixed else y_min_auto
        y_max = y_max_fixed if y_max_fixed else y_max_auto

        if len(coverage_filelist)>=40:
            n_rows_on_pdf=5
            n_cols_on_pdf=8
        else:
            n_cols_on_pdf=np.ceil(2*np.sqrt(len(coverage_filelist)/2))
            n_rows_on_pdf=np.ceil(np.sqrt(len(coverage_filelist)/2))
        fname=f"{regionset}_summary_plot_subplots_of_samples_by_{'_'.join(significance_col.split(' ')[:2]).replace(':','')+('_using_uncorrected_coverage' if use_uncorrected_coverage else '')}.pdf"
        with PdfPages(fname) as pdf:
            counter = 0
            for i, coveragefile in enumerate(coverage_filelist):

                plt.subplot(n_rows_on_pdf, n_cols_on_pdf, counter + 1)

                samplename = str(Path(coveragefile).parent.parent).split("/")[-1]

                #avg_centersize=150
                regions_df=pd.read_csv("/".join(coveragefile.split("/")[:-1])+"/regions.bed",sep="\t",header=None)
                avg_centersize=int(round((regions_df[2]-regions_df[1]).mean()))

                plt.title(samplename.replace("_trainonRandom_allBinSameSize_extto15k_bs150_autoML_800maxrepaired",""))

                summary_df_this_sample_and_region = summary_df[
                    (summary_df["sample"] == samplename) & (summary_df["region-set"] == regionset)]
                df_cov = pd.read_csv(coveragefile)
                cov_col = df_cov.columns[1]

                if not normalize_by_intercept:
                    y = [y - df_cov[cov_col].mean() for y in df_cov[cov_col].values]
                if normalize_by_intercept:
                    intercept = summary_df_this_sample_and_region["Intercept"].values[0]
                    y = [y - intercept for y in df_cov[cov_col].values]
                if smooth_sigma:
                    y = ndimage.gaussian_filter(y, mode="nearest", sigma=smooth_sigma)

                central_bin_x_values=[x*avg_centersize-avg_centersize/2 for x in [0.05,0.175,0.5,0.825,0.95]]
                x=list(np.arange(-extend_to,extend_to+1, binsize)) # set x such that all bins have the same size
                if len(x)!=len(y):  # check if this fits the actual data
                    # if not, assume that the core region has been split differently, as given in central_bin_x_values.
                    x=list(np.arange(-extend_to+(binsize/2)-avg_centersize/2,(binsize/2)-avg_centersize/2,binsize))+central_bin_x_values+list(np.arange(avg_centersize/2+(binsize/2),extend_to+avg_centersize/2+(binsize/2),binsize))
                try:
                    assert len(x) == len(y)
                except AssertionError:
                    sys.exit(f"ERROR: --binsize and/or --extend_to does not fit to the df {coveragefile}. "
                             "Are you sure you entered the same parameters as originally used for LIQUOIRCE? "
                             "Please adjust them accordingly. Aborting")

                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                if samplename in control_name_list:
                    color="darkseagreen"
                else:
                    if summary_df_this_sample_and_region[significance_col].values[0]!= \
                            summary_df_this_sample_and_region[significance_col].values[0]:
                        significance_status="n.s."
                    else:
                        significance_status=summary_df_this_sample_and_region[significance_col].values[0]
                    sign_to_color_dict={"Significantly stronger coverage drop":"firebrick",
                        "Significantly weaker coverage drop":"darkblue", "n.s.":"grey","N/A":"grey"}
                    color=sign_to_color_dict[significance_status]

                plt.plot(x, y, alpha=0.6, color=color, linewidth=1, zorder=1)

                plt.ylabel(f"Bias corrected coverage,\n "
                           f"normalized by {'intercept' if normalize_by_intercept else 'mean'}", labelpad=12)
                plt.xlabel("Relative distance to center of ROI [bp]")
                plt.ylim(y_min, y_max)
                if not i > len(coverage_filelist) - n_cols_on_pdf - 1:
                    plt.gca().label_outer()
                elif counter % n_cols_on_pdf:
                    plt.ylabel("")
                    plt.yticks([])
                plt.locator_params(axis='y', nbins=5)
                plt.locator_params(axis='x', nbins=3)

                if counter == 0 and (significance_col in summary_df_this_sample_and_region.columns):
                    legend_elems = [
                        Line2D([0], [1], lw=1.5, color="firebrick",label='Case samples with significantly stronger '
                                                                         'coverage drop',markersize=1),
                        Line2D([0], [1], lw=1.5, color="darkblue",label='Case samples with significantly weaker '
                                                                        'coverage drop',markersize=1),
                        Line2D([0], [1], lw=1.5, color="grey",label='Case samples not sign. different or N/A',markersize=1),
                        Line2D([0], [1], lw=1.2, color="darkseagreen", label='Control samples', markersize=1)]
                    plt.legend(handles=legend_elems, fontsize=20, loc="lower left", bbox_to_anchor=(0, 1.15), ncol=1)

                if counter + 1 == n_rows_on_pdf*n_cols_on_pdf or i == len(coverage_filelist) - 1:
                    plt.suptitle("Coverage plots for region-set %s" % (regionset), fontsize=40)
                    ttl = plt.gca().title
                    ttl.set_position([.5, 1.05])

                    plt.xlim([-extend_to, extend_to])

                    # plt.legend(handles=legend_elems,fontsize=13)

                    fig = plt.gcf()
                    fig.set_size_inches(48, 27)
                    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90 if len(coverage_filelist)>=40 else 0.8, hspace=0.25)
                    pdf.savefig()
                    plt.close()
                    counter = -1
                counter += 1


def create_summary_table_LIQUORICE(dirname: str, control_name_list: typing.List[str],
                                   these_regionsets_only: typing.List[str],
                                   use_uncorrected_coverage:bool =False,
                                   prediction_interval_alpha: int=0.05) -> pd.DataFrame:
    """
    For a LIQUORICE result directory, creates and writes to csv a pd.DataFrame summarizing the coverage drop metrics
    for all samples and
    region-sets. If **control_name_list** is given, compares the case samples to control samples and assesses
    significant differences, separately for each region-set.

    :param use_uncorrected_coverage: If True, summarize the coverage profile that is not corrected for biases by
     LIQUORICE instead of the corrected coverage.
    :param these_regionsets_only: Summarize only data for these regionsets.
    :param dirname: Output directory of LIQUORICE in which to search for fitted_gaussians_parameter_summary.csv
        (or uncorrected_fitted_gaussians_parameter_summary.csv) files
    :param control_name_list: Sample names of the controls, which will be used as reference for generating z-scores.
    :param prediction_interval_alpha: Alpha level for the prediction interval. Default 0.05: 95% prediction interval
    :return: A `pandas.DataFrame` in which all parameters saved in the 'fitted_gaussians_parameter_summary.csv'
        (or uncorrected_fitted_gaussians_parameter_summary.csv) files
        are summarized over all samples and region-sets. If control_name_list is not empty, additional columns of the
        DataFame contain metric comparisons to the the control samples in the form of z-scores of dip area and depth.
    """
    parameter_filename="fitted_gaussians_parameter_summary.csv" if not use_uncorrected_coverage else \
        "uncorrected_fitted_gaussians_parameter_summary.csv"
    param_summaries = glob.glob(f"{dirname}/*/*/{parameter_filename}") + glob.glob(f"{dirname}/*/{parameter_filename}")
    if these_regionsets_only:
        param_summaries = [x for x in param_summaries if x.split("/")[-2] in these_regionsets_only]
    if len(param_summaries) == 0:
        sys.exit("Could not find any 'fitted_gaussians_parameter_summary.csv' files in this directory. Exiting")

    summary_df = pd.DataFrame
    for idx, param_summary in enumerate(param_summaries):
        if not idx:
            summary_df = pd.read_csv(param_summary)
        else:
            summary_df = summary_df.append(pd.read_csv(param_summary))

    control_summary_df = summary_df[summary_df["sample"].isin(control_name_list)].copy()
    summary_df = summary_df.assign(**{
        "is control": ["yes" if sample in control_name_list else "no" for sample in summary_df["sample"].values]})

    summary_df = summary_df.assign(**{
        "Dip area: z-score vs controls in same region set": summary_df.apply(
        lambda x: zscore_to_control(x, control_summary_df, "Total dip area (AOC combined model)"), axis=1),
        "Dip area: Prediction interval of controls": summary_df.apply(
            lambda x: get_prediction_interval_per_row(x, control_summary_df, "Total dip area (AOC combined model)",
                                                      prediction_interval_alpha),axis=1)})
    summary_df = summary_df.assign(**{
        "Dip area: interpretation vs controls in same region set": summary_df.apply(
            lambda x: check_difference_to_control(x, control_summary_df, "Total dip area (AOC combined model)",
                                                  "Dip area: Prediction interval of controls",
                                                  True), axis=1)})

    summary_df = summary_df.assign(**{
        "Dip depth: z-score vs controls in same region set": summary_df.apply(
            lambda x: zscore_to_control(x, control_summary_df, "Total dip depth"), axis=1),
        "Dip depth: Prediction interval of controls": summary_df.apply(
            lambda x: get_prediction_interval_per_row(x, control_summary_df, "Total dip depth",
                                                      prediction_interval_alpha), axis=1)})
    summary_df = summary_df.assign(**{
        "Dip depth: interpretation vs controls in same region set": summary_df.apply(
            lambda x: check_difference_to_control(x, control_summary_df, "Total dip depth",
                                                  "Dip depth: Prediction interval of controls",False), axis=1)})

    return summary_df

def boxplot_score_summary(summary_df: pd.DataFrame,comparison_col: str,
                          use_uncorrected_coverage: bool):
    """
    Summarize scores via boxplots, with one box per group (case/control) - region-set combination.

    :param summary_df: The output of the function :func:`create_summary_table_LIQUORICE`.
    :param comparison_col: Use this column for the y axis of the boxplots.
    :param use_uncorrected_coverage: If True, indicate in the output filename that the uncorrected coverage scores are
        shown.
    """
    with sns.axes_style("whitegrid"):
        alpha=0.6

        summary_df["Case/Control"]=["Case" if x =="no" else "Control" for x in summary_df["is control"]]
        summary_df=summary_df.rename({"region-set":"Region-set"},axis=1)
        # summary_df=summary_df.replace({"sknmc_specific":"EwS-specific DHSs",
        #                                   "concat_haem_clusters_v2.0_LOhg38":"Blood-specific DHSs",
        #                                   "liverDHS_LOhg38":"Liver-specific DHSs"})

        sns.violinplot(x="Region-set",y=comparison_col,hue="Case/Control",data=summary_df,
                       palette={"Control":"seagreen","Case":"firebrick"},cut=0,inner=None,linewidth=0,scale="area",
                       scale_hue=False)
        for violin in plt.gca().collections:
            violin.set_alpha(alpha)

        sns.swarmplot(x="Region-set",y=comparison_col,hue="Case/Control",data=summary_df,dodge=True,
                      palette={"Control":"gainsboro","Case":"gainsboro"},size=3,edgecolor="grey")

        legend_elems=[
            mpatches.Patch(color="seagreen", label='Control samples',alpha=alpha,linewidth=0),
            mpatches.Patch(color="#CA4B47", label='Case samples',alpha=alpha,linewidth=0)]
        plt.legend(handles=legend_elems,loc="best")

        plt.ylabel("Total dip area\n(AOC combined model)" if comparison_col=="Total dip area (AOC combined model)" else
                   "Total dip depth")
        plt.gcf().set_size_inches(13, 6.5)
        plt.tight_layout()
        sns.despine(left=True,bottom=True)


        fname=f"boxplot_summary_by_dip_{'area' if comparison_col=='Total dip area (AOC combined model)' else 'depth'}" \
              f"{'_using_uncorrected_coverage' if use_uncorrected_coverage else ''}.pdf"
        plt.savefig(fname)
        plt.close()

def parse_args():
    """
    Parses the arguments from the command line. For a full list of arguments,
    see the documentation of the LIQUORICE_summary command line tool.

    :return: An `argparse.ArgumentParser` object storing the arguments.
    """
    parser = argparse.ArgumentParser(
            description="Summarize output of LIQUORICE for multiple samples and regions of interests in a table and "
                        "plots. Please make sure that for for any given region-set LIQUORICE has been run with "
                        "consistent 'binsize' and 'extend_to' settings across samples.)",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    optional_keyword_args = parser.add_argument_group('Optional named arguments')

    optional_keyword_args.add_argument("--dirname",
                                       help="Directory in which LIQUORICE's output is contained. Can contain output "
                                            "of multiple samples.", type=str,
                                       default=".", required=False)

    optional_keyword_args.add_argument('--control_name_list',
                                       help='List of samples that serve as reference control samples. Used to'
                                            'infer z-scores.',
                                       required=False, nargs="+",
                                       default=[])

    optional_keyword_args.add_argument('--binsize',
                                       help="--binsize setting that was used for LIQUORICE. Default: infer "
                                            "automatically", default=None, type=int)

    optional_keyword_args.add_argument(
        '--extend_to', help="--extend_to that was used for LIQUORICE. Default: infer automatically",
                                       default=None, type=int)

    optional_keyword_args.add_argument(
        '--smooth_sigma', help="Determines how strong the coverage drops should be smoothed for the overlay plot. Set "
                               "to 0 for no smoothing.", default=2.0, type=float)

    optional_keyword_args.add_argument(
        '--use_uncorrected_coverage',
        help="Per default the corrected coverages will be plotted. Set if you want to plot/summarize the uncorrected  "
             "coverage instead.", action="store_true")

    # optional_keyword_args.add_argument('--table_only', help="Only generate overview table, don't plot", default=False,
    #                                    type=bool)

    optional_keyword_args.add_argument('--these_regionsets_only',
                                       help='List of region sets for which a summary should be calculated. Default: '
                                            'summarize all detected region-sets', required=False, nargs="+",
                                       default=False)

    optional_keyword_args.add_argument('--prediction_interval_alpha',
                                       help='Alpha level for the prediction interval. Samples are deemed significantly '
                                            "different from the controls if their score lies outside the "
                                            "prediction interval of the control group. "
                                            "Note: Significance testing assumes a normal"
                                            "distribution of the scores of the control group. If tests for normality "
                                            "fail, assessment of significant differences is unavailable. "
                                            'Default 0.05: 95 precent prediction interval.', required=False,type=float,
                                       default=0.05)
    return parser


def main():
    """
    Main function for the LIQUOIRCE_summary tool. Calls the argument parser, checks user input, and calls the functions
    :func:`verify_consistant_binning_settings`
    :func:`create_summary_table_LIQUORICE` (saving output to .csv),
    :func:`plot_overlay`, :func:`plot_as_subplots`, :func:`boxplot_score_summary` and
    :func:`plot_control_distributions_and_test_normality`.
    """

    parser=parse_args()
    args = parser.parse_args()
    dirname = args.dirname
    control_name_list = args.control_name_list
    binsize = args.binsize
    extend_to = args.extend_to
    table_only = False #args.table_only
    these_regionsets_only = args.these_regionsets_only
    use_uncorrected_coverage = args.use_uncorrected_coverage
    smooth_sigma = args.smooth_sigma

    if (binsize and not extend_to) or (not binsize and extend_to):
        sys.exit("ERROR: Please specify either both --binsize AND --extend_to or set none of them and let the script "
                 "extract these parameters from the <dirname>/<samplename>/<regionset name>/binning_settings.json "
                 "file.")
    if len(set(control_name_list))!=len(control_name_list):
        sys.exit("ERROR: The list of control sample names (--control_names_list) contained duplicate entries. Please "
                 "remove the duplicates and try again.")
    summary_df = create_summary_table_LIQUORICE(dirname=dirname, control_name_list=control_name_list,
                                                these_regionsets_only=these_regionsets_only,
                                                use_uncorrected_coverage=use_uncorrected_coverage,
                                                prediction_interval_alpha=args.prediction_interval_alpha)
    summary_df.to_csv(f"summary_across_samples_and_ROIS{'_using_uncorrected_coverage' if use_uncorrected_coverage else ''}.csv", index=False)

    if not table_only:
        if control_name_list:
            for name in control_name_list:
                if "/" in name:
                    sys.exit(f"ERROR: Sample names must not contain '/'. Please correct the name '{name}' under "
                             "'--control_name_list'")
        # nr_different_regionsets=len(set(list(summary_df["region-set"].values)))
        # nr_samples=len(set(list(summary_df["sample"].values)))
        # if summary_df.shape[0] != nr_different_regionsets*nr_samples:
        #     sys.exit("ERROR: Cannot continue plotting because the data for some region-sets is available only for a "
        #              "subset of all samples. Please check the output ''")

        for comparison_metric in ["area","depth"]:
            plot_overlay(dirname=dirname,
                         summary_df=summary_df,
                         control_name_list=control_name_list,
                         extend_to=extend_to,
                         binsize=binsize,
                         significance_col="Dip area: interpretation vs controls in same region set" if
                         comparison_metric == "area" else "Dip depth: interpretation vs controls in same region set",
                         use_uncorrected_coverage=use_uncorrected_coverage, smooth_sigma=smooth_sigma,
                         normalize_by_intercept=True if not use_uncorrected_coverage else False)

            plot_as_subplots(dirname=dirname, summary_df=summary_df, control_name_list=control_name_list,
                             extend_to=extend_to,
                             binsize=binsize,
                             significance_col="Dip area: interpretation vs controls in same region set" if
                             comparison_metric == "area" else "Dip depth: interpretation vs controls in same "
                                                              "region set",
                             use_uncorrected_coverage=use_uncorrected_coverage,
                             normalize_by_intercept=True if not use_uncorrected_coverage else False)

            boxplot_score_summary(summary_df=summary_df,
                                  comparison_col="Total dip depth" if comparison_metric=="depth"
                                  else "Total dip area (AOC combined model)",
                                  use_uncorrected_coverage=use_uncorrected_coverage)

            plot_control_distributions_and_test_normality(summary_df=summary_df,
                                                          col="Total dip depth" if comparison_metric=="depth" else
                                                          "Total dip area (AOC combined model)",
                                                          use_uncorrected_coverage=use_uncorrected_coverage)



if __name__ == "__main__":
    sys.exit(main())
