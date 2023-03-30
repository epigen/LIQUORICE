import logging
from _datetime import datetime
import typing
import os
import pandas as pd

from liquorice.utils import RegionFilteringAndBinning
from liquorice.utils import CoverageAndSequenceTablePreparation
from liquorice.utils import CorrectForCNAs
from liquorice.utils import BinTableBiasFactors
from liquorice.utils import BiasModel
from liquorice.utils import Plotting
from liquorice.utils import AggregateAcrossRegions
from liquorice.utils import FitGaussians


def add_biasfactors_percentile_split(avg_readlength, liq_table, n_cores, sampled_fragment_lengths,
                                     skip_these_biasfactors=None):
    """
    Returns a table with added bias-factors, taking into account different bin sizes. Also adds a "bin size" column.

    :param avg_readlength: Average length of reads in the .bam file.
    :param liq_table: pandas.DataFrame` with one row per bin, containing columns "chromosome", "start", "end",
        "bin nr.", "coverage", "sequence", and "mappability". Suitable input is the output of the
        :func:`get_complete_table` method of the :attr:`liquorice.utils.CoverageAndSequenceTablePreparation` class
        object.
    :param n_cores: Max number of cores to use for calculations.
    :param sampled_fragment_lengths:  A list containing fragment lengths that are representative of the sample's global
     fragment size distribution. Typically a few hundred fragments will suffice here.
    :param skip_these_biasfactors: Do not calculate these bias factors. Only these entries are allowed:
        ["di and trinucleotides and GC content","mappability", "di and trinucleotides"]
    :return: A `pandas.DataFrame` with added bias-factors, taking into account different bin sizes, and a
        "bin size" column.
    """
    if skip_these_biasfactors is None:
        skip_these_biasfactors = []
    liq_table["bin size"] = liq_table["end"] - liq_table["start"]

    liq_tables_with_biasfactors_all_sizes = []
    bins_sizes=liq_table["bin size"].unique()
    logging.info(f"Calculating bias factors for bins of various sizes, using up to {n_cores} cores "
                 f" - this may take a while. "
                 f"(The following 'UserWarning's from modin can be ignored.)")
    logging.debug(f"Running for bin sizes {bins_sizes}.")

    prev_log_level=logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel("WARNING") # avoid that the logging gets cluttered with messages for every bin size
    for this_size in bins_sizes:
        table_this_size = liq_table[liq_table["bin size"] == this_size].copy()
        table_this_size["orig index"] = table_this_size.index
        table_this_size = table_this_size.reset_index()
        liq_table_with_biasfactors = BinTableBiasFactors.BiasFactorHandler(
            binsize=int(this_size),
            fragments=sampled_fragment_lengths,
            readlength=avg_readlength,
            df=table_this_size,
            n_cores=n_cores,
            skip_these_biasfactors=skip_these_biasfactors).get_table_with_bias_factors()
        liq_tables_with_biasfactors_all_sizes.append(liq_table_with_biasfactors)
        del table_this_size
    logging.getLogger().setLevel(prev_log_level)

    liq_table_with_biasfactors = pd.concat(liq_tables_with_biasfactors_all_sizes)
    liq_table_with_biasfactors = liq_table_with_biasfactors.set_index("orig index").sort_index().drop("index", axis=1)

    return liq_table_with_biasfactors


def train_biasmodel_for_sample(samplename: str, bam_filepath: str, bed_filepath: str, refgenome_filepath: str,
                               refgenome_chromsizes_filepath: str, refgenome_mappability_bigwig_path: str,
                               blacklist_bed_filepath: typing.Union[None,str],
                               sampled_fragment_lengths: typing.List[int], avg_readlength: int,
                               mean_seq_depth: float,
                               cna_seg_filepath: typing.Union[None,str],
                               n_cores: int, extend_to: int = 0, binsize: int = 500,
                               biasmodel_output_path: str = "trained_biasmodel.joblib",
                               nr_of_bins_for_training_and_testing: typing.Union[None,int] = None,
                               save_training_table: bool = False,
                               no_chr_prefix: bool=False,
                               percentile_split_core_rois:bool=False) -> None:
    """
    Go through all steps of LIQUORICE up to the biasmodel training. The resulting biasmodel under
    **biasmodel_output_path** can then be used when LIQUORICE is run for region-sets of interest.

    :param samplename: Name of the sample (to be used in plots and output tables).
    :param bam_filepath: Path to the .bam file containing the mapped reads for the sample. Does not need to be
        duplicate-filtered, this is done by the function that calculates the coverage.
    :param bed_filepath: path to a .bed file containing regions that should be used to build the biasmodel.
        A .bed file that contains many random regions across the genome is recommended.
    :param refgenome_filepath: Path to the reference genome .fa(.gz) file. Must have a .fai index in the same dirname.
    :param refgenome_chromsizes_filepath: Path to a tab-delimited file containing the chromosome sizes for the
        reference genome. The first column must be the chromosome name, the second column its size.
        Can be the .fa.fai file associated to the reference genome.
    :param refgenome_mappability_bigwig_path: Path to a .bigWig file containing (forward) mappability values scaled
        between 0 and 1.
    :param blacklist_bed_filepath: .bed file of a black list, such as the one from
        `here <https://github.com/Boyle-Lab/Blacklist/blob/master/lists/hg38-blacklist.v2.bed.gz> `_ for hg38
        (unzip first). Regions that overlap any of the regions in this blacklist after extension by **extend_to**
        will be excluded from further analysis. Set to None to skip this step.
    :param sampled_fragment_lengths: A list containing fragment lengths that are representative of the sample's global
        fragment size distribution. Typically a few hundred fragments will suffice here.
    :param avg_readlength: (Average) length of the reads in the .bam file.
    :param mean_seq_depth: A float that quantifies the global coverage/sequencing depth. Coverages per bin will be
        normalized (i.e. divided) by this value.
    :param cna_seg_filepath: If specified, use this .seg file to correct the coverage by the values specified in
       this file prior to model training. Use this if you want to normalize out the effects of copy number aberrations
       (CNAs) on the coverage. File must be tab-separated, with column names as first line. The second, third and fourth
       column must be chromosome, start, and end of the segment, and the last column must be the log2-ratio of
       observed/expected read depth.
    :param binsize: Size of the bins. Use higher values to reduce noise, lower values to increase spatial
        resolution. Using the same binsize for the biasmodel generation as for the region-set-of-interest is probably
        preferable.
    :param extend_to: The regions will be extended by this value in both directions. Outmost bins will have their
        center at *<center of the region>*+-*<extend_to>*. Here, the default is 0, meaning only a single bin will be
        generated per region in the .bed file. This speeds up the computation - for more precise biasmodels, we
        recommend running for a larger set of regions rather than increasing the **extend_to** parameter.
    :param n_cores: Maximum number of cores to use during steps that allow multiprocessing/multithreading.
    :param biasmodel_output_path: Path to which the trained biasmodel should be saved to. Must have a
        .joblib extension.
    :param nr_of_bins_for_training_and_testing: Subset the training_df to this many bins. Can speed up the
        computation time of the model training, but using too few bins will make the model less precise. To speed up
        computations, we would recommend decreasing the number of regions in the .bed file rather than altering this
        parameter, as this is more efficient.
    :param save_training_table: If `True`, save the table that was used to train the biasmodel (coverage and biasfactors
        per bin) as "training_table.csv".
    :param no_chr_prefix: If True, set the list of allowed chromosomes to [str(i) for i in range(1,23)] instead of
        ["chr"+str(i) for i in range(1,23)]
    :param percentile_split_core_rois: If set, split the central region into 5 bins of variable size instead of always
        using a fixed binsize. `extend_to` should not be set to 0 if this is used.
    """

    if binsize % 2:
        raise TypeError("binsize must be a multiple of 2.")

    # Filter regions in the input .bed file containing the regions of interest, extending them, and split them into bins
    prepareBinnedBed = RegionFilteringAndBinning.RegionFilteringAndBinning(
        bed_filepath=bed_filepath,
        binsize=binsize,
        extend_to=extend_to,
        refgenome_chromsizes_filepath=refgenome_chromsizes_filepath,
        chromosomes_list=["chr"+str(i) for i in range(1,23)] if not no_chr_prefix else [str(i) for i in range(1,23)],
        blacklist_bed_filepath=blacklist_bed_filepath)
    if percentile_split_core_rois:
        prepareBinnedBed.write_bin_bedfile_percentile_split_central_roi(
            out_bedfile_path_bins="bins.bed",out_bedfile_path_regions_that_passed_filter="regions.bed")
    else:
        prepareBinnedBed.write_bin_bedfile(out_bedfile_path_bins="bins.bed",
                                           out_bedfile_path_regions_that_passed_filter="regions.bed")

    # Calculate coverage (normalized by sequencing depth), mappability, and genomic sequence for every bin
    liq_table = CoverageAndSequenceTablePreparation.CoverageAndSequenceTablePreparation(
        bam_filepath=bam_filepath,
        bins_bed_filepath="bins.bed",
        refgenome_filepath=refgenome_filepath,
        refgenome_chromsizes_filepath=refgenome_chromsizes_filepath,
        refgenome_mappability_bigwig_path=refgenome_mappability_bigwig_path,
        readlength=avg_readlength,
        longest_fraglen=max(sampled_fragment_lengths),
        mean_seq_depth=mean_seq_depth,
        n_cores=n_cores
    ).get_complete_table()

    # Correct for CNAs if the corresponding file is specified
    if cna_seg_filepath is not None:
        liq_table = CorrectForCNAs.correct_coverage_per_bin_for_cnas(df=liq_table,cna_seg_filepath=cna_seg_filepath,
                                                                     n_cores=n_cores)

    # Calculate the bias factors for each of the bins
    if percentile_split_core_rois:
        liq_table=add_biasfactors_percentile_split(avg_readlength, liq_table, n_cores, sampled_fragment_lengths)
    else:
        liq_table = BinTableBiasFactors.BiasFactorHandler(
            binsize=binsize,
            fragments=sampled_fragment_lengths,
            readlength=avg_readlength,
            df=liq_table,
            n_cores=n_cores
        ).get_table_with_bias_factors()

    if save_training_table:
        logging.info(f"Writing bins and their bias factors to .csv")
        liq_table[[col for col in liq_table.columns if not col in ["sequence","mappability"]]].to_csv(
            "training_table.csv",index=False)

    # Train a bias-model

    biasmodel = BiasModel.BiasModel(
        training_df=liq_table,
        df_to_correct=None,
        biasmodel_path=biasmodel_output_path,
        nr_of_bins_for_training_and_testing=nr_of_bins_for_training_and_testing,
        use_binsize_as_feature=True if percentile_split_core_rois else False
    )
    biasmodel.train_biasmodel()

    # Set up an object for plotting
    plotting = Plotting.Plotting(
        samplename=samplename,
        out_dir=os.getcwd())

    # Plot the association of GC content with the uncorrected coverage
    q5=liq_table["coverage"].quantile(0.05)
    q95=liq_table["coverage"].quantile(0.95)
    plotting.plot_coverage_bias_correlation(df=liq_table, biasfactor_column="GC content",
                                            coverage_column="coverage",
                                            ymin=q5-((q95-q5)/2),
                                            ymax=q95+((q95-q5)/2))

    # # Plot
    # logging.info("Plotting bias factors/coverage/bin nr. associations ...")
    # plotting.plot_correlations_biasfactors_coverage_bin_nr(df=liq_table,y_varnames_list=["coverage"] if not
    # percentile_split_core_rois else ["coverage","bin size"],fit_regression=False)


def run_liquorice_on_regionset_with_pretrained_biasmodel(
    samplename: str,regionset_name: str, bam_filepath: str,
    bed_filepath: str,
    biasmodel_path: str, refgenome_filepath: str,
    refgenome_chromsizes_filepath: str,
    refgenome_mappability_bigwig_path: str,
    blacklist_bed_filepath: typing.Union[None,str],
    sampled_fragment_lengths: typing.List[int],
    avg_readlength: int,
    cna_seg_filepath: typing.Union[None,str],
    mean_seq_depth: float, n_cores: int,
    binsize: int =500, extend_to: int =20000,
    use_default_fixed_sigma_values: bool =True,
    save_biasfactor_table: bool =False,
    save_corrected_coverage_table: bool =False,
    no_chr_prefix: bool=False,
    percentile_split_core_rois:bool=False,
    use_this_roi_biasfactortable:typing.Union[None,str]=None) -> None:
    """
    Run the complete LIQUORICE workflow on a given region-set, using a pre-trained bias model. Main steps of this
    workflow include: Filtering regions in the input .bed and splitting remaining regions into bins; calculating
    sequence, coverage, and mappability for every bin; calculating bias factors for every bin, using the pre-trained
    model and the inferred bias-factors to correct the coverage; aggregating information across regions and fitting
    gaussian functions and an intercept to the corrected, aggregated coverage data. Also creates plots and result
    tables.

    :param samplename: Name of the sample (to be used in plots and output tables).
    :param regionset_name: Name of the region-set (to be used in plots and output tables).
    :param bam_filepath: Path to the .bam file containing the mapped reads for the sample. Does not need to be
        duplicate-filtered, this is done by the function that calculates the coverage.
    :param bed_filepath: path to a .bed file containing regions-of-interest that should be extended and split into bins.
        This could be e.g. a list of DNase I hypersensitivity sites or enhancer peaks.
    :param biasmodel_path: Path to a trained biasmodel.
    :param refgenome_filepath: Path to the reference genome .fa(.gz) file. Must have a .fai index in the same dirname.
    :param refgenome_chromsizes_filepath: Path to a tab-delimited file containing the chromosome sizes for the
        reference genome. The first column must be the chromosome name, the second column its size.
        Can be the .fa.fai file associated to the reference genome.
    :param refgenome_mappability_bigwig_path: Path to a .bigWig file containing (forward) mappability values scaled
        between 0 and 1.
    :param blacklist_bed_filepath: .bed file of a black list, such as the one from
        `here <https://github.com/Boyle-Lab/Blacklist/blob/master/lists/hg38-blacklist.v2.bed.gz> `_ for hg38
        (unzip first). Regions that overlap any of the regions in this blacklist after extension by **extend_to**
        will be excluded from further analysis. Set to None to skip this step.
    :param sampled_fragment_lengths: A list containing fragment lengths that are representative of the sample's global
        fragment size distribution. Typically a few hundred fragments will suffice here.
    :param avg_readlength: (Average) length of the reads in the .bam file.
    :param cna_seg_filepath: If specified, use this .seg file to correct the coverage by the values specified in
       this file prior to correcting coverage with the bias model. Use this if you want to normalize out the effects of
       copy number aberrations (CNAs) on the coverage. File must be tab-separated, with column names as first line.
       The second, third and fourth column must be chromosome, start, and end of the segment, and the last column must
       be the log2-ratio of observed/expected read depth.
    :param mean_seq_depth: A float that quantifies the global coverage/sequencing depth. Coverages per bin will be
        normalized (i.e. divided) by this value.
    :param binsize: Size of the bins. Use higher values to reduce noise, lower values to increase spatial
        resolution. Using the same binsize for the biasmodel generation as for the region-set-of-interest is probably
        preferable.
    :param extend_to: The regions will be extended by this value in both directions. Outmost bins will have their
        center at * <center of the region> * +- * <extend_to> *.
    :param use_default_fixed_sigma_values: If `True`, use the following contraints for the sigma values of the small,
        medium, and large Gaussian, repectively: 149-150 bp, 757-758 bp, and 6078-6079 bp.
    :param n_cores: Maximum number of cores to use during steps that allow multiprocessing/multithreading.
    :param save_biasfactor_table: If `True`, save a table of bin coordinates, bin number, coverage, corrected coverage and biasfactor values under
        "coverage_and_biasfactors_per_bin.csv".
    :param save_corrected_coverage_table: If `True`, save a table of bin coordinates, bin number, coverage, and corrected coverage under
        "coverage_per_bin.csv".
    :param no_chr_prefix: If True, set the list of allowed chromosomes to [str(i) for i in range(1,23)] instead of
        ["chr"+str(i) for i in range(1,23)]
    :param percentile_split_core_rois: If set, split the central region into 5 bins of variable size instead of always
        using a fixed binsize. `extend_to` should not be set to 0 if this is used.
    """

    logging.info(f"###### Starting analysis for {regionset_name} ######")


    #  Filter regions in the input .bed file containing the regions of interest, extending them, and split them
    #  into bins
    prepareBinnedBed = RegionFilteringAndBinning.RegionFilteringAndBinning(
        bed_filepath=bed_filepath,
        binsize=binsize,
        extend_to=extend_to,
        refgenome_chromsizes_filepath=refgenome_chromsizes_filepath,
        chromosomes_list=["chr"+str(i) for i in range(1,23)] if not no_chr_prefix else [str(i) for i in range(1,23)],
        blacklist_bed_filepath=blacklist_bed_filepath)
    if percentile_split_core_rois:
        prepareBinnedBed.write_bin_bedfile_percentile_split_central_roi(
            out_bedfile_path_bins="bins.bed",out_bedfile_path_regions_that_passed_filter="regions.bed")
    else:
        prepareBinnedBed.write_bin_bedfile(out_bedfile_path_bins="bins.bed",
                                           out_bedfile_path_regions_that_passed_filter="regions.bed")
    regions_df=pd.read_csv("regions.bed",sep="\t",header=None)
    avg_centersize=int(round((regions_df[2]-regions_df[1]).mean())) # required downstream

    if not use_this_roi_biasfactortable:

        #  Calculate coverage (normalized by sequencing depth), mappability, and genomic sequence for every bin
        liq_table = CoverageAndSequenceTablePreparation.CoverageAndSequenceTablePreparation(
            bam_filepath=bam_filepath,
            bins_bed_filepath="bins.bed",
            refgenome_filepath=refgenome_filepath,
            refgenome_chromsizes_filepath=refgenome_chromsizes_filepath,
            refgenome_mappability_bigwig_path=refgenome_mappability_bigwig_path,
            readlength=avg_readlength,
            longest_fraglen=max(sampled_fragment_lengths),
            n_cores=n_cores,
            mean_seq_depth=mean_seq_depth
        ).get_complete_table()

        # Correct for CNAs if the corresponding file is specified
        if cna_seg_filepath is not None:
            liq_table = CorrectForCNAs.correct_coverage_per_bin_for_cnas(df=liq_table,cna_seg_filepath=cna_seg_filepath,
                                                                         n_cores=n_cores)

        # Calculate the bias factors for each of the bins
        if percentile_split_core_rois:
            liq_table=add_biasfactors_percentile_split(avg_readlength, liq_table, n_cores, sampled_fragment_lengths)
        else:
            liq_table = BinTableBiasFactors.BiasFactorHandler(
                binsize=binsize,
                fragments=sampled_fragment_lengths,
                readlength=avg_readlength,
                df=liq_table,
                n_cores=n_cores
            ).get_table_with_bias_factors()

    elif use_this_roi_biasfactortable:
        liq_table=pd.read_csv(use_this_roi_biasfactortable)
        if "corrected coverage" in liq_table.columns:
            liq_table=liq_table.drop("corrected coverage",axis=1)

    # Use the pre-trained bias-model to correct coverage
    biasmodel = BiasModel.BiasModel(
        training_df=None,
        df_to_correct=liq_table,
        biasmodel_path=biasmodel_path,
        use_binsize_as_feature=True if percentile_split_core_rois else False
    )
    liq_table=biasmodel.get_table_with_corrected_coverage_using_trained_biasmodel()

    # write csv
    if save_biasfactor_table:
        logging.info(f"Writing bins and their bias factors to .csv")
        liq_table[[col for col in liq_table.columns if not col in ["sequence","mappability"]]].to_csv(
            "coverage_and_biasfactors_per_bin.csv",index=False)
    if save_corrected_coverage_table:
        logging.info(f"Writing bins and their bias factors to .csv")
        liq_table[["chromosome","start","end","bin nr.","coverage","corrected coverage"]].to_csv(
            "coverage_per_bin.csv",index=False)

    # Set up an object for plotting
    plotting = Plotting.Plotting(
        samplename=samplename,
        out_dir=os.getcwd())

    # Plot the association of GC content with corrected and uncorrected coverage
    q5=liq_table["coverage"].quantile(0.05)
    q95=liq_table["coverage"].quantile(0.95)
    plotting.plot_coverage_bias_correlation(df=liq_table, biasfactor_column="GC content",
                                            coverage_column="coverage",
                                            ymin=q5-((q95-q5)/2),
                                            ymax=q95+((q95-q5)/2))
    q5=liq_table["corrected coverage"].quantile(0.05)
    q95=liq_table["corrected coverage"].quantile(0.95)
    plotting.plot_coverage_bias_correlation(df=liq_table, biasfactor_column="GC content",
                                            coverage_column="corrected coverage",
                                            ymin=q5-((q95-q5)/2),
                                            ymax=q95+((q95-q5)/2))

    # logging.info("Plotting bias factors/coverage/bin nr. associations ...")
    # plotting.plot_correlations_biasfactors_coverage_bin_nr(df=liq_table,y_varnames_list=None if not
    #     percentile_split_core_rois else ["corrected coverage","coverage","bin nr.","bin size"])

    # Aggregate coverage information across regions
    aggregated_corrected_coverage=AggregateAcrossRegions.aggregate_across_regions(liq_table,
                                                                                  "corrected coverage")
    aggregated_corrected_coverage.to_csv("corrected_coverage_mean_per_bin.csv")
    aggregated_uncorrected_coverage=AggregateAcrossRegions.aggregate_across_regions(liq_table,"coverage")
    aggregated_uncorrected_coverage.to_csv("uncorrected_coverage_mean_per_bin.csv")

    # Default sigma contraints:
    default_sigma_contraints={
        "g1_min_sigma":149,
        "g1_max_sigma":149+1,
        "g2_min_sigma":757,
        "g2_max_sigma":757+1,
        "g3_min_sigma":6078,
        "g3_max_sigma":6078+1}

    if use_default_fixed_sigma_values:
        sigma_contraints=default_sigma_contraints
    else:
        sigma_contraints={}

    # Fit gaussian functions to the aggregated corrected coverage
    fit_gaussians_obj = FitGaussians.FitGaussians(
        unfitted_y_values=aggregated_corrected_coverage,
        extend_to=extend_to,
        binsize=binsize,
        samplename=samplename,
        regionset_name=regionset_name,
        avg_centersize=avg_centersize)
    fit_df=fit_gaussians_obj.fit_gaussian_models(**sigma_contraints)
    fit_df.to_csv("fitted_gaussians_parameter_summary.csv",index=False)
    plotting.plot_fitted_model(fit_gaussians_obj=fit_gaussians_obj)

    #  Fit gaussian functions to the aggregated uncorrected coverage
    fit_gaussians_obj_uncorrected = FitGaussians.FitGaussians(
        unfitted_y_values=aggregated_uncorrected_coverage,
        extend_to=extend_to,
        binsize=binsize,
        samplename=samplename,
        regionset_name=regionset_name,
        avg_centersize=avg_centersize)
    #  Use (practically) the same sigma values as for the corrected coverage to keep things comparable
    fit_df_uncorrected=fit_gaussians_obj_uncorrected.fit_gaussian_models(g1_min_sigma=int(fit_df["G1_sigma"]),
                                                                         g1_max_sigma=int(fit_df["G1_sigma"])+1,
                                                                         g2_min_sigma=int(fit_df["G2_sigma"]),
                                                                         g2_max_sigma=int(fit_df["G2_sigma"])+1,
                                                                         g3_min_sigma=int(fit_df["G3_sigma"]),
                                                                         g3_max_sigma=int(fit_df["G3_sigma"])+1)
    fit_df_uncorrected.to_csv("uncorrected_fitted_gaussians_parameter_summary.csv",index=False)

    # Compare corrected vs uncorrected coverage
    plotting.plot_corrected_vs_uncorrected_coverage(
        x_values=fit_gaussians_obj.x_values,
        uncorrected_y_values=fit_gaussians_obj_uncorrected.unfitted_y_values,
        corrected_y_values=fit_gaussians_obj.unfitted_y_values,
        uncorrected_y_intercept=fit_gaussians_obj_uncorrected.intercept_y_values[0],
        corrected_y_intercept=fit_gaussians_obj.intercept_y_values[0])


def run_liquorice_train_biasmodel_on_same_regions(
        samplename: str,regionset_name: str, bam_filepath: str,
        bed_filepath: str, refgenome_filepath: str,
        refgenome_chromsizes_filepath: str,
        refgenome_mappability_bigwig_path: str,
        blacklist_bed_filepath: typing.Union[None,str],
        sampled_fragment_lengths: typing.List[int],
        avg_readlength: int,
        cna_seg_filepath: typing.Union[None,str],
        mean_seq_depth: float, n_cores: int,
        binsize: int =500, extend_to: int =20000,
        biasmodel_output_path: str = "trained_biasmodel.joblib",
        nr_of_bins_for_training_and_testing: typing.Union[None,int] = 10000,
        skip_central_n_bins_for_training:int= 0,
        save_training_table: bool = False,
        use_default_fixed_sigma_values: bool =True,
        save_biasfactor_table: bool =False,
        save_corrected_coverage_table: bool =False,
        no_chr_prefix: bool=False,
        percentile_split_core_rois:bool=False,
        use_cross_validated_predictions:bool=False,
        use_this_roi_biasfactortable:typing.Union[None,str]=None,
        speed_mode:bool=False) -> None:
    """
    Run the complete LIQUORICE workflow on a given region-set, and train the bias-model on data from the same region-
    set. Main steps of this workflow include: Filtering regions in the input .bed and splitting remaining regions into
    bins; calculating
    sequence, coverage, and mappability for every bin; calculating bias factors for every bin, training a bias model,
    using the trained
    model and the inferred bias-factors to correct the coverage; aggregating information across regions and fitting
    gaussian functions and an intercept to the corrected, aggregated coverage data. Also creates plots and result
    tables.

    :param samplename: Name of the sample (to be used in plots and output tables).
    :param regionset_name: Name of the region-set (to be used in plots and output tables).
    :param bam_filepath: Path to the .bam file containing the mapped reads for the sample. Does not need to be
        duplicate-filtered, this is done by the function that calculates the coverage.
    :param bed_filepath: path to a .bed file containing regions-of-interest that should be extended and split into bins.
        This could be e.g. a list of DNase I hypersensitivity sites or enhancer peaks.
    :param refgenome_filepath: Path to the reference genome .fa(.gz) file. Must have a .fai index in the same dirname.
    :param refgenome_chromsizes_filepath: Path to a tab-delimited file containing the chromosome sizes for the
        reference genome. The first column must be the chromosome name, the second column its size.
        Can be the .fa.fai file associated to the reference genome.
    :param refgenome_mappability_bigwig_path: Path to a .bigWig file containing (forward) mappability values scaled
        between 0 and 1.
    :param blacklist_bed_filepath: .bed file of a black list, such as the one from
        `here <https://github.com/Boyle-Lab/Blacklist/blob/master/lists/hg38-blacklist.v2.bed.gz> `_ for hg38
        (unzip first). Regions that overlap any of the regions in this blacklist after extension by **extend_to**
        will be excluded from further analysis. Set to None to skip this step.
    :param sampled_fragment_lengths: A list containing fragment lengths that are representative of the sample's global
        fragment size distribution. Typically a few hundred fragments will suffice here.
    :param avg_readlength: (Average) length of the reads in the .bam file.
    :param cna_seg_filepath: If specified, use this .seg file to correct the coverage by the values specified in
       this file prior to correcting coverage with the bias model. Use this if you want to normalize out the effects of
       copy number aberrations (CNAs) on the coverage. File must be tab-separated, with column names as first line.
       The second, third and fourth column must be chromosome, start, and end of the segment, and the last column must
       be the log2-ratio of observed/expected read depth.
    :param mean_seq_depth: A float that quantifies the global coverage/sequencing depth. Coverages per bin will be
        normalized (i.e. divided) by this value.
    :param binsize: Size of the bins. Use higher values to reduce noise, lower values to increase spatial
        resolution. Using the same binsize for the biasmodel generation as for the region-set-of-interest is probably
        preferable.
    :param extend_to: The regions will be extended by this value in both directions. Outmost bins will have their
        center at * <center of the region> * +- * <extend_to> *.
    :param use_default_fixed_sigma_values: If `True`, use the following contraints for the sigma values of the small,
        medium, and large Gaussian, repectively: 149-150 bp, 757-758 bp, and 6078-6079 bp.
    :param n_cores: Maximum number of cores to use during steps that allow multiprocessing/multithreading.
    :param save_biasfactor_table: If `True`, save a table of bin coordinates, bin number, coverage, corrected coverage and biasfactor values under
        "coverage_and_biasfactors_per_bin.csv".
    :param save_corrected_coverage_table: If `True`, save a table of bin coordinates, bin number, coverage, and corrected coverage under
        "coverage_per_bin.csv".
    :param no_chr_prefix: If True, set the list of allowed chromosomes to [str(i) for i in range(1,23)] instead of
        ["chr"+str(i) for i in range(1,23)]
    :param biasmodel_output_path: Path to which the trained biasmodel should be saved to. Must have a
        .joblib extension.
    :param nr_of_bins_for_training_and_testing: Subset the training_df to this many bins. Can speed up the
        computation time of the model training, but using too few bins will make the model less precise. To speed up
        computations, we would recommend decreasing the number of regions in the .bed file rather than altering this
        parameter, as this is more efficient.
    :param save_training_table: If `True`, save the table that was used to train the biasmodel (coverage and biasfactors
        per bin) as "training_table.csv".
    :param skip_central_n_bins_for_training: The n most central bins will not be used for training the bias model.
    :param no_chr_prefix: If True, set the list of allowed chromosomes to [str(i) for i in range(1,23)] instead of
        ["chr"+str(i) for i in range(1,23)]
    :param percentile_split_core_rois: If set, split the central region into 5 bins of variable size instead of always
        using a fixed binsize. `extend_to` should not be set to 0 if this is used.
    :param use_cross_validated_predictions: Instead of training on the full dataset, train twice on half of the dataset
        and predict the other half. Ignores nr_of_bins_for_training_and_testing
    :param use_this_roi_biasfactortable: If set use the specified biasfactor table and only train/apply the biasmodel.
    :param speed_mode: Only perform GC correction, don't correct using mappability or di/trinucleotides. Setting this
        flag makes LIQUORICE considerably faster, but may lead to less accurate results.

    """

    logging.info(f"###### Starting analysis for {regionset_name} ######")
    if speed_mode:
        logging.warning(f"Speed mode is active. Will only calculate & correct GC bias, ignoring the other biases.")

    #  Filter regions in the input .bed file containing the regions of interest, extending them, and split them
    #  into bins
    prepareBinnedBed = RegionFilteringAndBinning.RegionFilteringAndBinning(
        bed_filepath=bed_filepath,
        binsize=binsize,
        extend_to=extend_to,
        refgenome_chromsizes_filepath=refgenome_chromsizes_filepath,
        chromosomes_list=["chr"+str(i) for i in range(1,23)] if not no_chr_prefix else [str(i) for i in range(1,23)],
        blacklist_bed_filepath=blacklist_bed_filepath)
    if percentile_split_core_rois:
        prepareBinnedBed.write_bin_bedfile_percentile_split_central_roi(out_bedfile_path_bins="bins.bed",
                                                                        out_bedfile_path_regions_that_passed_filter="regions.bed")
    else:
        prepareBinnedBed.write_bin_bedfile(out_bedfile_path_bins="bins.bed",
                                           out_bedfile_path_regions_that_passed_filter="regions.bed")
    regions_df=pd.read_csv("regions.bed",sep="\t",header=None)
    avg_centersize=int(round((regions_df[2]-regions_df[1]).mean())) # required downstream

    if not use_this_roi_biasfactortable:
        #  Calculate coverage (normalized by sequencing depth), mappability, and genomic sequence for every bin
        liq_table = CoverageAndSequenceTablePreparation.CoverageAndSequenceTablePreparation(
            bam_filepath=bam_filepath,
            bins_bed_filepath="bins.bed",
            refgenome_filepath=refgenome_filepath,
            refgenome_chromsizes_filepath=refgenome_chromsizes_filepath,
            refgenome_mappability_bigwig_path=refgenome_mappability_bigwig_path,
            readlength=avg_readlength,
            longest_fraglen=max(sampled_fragment_lengths),
            n_cores=n_cores,
            mean_seq_depth=mean_seq_depth,
            skip_these_steps=[] if not speed_mode else ["mappability"]
        ).get_complete_table()


        # Correct for CNAs if the corresponding file is specified
        if cna_seg_filepath is not None:
            liq_table = CorrectForCNAs.correct_coverage_per_bin_for_cnas(df=liq_table,cna_seg_filepath=cna_seg_filepath,
                                                                         n_cores=n_cores)

        # Calculate the bias factors for each of the bins
        if percentile_split_core_rois:
            liq_table=add_biasfactors_percentile_split(avg_readlength, liq_table, n_cores,
                                                                 sampled_fragment_lengths,skip_these_biasfactors=["di and trinucleotides","mappability"] if
                speed_mode else [])
        else:
            liq_table = BinTableBiasFactors.BiasFactorHandler(
                binsize=binsize,
                fragments=sampled_fragment_lengths,
                readlength=avg_readlength,
                df=liq_table,
                n_cores=n_cores,
                skip_these_biasfactors=["di and trinucleotides","mappability"] if speed_mode else []
            ).get_table_with_bias_factors()

        if save_training_table:
            logging.info(f"Writing bins and their bias factors to .csv")
            liq_table[[col for col in liq_table.columns if not col in ["sequence","mappability"]]].to_csv(
                "training_table.csv",index=False)

    elif use_this_roi_biasfactortable:
        liq_table=pd.read_csv(use_this_roi_biasfactortable)

    # Train a bias-model
    strt_idx = (liq_table["bin nr."].unique().shape[0] // 2) - (skip_central_n_bins_for_training // 2)
    end_idx = (liq_table["bin nr."].unique().shape[0] // 2) + (skip_central_n_bins_for_training // 2)
    n_central_binnrs=list(range(strt_idx,end_idx + 1)) if skip_central_n_bins_for_training else []
    logging.info(f"Skipping the {skip_central_n_bins_for_training} central bins (numbers {n_central_binnrs})"
                 f" for training of the bias-model.")

    biasmodel = BiasModel.BiasModel(
        training_df=liq_table[~ liq_table["bin nr."].isin(n_central_binnrs)],
        df_to_correct=liq_table,
        biasmodel_path=biasmodel_output_path,
        nr_of_bins_for_training_and_testing=nr_of_bins_for_training_and_testing
    )
    if not use_cross_validated_predictions:

        biasmodel.train_biasmodel()
        # Use the pre-trained bias-model to correct coverage
        liq_table_with_corr_cov=biasmodel.get_table_with_corrected_coverage_using_trained_biasmodel()
    else:
        liq_table_with_corr_cov=biasmodel.train_biasmodel_2fold_CV_and_predict(exclude_these_bin_nrs=n_central_binnrs)

    # write csv
    if save_biasfactor_table:
        logging.info(f"Writing bins and their bias factors to .csv")
        liq_table_with_corr_cov[[col for col in liq_table_with_corr_cov.columns if not col in ["sequence","mappability"]]].to_csv(
            "coverage_and_biasfactors_per_bin.csv",index=False)
    if save_corrected_coverage_table:
        logging.info(f"Writing bins and their bias factors to .csv")
        liq_table_with_corr_cov[["chromosome","start","end","bin nr.","coverage","corrected coverage"]].to_csv(
            "coverage_per_bin.csv",index=False)

    # Set up an object for plotting
    plotting = Plotting.Plotting(
        samplename=samplename,
        out_dir=os.getcwd())

    # Plot the association of GC content with corrected and uncorrected coverage
    q5=liq_table_with_corr_cov["coverage"].quantile(0.05)
    q95=liq_table_with_corr_cov["coverage"].quantile(0.95)
    plotting.plot_coverage_bias_correlation(df=liq_table_with_corr_cov, biasfactor_column="GC content",
                                            coverage_column="coverage",
                                            ymin=q5-((q95-q5)/2),
                                            ymax=q95+((q95-q5)/2))
    q5=liq_table_with_corr_cov["corrected coverage"].quantile(0.05)
    q95=liq_table_with_corr_cov["corrected coverage"].quantile(0.95)
    plotting.plot_coverage_bias_correlation(df=liq_table_with_corr_cov, biasfactor_column="GC content",
                                            coverage_column="corrected coverage",
                                            ymin=q5-((q95-q5)/2),
                                            ymax=q95+((q95-q5)/2))

    #   logging.info("Plotting bias factors/coverage/bin nr. associations ...")
    #   plotting.plot_correlations_biasfactors_coverage_bin_nr(df=liq_table_with_corr_cov)

    # Aggregate coverage information across regions
    aggregated_corrected_coverage=AggregateAcrossRegions.aggregate_across_regions(liq_table_with_corr_cov,
                                                                                  "corrected coverage")
    aggregated_corrected_coverage.to_csv("corrected_coverage_mean_per_bin.csv")
    aggregated_uncorrected_coverage=AggregateAcrossRegions.aggregate_across_regions(liq_table_with_corr_cov,"coverage")
    aggregated_uncorrected_coverage.to_csv("uncorrected_coverage_mean_per_bin.csv")

    # Default sigma contraints:
    default_sigma_contraints={
        "g1_min_sigma":149,
        "g1_max_sigma":149+1,
        "g2_min_sigma":757,
        "g2_max_sigma":757+1,
        "g3_min_sigma":6078,
        "g3_max_sigma":6078+1}

    if use_default_fixed_sigma_values:
        sigma_contraints=default_sigma_contraints
    else:
        sigma_contraints={}

    # Fit gaussian functions to the aggregated corrected coverage
    fit_gaussians_obj = FitGaussians.FitGaussians(
        unfitted_y_values=aggregated_corrected_coverage,
        extend_to=extend_to,
        binsize=binsize,
        samplename=samplename,
        regionset_name=regionset_name,
        avg_centersize=avg_centersize)
    fit_df=fit_gaussians_obj.fit_gaussian_models(**sigma_contraints)
    fit_df.to_csv("fitted_gaussians_parameter_summary.csv",index=False)
    plotting.plot_fitted_model(fit_gaussians_obj=fit_gaussians_obj)

    #  Fit gaussian functions to the aggregated uncorrected coverage
    fit_gaussians_obj_uncorrected = FitGaussians.FitGaussians(
        unfitted_y_values=aggregated_uncorrected_coverage,
        extend_to=extend_to,
        binsize=binsize,
        samplename=samplename,
        regionset_name=regionset_name,
        avg_centersize=avg_centersize)
    #  Use (practically) the same sigma values as for the corrected coverage to keep things comparable
    fit_df_uncorrected=fit_gaussians_obj_uncorrected.fit_gaussian_models(g1_min_sigma=int(fit_df["G1_sigma"]),
                                                                         g1_max_sigma=int(fit_df["G1_sigma"])+1,
                                                                         g2_min_sigma=int(fit_df["G2_sigma"]),
                                                                         g2_max_sigma=int(fit_df["G2_sigma"])+1,
                                                                         g3_min_sigma=int(fit_df["G3_sigma"]),
                                                                         g3_max_sigma=int(fit_df["G3_sigma"])+1)
    fit_df_uncorrected.to_csv("uncorrected_fitted_gaussians_parameter_summary.csv",index=False)

    # Compare corrected vs uncorrected coverage
    plotting.plot_corrected_vs_uncorrected_coverage(
        x_values=fit_gaussians_obj.x_values,
        uncorrected_y_values=fit_gaussians_obj_uncorrected.unfitted_y_values,
        corrected_y_values=fit_gaussians_obj.unfitted_y_values,
        uncorrected_y_intercept=fit_gaussians_obj_uncorrected.intercept_y_values[0],
        corrected_y_intercept=fit_gaussians_obj.intercept_y_values[0])

