#import deeptools.countReadsPerBin as crpb
from liquorice.utils import deeptoolsCountReadsPerBinDontCheckProperPairSAMflag as crpb
import logging
import pandas as pd
import typing
import os


def sample_bam_to_get_sequencing_depth(bam_filepath: str, n_sites_to_sample: int = 10000, n_cores: int = 1,
                                       min_mapq: int = 20,
                                       chromosomes_list: typing.List[str] = None,
                                       **additional_crpb_kwargs: dict) -> float:
    """
    Estimates the overall sequencing coverage of the .bam file by sampling
    sites that are regularily distributed across the genome (using :func:`deeptools.countReadsPerBin.CountReadsPerBin`
    in a slightly modified version that allows fragments without the SAM flag is_proper_pair.).


    :param bam_filepath: Path to the .bam file for which mean coverage should be calculated.
    :param n_sites_to_sample: Number of sites (length 1) to sample.
    :param n_cores: Number of cores to be used by :func:`deeptools.countReadsPerBin.CountReadsPerBin`.
    :param min_mapq: `minMappingQuality` setting for `deeptools.countReadsPerBin.CountReadsPerBin`.
    :param chromosomes_list: A list of chromosomes that should be analyzed. Regions on chromosomes that are not in
        this list will be excluded from analysis. Default None means ["chr1", ..., "chr22"].
    :param additional_crpb_kwargs: Use to override the default parameters for
        :func:`deeptools.countReadsPerBin.CountReadsPerBin`. If not specified otherwise in `additional_crpb_kwargs`,
        the following attributes are passed to `CountReadsPerBin`: `ignoreDuplicates=True`, `samFlag_exclude=256`,
        `extendReads=True`.

    :return: The average coverage determined from the sampled regions.
    """

    if chromosomes_list is None:
        chromosomes_list=["chr"+str(i) for i in range(1,23)]

    crpb_kwargs={
        "minMappingQuality":min_mapq,
        "ignoreDuplicates":True,
        "numberOfProcessors":n_cores,
        "samFlag_exclude":256,  # not primary alignment
        "extendReads":True}
    crpb_kwargs.update(additional_crpb_kwargs)

    logging.info(f"Calculating coverage at {n_sites_to_sample} regions"
                 f" regularily distributed across the genome to determine overall coverage ...")
    crpb_obj = crpb.CountReadsPerBin(
        [bam_filepath],
        binLength=1,
        numberOfSamples=n_sites_to_sample,
        out_file_for_raw_data="sequencing_depth_at_sampled_regions.tsv",
        **crpb_kwargs)

    crpb_obj.run()

    seqdepth_df=pd.read_csv("sequencing_depth_at_sampled_regions.tsv",sep="\t",header=None)
    seqdepth_df.columns=["chr","start","end","coverage"]
    seqdepth_df["chr"]=seqdepth_df["chr"].astype("str")
    mean_seq_depth=seqdepth_df[seqdepth_df["chr"].isin(chromosomes_list)]["coverage"].mean()
    os.remove("sequencing_depth_at_sampled_regions.tsv")
    # mean_seq_depth=cr.run().mean(axis=0)[0]

    logging.info(f"Mean coverage at the {n_sites_to_sample} regions of length 1 is "
                 f"{mean_seq_depth}.")
    return mean_seq_depth
