import pandas as pd
import sys
import logging
import numpy as np
import os
os.environ["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # required by Ray, which is
# used by modin
import modin.pandas as modinpd
import swifter

bin_nr_without_unique_matching_segment=0

def correct_coverage_per_bin_for_cnas(df: pd.DataFrame,cna_seg_filepath:str,n_cores:int = 1) -> pd.DataFrame:
    """
    Extracts the log2(observed/expected) read depth ratio for the corresponding segment for each bin, and corrects each
    bin's coverage accordingly.

    :param df: A `pandas.DataFrame` with at least the following columns: "chromosome", "start","end","coverage"
    :param cna_seg_filepath: A .seg file that should be used to correct the coverage by the values specified in
       this file.  File must be tab-separated, with column names as first line. The second, third and fourth
       column must be chromosome, start, and end of the segment, and the last column must be the log2-ratio of
       observed/expected read depth.
    :param n_cores: How many cores to use to parallelize the operation.
    :return: A `pandas.DataFrame` similar to the input *df*, with the values in the column
        "coverage" changed according to the CNA correction factor of each bin, and an additional column
        "CNA-uncorrected coverage" that contains the original coverage values.
    """
    logging.info(f"Correcting the coverage for CNAs, using '{cna_seg_filepath}' ...")
    try:
        seg_df=pd.read_csv(cna_seg_filepath,sep="\t")
        seg_df=seg_df[seg_df.columns[[1,2,3,-1]]]
        seg_df.columns=["chromosome", "start","end","1/log2_correction_factor"]
        seg_df["chromosome"]=seg_df["chromosome"].astype("str")

    except IndexError:
        sys.exit(f"Error: There is something wrong with the cna_seg_filepath (or --cna_seg_file) '{cna_seg_filepath}'. "
                 "The file must be tab-separated, with column names as first line. The second, third and fourth "
                 "column must be chromosome, start, and end of the segment, and the last column must be the log2-ratio "
                 "of observed/expected read depth.")

    df["CNA-uncorrected coverage"]=df["coverage"].values
    global bin_nr_without_unique_matching_segment
    bin_nr_without_unique_matching_segment=0
    if n_cores==1:
        df["coverage"]=df.apply(lambda row: get_CNV_corrected_coverage_for_bin(bin_chrom=row["chromosome"],
                                                                               bin_start=row["start"],
                                                                               bin_end=row["end"],
                                                                               bin_coverage=row["coverage"],
                                                                               seg_df=seg_df), axis=1)
    else:
        df["coverage"]=df.swifter.progress_bar(False).set_npartitions(
            n_cores).apply(lambda row: get_CNV_corrected_coverage_for_bin(bin_chrom=row["chromosome"],
                                                                          bin_start=row["start"],
                                                                          bin_end=row["end"],
                                                                          bin_coverage=row["coverage"],
                                                                          seg_df=seg_df), axis=1)
    if bin_nr_without_unique_matching_segment:
        logging.info(f"Could not find a unique segment for CNA-correction for {bin_nr_without_unique_matching_segment} "
                     f" bins ({round(100*(bin_nr_without_unique_matching_segment/df.shape[0]),1)}% of total bins)."
                     f" This happens if a bin overlaps two different segments, "
                     f"the bin is in a region that is blacklisted/excluded by the CNA detection program, "
                     f"or if the .seg file is incomplete. CNA correction was skipped for these bins.")
    return df


def get_CNV_corrected_coverage_for_bin(bin_chrom: str, bin_start:int, bin_end:int, bin_coverage:float, seg_df: pd.DataFrame) -> float:
    """
    For a given bin, returns the coverage corrected for copy number aberrations.

    :param bin_chrom: Chromosome of the bin.
    :param bin_start: Genomic start position of the bin.
    :param bin_end: Genomic end position of the bin.
    :param bin_coverage: Bin's coverage before CNA correction
    :param seg_df: A `pandas.DataFrame` of a .seg file, with the following columns: "chromosome", "start","end",
        "1/log2_correction_factor".
    :return: A float indicating the CNV-corrected coverage for the bin.
    """
    global bin_nr_without_unique_matching_segment
    seg_df=seg_df[(seg_df["chromosome"]==bin_chrom) & (seg_df["start"]<bin_start) & (seg_df["end"]>bin_end)]
    if seg_df.shape[0]!=1:
        bin_nr_without_unique_matching_segment+=1
        logging.debug(f"Could not find a unique segment for CNA-correction for bin with coordinates "
                        f"{bin_chrom}:{bin_start}-{bin_end}. This can happen if a bin overlaps two different segments,"
                        f"the bin is in a region that is blacklisted/excluded by the CNA detection program,"
                        f"or if the .seg file is incomplete. Skipping CNA correction for this bin.")
        return bin_coverage
    else:
        return bin_coverage/np.power(2,seg_df["1/log2_correction_factor"].values[0])