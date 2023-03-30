import numpy as np
import pandas as pd
import pybedtools
import os
if os.getenv("TMPDIR") is not None:
    pybedtools.set_tempdir(os.getenv("TMPDIR"))
import logging
import sys
import typing

class RegionFilteringAndBinning:
    """
    A Binning object, which can be used to create a .bed file with the coordinates and bin numbers of every
    bin. The method :func:`.write_bin_bedfile` of this object can be used to create output.

    :param bed_filepath: path to a .bed file containing regions that should be extended and split into bins.
        This could be e.g. a list of DNase I hypersensitivity sites or enhancer peaks.
    :param binsize: Size of the bins. Use higher values to reduce noise, lower values to increase spatial
        resolution.
    :param extend_to: The regions will be extended by this value in both directions. Outmost bins will have their
        center at *<center of the region>*+-*<extend_to>*.
    :param refgenome_chromsizes_filepath: Path to a tab-delimited file containing the chromosome sizes for the
        reference genome. The first column must be the chromosome name, the second column its size.
        Can be the .fa.fai file associated to the reference genome. Must include all chromosomes given in
        chromosomes_list.
    :param chromosomes_list: A list of chromosomes that should be analyzed. Regions on chromosomes that are not in
        this list will be excluded from analysis. Default is ["chr1", ..., "chr22"].
    :param blacklist_bed_filepath: Optional: .bed file of a black list, such as the one from
        `here <https://github.com/Boyle-Lab/Blacklist/blob/master/lists/hg38-blacklist.v2.bed.gz> `_ for hg38 
        (unzip first). Regions that overlap any of the regions in this blacklist after extension by **extend_to** 
        will be excluded from further analysis.
    """
    # TODO check the N-content of the extended region..

    def __init__(self, bed_filepath: str, binsize: int, extend_to: int, refgenome_chromsizes_filepath: str,
                 chromosomes_list: typing.List[str] = ["chr"+str(i) for i in range(1,23)],
                 blacklist_bed_filepath: typing.Union[None,str] = None) -> None:

        self.bed_filepath = bed_filepath
        self.binsize = binsize
        self.extend_to = extend_to
        self.refgenome_chromsizes_filepath = refgenome_chromsizes_filepath
        self.chromosomes_list = chromosomes_list
        self.blacklist_bed_filepath = blacklist_bed_filepath

    def is_within_chromsome_borders(self, chrom: str, start: int, end: int,
                                    chromsize_dict: typing.Dict[str,int]) -> bool:
        """
        Check if a region is within its chromosome's borders.
        
        :param chrom: Chromosome name
        :param start: Start coordinate
        :param end: End coordinate
        :param chromsize_dict: Dictionary with chromosome names as keys and lengths as values. MUST contain **chrom** 
            as a key, otherwise `sys.exit()` is called.
        :return: True if the region is within the borders of the chromosome, False otherwise
        """
        if chrom not in chromsize_dict:
            logging.error(f"Chromosome {chrom} does not have a corresponding chromosome-size entry. Aborting.")
            sys.exit()
        if start > 0 and end < chromsize_dict[chrom]:
            return True
        else:
            return False

    def filter_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove regions that do not pass filters.
        
        :param df: A `pandas.DataFrame` of a .bed file (with columns "chromosome", "start", "end")
        :return: Filtered `DataFrame`, with regions excluded if (after extension by :attr:`.extend_by`) they :
            i) are not on standard chromosomes (:attr:`.chromosomes_list`), ii) fall within
            :attr:`.blacklist_bed_filepath`
            (if defined), iii) extend beyond chromosome ends, or iv) if their start coordinate is larger than their end
            coordinate.

        """
        # Sanity check: start < end
        n_regions_before = df.shape[0]
        df = df[df["start"] < df["end"]]
        n_removed = n_regions_before - df.shape[0]
        if n_removed:
            logging.warning(f"{n_removed} regions were excluded because their start coordinate was "
                            "smaller than their end coordinate")

        # filter for chromosomes_list
        n_regions_before = df.shape[0]
        df = df[df["chromosome"].isin(self.chromosomes_list)]
        n_removed = n_regions_before - df.shape[0]
        if n_removed:
            logging.warning(f"{n_removed} regions were excluded because they were not on the allowed"
                            f" chromosomes {self.chromosomes_list}")

        # filter for chromosome size
        n_regions_before = df.shape[0]
        chromlenghts = {}
        with open(self.refgenome_chromsizes_filepath) as f:
            for line in f:
                chrom, length = line.split()[0:2]
                chromlenghts[chrom] = int(length)
        within_chromends = df.apply(
            lambda row: self.is_within_chromsome_borders(row["chromosome"], row["start"] - self.extend_to,
                                                         row["end"] + self.extend_to, chromlenghts), axis=1)
        df = df[within_chromends]
        n_removed = n_regions_before - df.shape[0]
        if n_removed:
            logging.warning(f"{n_removed} regions were excluded because they exeeded the "
                            f"chromosome borders after extension by {self.extend_to} bp.")

        # filter for blacklist
        if self.blacklist_bed_filepath:
            n_regions_before = df.shape[0]
            blacklist_bedtool = pybedtools.bedtool.BedTool(self.blacklist_bed_filepath)
            not_within_blacklist = df.apply(lambda row: not blacklist_bedtool.any_hits(
                pybedtools.create_interval_from_list(
                    [row["chromosome"], row["start"] - self.extend_to, row["end"] + self.extend_to])), axis=1)
            df = df[not_within_blacklist]
            n_removed = n_regions_before - df.shape[0]
            if n_removed:
                logging.warning(f"{n_removed} regions were excluded because they were within the "
                                f"blacklist '{self.blacklist_bed_filepath}' after extension by {self.extend_to} bp.")
        pybedtools.cleanup()
        return df

    def get_binstarts(self, center: int) -> typing.List[int]:
        """
        Gets the bin start coordinates for a given center

        :param center: Coordinate of the center of the region of interest
        :return: A list with start coordinates, one per bin. Length / number of created bins
            depends on :attr:`.extend_to` and :attr:`.binsize`.
        """
        upstream = list(
            range(int(center + self.binsize / 2), int(center + self.binsize / 2 + self.extend_to), self.binsize))
        downstream = [(center - (coord - center)) - self.binsize for coord in upstream[::-1]]
        center_bin = int(center - self.binsize / 2)
        return downstream + [center_bin] + upstream

    def get_binstarts_percentile_split_central_roi(self,start, end):
        """
        Gets the bin start coordinates for a given core region, splitting the core region into 5 bins with a length of
        10,15,50,15,and 10 % of the core region length, respectively. The other bins, outside the core region,
        have a length of :attr:`.binsize'.
        The most upstream bin starts :attr:`.extend_to` bp upstream of the core region start, and the most downstream
        bin ends :attr:`.extend_to` bp downstream of the core region end.

        :param start: Coordinate of the start of the region of interest
        :param end: Coordinate of the end of the region of interest

        :return: A list with start coordinates, one per bin. Length / number of created bins
            depends on :attr:`.extend_to` and :attr:`.binsize`.
        """
        length=int(end)-int(start)
        downstream = list(range(start,int(start-self.extend_to)-1, -self.binsize))[::-1][:-1]
        upstream = list(range(int(end), int(end+self.extend_to), self.binsize))
        center_bin = [int(start +n*length) for n in [0,0.1,0.25,0.75,0.9]]
        return downstream + center_bin + upstream

    def write_bin_bedfile(self, out_bedfile_path_bins: str,
                          out_bedfile_path_regions_that_passed_filter: typing.Union[None,str]) -> None:
        """
        Splits every region in the :attr:`.bins_bed_filepath` file into bins, such that the central bin's center is at
        the center of the region, and such that the outermost bins each extend :attr:`.binsize`/2 over the edges of
        (*<the region's center>* - :attr:`.extend_to`) or (*<the region's center>* + :attr:`.extend_to`),
        respectively. Also assigns a bin nr. to every bin, starting with the most downstream bin.

        :param out_bedfile_path_bins: Path to which the output .bed file with bin coordinates should be written to.
        :param out_bedfile_path_regions_that_passed_filter: If not `None`, write the regions that passed all filters and
            that are the basis of the bins to this path (should end with .bed).
        """
        logging.info(f"Splitting input bedfile '{self.bed_filepath}' into bins ...")

        region_bed = pd.read_csv(self.bed_filepath, sep="\t", header=None)
        region_bed = region_bed[[0, 1, 2]]
        region_bed.columns = ["chromosome", "start", "end"]
        region_bed["chromosome"]=region_bed["chromosome"].astype("str")

        region_bed = self.filter_regions(region_bed)

        if out_bedfile_path_regions_that_passed_filter is not None:
            region_bed.to_csv(out_bedfile_path_regions_that_passed_filter,sep="\t",index=False,header=False)

        region_bed["center"] = round((region_bed["start"] + region_bed["end"]) / 2).astype(int)

        region_bed["bin starts"] = region_bed.apply(lambda row: self.get_binstarts(row["center"]), axis=1)

        bin_intervals = [[x[0]] + list(np.repeat(x[1:], 2)) + [x[-1] + self.binsize] for x in
            region_bed["bin starts"].values]
        bin_intervals = np.reshape(bin_intervals, (-1, 2))

        df = pd.DataFrame(bin_intervals)
        nr_of_bins_per_region = int(bin_intervals.shape[0] / region_bed.shape[0])
        df["chromosome"] = np.repeat(region_bed["chromosome"].values, nr_of_bins_per_region)
        df["bin nr."] = list(np.arange(0, nr_of_bins_per_region)) * region_bed.shape[0]
        df.columns = ["start", "end", "chromosome", "bin nr."]
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
        df = df[["chromosome", "start", "end", "bin nr."]]

        df.to_csv(out_bedfile_path_bins, sep="\t", header=False, index=False)
        logging.info(f"Wrote bed file with bin coordinates to '{out_bedfile_path_bins}'.")

    def write_bin_bedfile_percentile_split_central_roi(self, out_bedfile_path_bins: str,
                          out_bedfile_path_regions_that_passed_filter: typing.Union[None,str]) -> None:
        """
        Splits every region in the :attr:`.bins_bed_filepath` file into bins. The core region (given in
        :attr:`.bed_filepath`) is splitted into 5 bins with a length of 10,15,50,15,and 10 % of the core region length,
        respectively. The other bins, outside the core region, have a length of :attr:`.binsize'.
        The most upstream bin starts :attr:`.extend_to` bp upstream of the core region start, and the most downstream
        bin ends :attr:`.extend_to` bp downstream of the core region end. Also assigns a bin nr. to every bin,
        starting with the most downstream bin.
        :param out_bedfile_path_bins: Path to which the output .bed file with bin coordinates should be written to.
        :param out_bedfile_path_regions_that_passed_filter: If not `None`, write the regions that passed all filters
            and that are the basis of the bins to this path (should end with .bed).
        """
        logging.info(f"Splitting input bedfile '{self.bed_filepath}' into bins ...")

        region_bed = pd.read_csv(self.bed_filepath, sep="\t", header=None)
        region_bed = region_bed[[0, 1, 2]]
        region_bed.columns = ["chromosome", "start", "end"]
        region_bed["chromosome"]=region_bed["chromosome"].astype("str")

        region_bed = self.filter_regions(region_bed)

        if out_bedfile_path_regions_that_passed_filter is not None:
            region_bed.to_csv(out_bedfile_path_regions_that_passed_filter,sep="\t",index=False,header=False)

        region_bed["center"] = round((region_bed["start"] + region_bed["end"]) / 2).astype(int)

        region_bed["bin starts"] = region_bed.apply(lambda row: self.get_binstarts_percentile_split_central_roi(
            row["start"], row["end"]), axis=1)

        bin_intervals = [[x[0]] + list(np.repeat(x[1:], 2)) + [x[-1] + self.binsize] for x in
            region_bed["bin starts"].values]
        bin_intervals = np.reshape(bin_intervals, (-1, 2))

        df = pd.DataFrame(bin_intervals)
        nr_of_bins_per_region = int(bin_intervals.shape[0] / region_bed.shape[0])
        df["chromosome"] = np.repeat(region_bed["chromosome"].values, nr_of_bins_per_region)
        df["bin nr."] = list(np.arange(0, nr_of_bins_per_region)) * region_bed.shape[0]
        df.columns = ["start", "end", "chromosome", "bin nr."]
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
        df = df[["chromosome", "start", "end", "bin nr."]]

        df.to_csv(out_bedfile_path_bins, sep="\t", header=False, index=False)
        logging.info(f"Wrote bed file with bin coordinates to '{out_bedfile_path_bins}'.")


