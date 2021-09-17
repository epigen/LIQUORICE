from liquorice.utils import deeptoolsCountReadsPerBinDontCheckProperPairSAMflag as crpb
import pandas as pd
import os
import pybedtools
if os.getenv("TMPDIR") is not None:
    pybedtools.set_tempdir(os.getenv("TMPDIR"))
import pyBigWig
import logging
import os
import typing
from multiprocessing import Pool
import pysam
import numpy as np
import sys
os.environ["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # required by Ray, which is
# used by modin
import modin.pandas as modinpd
import swifter

def run_get_coverage_of_region_per_chunk(dataframe_chunk,bam_filepath,min_mapq,n_cores, additional_crpb_kwargs,):
    """
    Uses :func`deeptools.countReadsPerBin.get_coverage_of_region()` to calculate the coverage in every bin
    in :param:`dataframe_chunk`. Adds the result as a column "region coverage array" and returns
    param:`dataframe_chunk`.
    This function can receive only picklable arguments, and can therefore be called multiple times in parallel.

    :param dataframe_chunk: A `pandas.DataFrame` with the columns "chromosome","region start","region end".
        Usually, this would be only a chunk of the full DataFrame.
    :param bam_filepath: Path to the .bam file containing the mapped reads for the sample. Does not need to be
        duplicate-filtered, this is done by the function that calculates the coverage.
    :param min_mapq: Minimum mapping quality for a fragment to still be counted when calculating the coverage.
    :param n_cores: Number of cores used by `deeptools.countReadsPerBin`. Should probably be 1 if the function is called
        in parallel
    :param additional_crpb_kwargs: Additional keywords to be used for deeptool's `CountReadsPerBin`. Can be used
        to override the default settings: *"ignoreDuplicates":True,  "samFlag_exclude":256,  "extendReads":True*
    :return: A `pandas.DataFrame` with the columns "chromosome","region start","region end" and "region coverage array".
    """

    crpb_kwargs={
        "minMappingQuality":min_mapq,
        "ignoreDuplicates":True,
        "numberOfProcessors":n_cores,
        "samFlag_exclude":256,  # not primary alignment
        "extendReads":True, # extend reads does not seem to fully work
        "numberOfSamples":0} # required for technical reasons
    crpb_kwargs.update(additional_crpb_kwargs)
    crpb_obj = crpb.CountReadsPerBin([bam_filepath], **crpb_kwargs)

    pysam_alignment_file=pysam_alignment_file=pysam.AlignmentFile(bam_filepath)

    stepsize=1
    dataframe_chunk["region coverage array"]=dataframe_chunk.apply(
        lambda row: crpb_obj.get_coverage_of_region(
            pysam_alignment_file,row["chromosome"],[(row["region start"],row["region end"],stepsize)]),axis=1)

    pysam_alignment_file.close()
    return dataframe_chunk

def parallelize_get_coverage_of_region(df, n_cores, bam_filepath,min_mapq, additional_crpb_kwargs):
    """
    Splits the provided `pandas.DataFrame` into chunks of the size :param`n_cores`, and
    runs :func`run_get_coverage_of_region_per_chunk` in parallel on every chunk.
    From: https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1

    :param df: Full `pandas.DataFrame` with the columns "chromosome","region start","region end".
    :param n_cores: Run this many processes in parallel.
    :param bam_filepath: Path to the .bam file containing the mapped reads for the sample. Does not need to be
        duplicate-filtered, this is done by the function that calculates the coverage.
    :param min_mapq: Minimum mapping quality for a fragment to still be counted when calculating the coverage.
    :param additional_crpb_kwargs: Additional keywords to be used for deeptool's `CountReadsPerBin`. Can be used
        to override the default settings: *"ignoreDuplicates":True,  "samFlag_exclude":256,  "extendReads":True*
    :return: Full `pandas.DataFrame` with the columns "chromosome","region start","region end" and
        "region coverage array".
    """
    n_cores=min(df.shape[0],n_cores) # cannot split a df in more pieces than it is long
    list_of_splitted_dfs = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    tasks=[(splitted_df,bam_filepath,min_mapq,1,additional_crpb_kwargs)
        for splitted_df in list_of_splitted_dfs]
    df = pd.concat(pool.starmap(run_get_coverage_of_region_per_chunk, tasks))
    pool.close()
    pool.join()
    return df

class CoverageAndSequenceTablePreparation:
    """
    Object used to set up a `pandas.DataFrame` containing basic information about each bin: Coverage (normalized for
    overall sequencing depth), genomic sequence & mappability (both incl. surrounding regions; used for bias
    correction). Usually, a user would want to call :func:`.get_complete_table` on this object, and save the
    resulting `pandas.DataFrame` for subsequent bias-correction.

    :param bam_filepath: Path to the .bam file containing the mapped reads for the sample. Does not need to be
        duplicate-filtered, this is done by the function that calculates the coverage.
    :param bins_bed_filepath: Path to the .bed file containing chromosome, start, end, and bin nr. of every bin to be
        analyzed (in that order). Output of
        :func:`liquorice.utils.RegionFilteringAndBinning.Binning.write_bin_bedfile` is a suitable input here.
    :param refgenome_filepath: Path to the reference genome .fa(.gz) file. Must have a .fai index in the same dirname.
    :param refgenome_chromsizes_filepath: Path to a tab-delimited file containing the chromosome sizes for the
        reference genome. The first column must be the chromosome name, the second column its size.
        Can be the .fa.fai file associated to the reference genome.
    :param refgenome_mappability_bigwig_path: Path to a .bigWig file containing (forward) mappability values scaled
        between 0 and 1.
    :param readlength: (Average) length of the reads in the .bam file.
    :param longest_fraglen: Length by which sequencing and mappability information is extended beyond a bin's
        borders. Typically, this should be the longest sampled fragment length.
    :param min_mapq: Minimum mapping quality for a fragment to still be counted when calculating the coverage.
    :param n_cores: Number of cores to be used by deeptool's `CountReadsPerBin`, which is used for coverage
        calculations. Set to higher values to speed up the process.
    :param mean_seq_depth: A float that quantifies the global coverage/sequencing depth. Coverages per bin will be
        normalized (i.e. divided) by this value.
    :param skip_these_steps: A list containing the steps which should be skipped. May contain: "coverage","sequence",
        "mappability". Default []: Don't skip any steps.
    :param **additional_crpb_kwargs: Additional keywords to be used for deeptool's `CountReadsPerBin`. Can be used
        to override the default settings: *"ignoreDuplicates":True,  "samFlag_exclude":256,  "extendReads":True*
    """

    def __init__(self, bam_filepath: str, bins_bed_filepath: str, refgenome_filepath: str,
                 refgenome_chromsizes_filepath: str, refgenome_mappability_bigwig_path: str,
                 readlength: int, longest_fraglen: int, mean_seq_depth: float, n_cores: int = 1,
                 min_mapq: int = 20, skip_these_steps: typing.List[str] =[],**additional_crpb_kwargs : dict) -> None:

        self.bam_filepath=bam_filepath
        self.bins_bed_filepath=bins_bed_filepath
        self.refgenome_filepath=refgenome_filepath
        self.refgenome_chromsizes_filepath=refgenome_chromsizes_filepath
        self.refgenome_mappability_bigwig_path=refgenome_mappability_bigwig_path

        self.readlength=readlength
        self.longest_fraglen=longest_fraglen

        self.min_mapq=min_mapq
        self.n_cores=n_cores

        self.additional_crpb_kwargs=additional_crpb_kwargs

        self.df=None
        self.mean_seq_depth=mean_seq_depth

        for item in skip_these_steps:
            if item not in ["coverage","sequence","mappability"]:
                raise ValueError('skip_these_steps may only contain the following items: "coverage","sequence",'
                                 '"mappability"')
        self.skip_these_steps=skip_these_steps


    def read_bed(self) -> pd.DataFrame:
        """
        Reads in a .bed file and return corresponding `pandas.DataFrame`. 4th column in the .bed file is interpreted
        as bin nr. Layout of the .bed file MUST be ["chromosome","start","end","bin nr."].

        :return: A `pandas.DataFrame` with the columns ["chromosome","start","end","bin nr."].
        """
        result_df=pd.read_csv(self.bins_bed_filepath, sep="\t", header=None)
        result_df.columns=["chromosome","start","end","bin nr."]
        result_df["chromosome"]=result_df["chromosome"].astype("str")
        return result_df

    def get_coverage_per_bin(self) -> pd.Series:
        """
        Calculates the normalized coverage per bin, based on :attr:`.bins_bed_filepath` and :attr:`.bam_filepath`.
        Uses deeptool's `countReadsPerBin` for calculating the coverage in every bin in :attr:`.df`.
        The obtained coverage value is then divided by :attr:`.mean_seq_depth`.
        This function is faster than :func:`get_coverage_per_bin_mean_over_per_basepair_values_chunked`, but provides
        less accurate results: Rather than the mean over the coverage of each base-pair in the bin, the total number of
        reads mapping to the bin (even partially mapping reads) are reported (after normalization for
        `:attr:.mean_seq_depth`).

        :return: A `pandas.Series` with the coverage per bin, normalized for total sequencing coverage.
        """

        crpb_kwargs={
            "minMappingQuality":self.min_mapq,
            "ignoreDuplicates":True,
            "numberOfProcessors":self.n_cores,
            "samFlag_exclude":256,  # not primary alignment
            "extendReads":True}
        crpb_kwargs.update(self.additional_crpb_kwargs)

        logging.info(f"Calculating coverage at every bin defined in the bed file '{self.bins_bed_filepath}', "
                     f"using {self.n_cores} cores - this may take a while ...")
        crpb_obj = crpb.CountReadsPerBin(
            [self.bam_filepath],
            bedFile=self.bins_bed_filepath,
            out_file_for_raw_data="readcount_per_bin.tsv",
            **crpb_kwargs)
        crpb_obj.run()

        # re-order, as multiprocessing messes up the order of readcount_per_bin.tsv
        raw_coverage_df=pd.read_csv("readcount_per_bin.tsv",sep="\t",header=None)
        raw_coverage_df.columns=["chromosome","start","end","raw coverage"]
        raw_coverage_df["chromosome"]=raw_coverage_df["chromosome"].astype("str")

        raw_coverage_df=raw_coverage_df.drop_duplicates() # It can happen that 2 close regions share some bins
        result_df = pd.merge(self.df, raw_coverage_df, left_on=["chromosome", "start", "end"],
                             right_on=["chromosome","start","end"], how="left")
        assert result_df.shape[0] == self.df.shape[0]
        try:
            assert all([x==y for x,y in zip(result_df["start"].values, self.df["start"].values)])
        except AssertionError:
            logging.error("The following result and self.df starts are not the same!:\n "
                          f"{[(x,y) for x,y in zip(result_df['start'].values, self.df['start'].values) if x != y]}")
            raise AssertionError

        # Normalize coverage by genome-wide average
        result_df["coverage"]=result_df["raw coverage"]/self.mean_seq_depth

        # clean up
        os.remove("readcount_per_bin.tsv")

        return result_df["coverage"]


    def get_coverage_per_bin_mean_over_per_basepair_values_chunked(self) -> pd.Series:
        """
        Calculates the normalized coverage per bin by averaging the per-base coverage over all positions in the bin and
        then deviding the result by :attr:`.mean_seq_depth`.
        Requires the following columns in :attr:`df`: "chromosome","start","end","bin nr.","region nr.","bin size".
        Uses deeptool's `countReadsPerBin` for calculating the coverage in every bin in :attr:`.df`.
        The obtained coverage value is then divided by :attr:`.mean_seq_depth`.

        :return: A `pandas.Series` with the coverage per bin, normalized for total sequencing coverage.
        """

        logging.info(f"Calculating coverage at every bin defined in the bed file '{self.bins_bed_filepath}', "
                     f"using {self.n_cores} cores - this may take a while ...")

        df_regions=self.df.groupby(["region nr."]).agg({"chromosome":lambda x:x.values[0],"start":"min","end":"max",})
        df_regions=df_regions.rename({"start":"region start","end":"region end"},axis=1)
        df_regions=parallelize_get_coverage_of_region(df_regions,self.n_cores, self.bam_filepath,self.min_mapq,
                                                      self.additional_crpb_kwargs)
        self.df=pd.merge(df_regions[["region coverage array","region start"]],self.df,left_index=True,
                         right_on="region nr.")
        self.df["relative start"]=self.df["start"]-self.df["region start"]
        self.df["relative end"]=self.df["relative start"]+self.df["bin size"]

        if True: # based on testing results, this version works more stably than the one below:
            result=self.df.apply(
                lambda row: np.mean(
                    row["region coverage array"][row["relative start"]:row["relative end"]]/self.mean_seq_depth),axis=1)
        else:
            result=self.df.swifter.progress_bar(False).set_npartitions(self.n_cores).apply(
                lambda row: np.mean(
                    row["region coverage array"][row["relative start"]:row["relative end"]]/self.mean_seq_depth),axis=1)

        self.df=self.df.drop(["region coverage array","region start","relative start","relative end",],axis=1)

        #result=result/self.mean_seq_depth # TODO what is better: This, or the inner mean in the row where result is
        # first created?

        return result


    def get_sequence_per_bin(self) -> typing.List[str]:
        """
        Get a list of genomic sequences, one per entry in the .bed file attr:`bins_bed_filepath`.
        Sequences are extended by :attr:`longest_fraglen`

        :return: A list of the genomic sequences of every bin in the attr:`bins_bed_filepath`.
        """
        bed = pybedtools.BedTool(self.bins_bed_filepath)

        # in case self.refgenome_chromsizes_filepath is a .fa.fai file, we need to convert it to a chrom.sizes file
        # for use in bedtools
        if self.refgenome_chromsizes_filepath.endswith(".fai"):
            with open("tmp.chrom.sizes","w") as outfile:
                with open(self.refgenome_chromsizes_filepath) as infile:
                    for line in infile:
                        line=line.split()
                        print(f"{line[0]}\t{line[1]}",file=outfile)
            bed = bed.slop(g="tmp.chrom.sizes",b=self.longest_fraglen)
        else:
            bed = bed.slop(g=self.refgenome_chromsizes_filepath,b=self.longest_fraglen)
        fasta_of_sequences = bed.sequence(fi=self.refgenome_filepath).seqfn

        sequence_list=[]
        with open(fasta_of_sequences) as infile:
            for line in infile:
                if line.startswith(">"):
                    continue
                sequence_list.append(line.rstrip())

        #  clean up if necessary
        if self.refgenome_chromsizes_filepath.endswith(".fai"):
            os.remove("tmp.chrom.sizes")
        del bed

        pybedtools.cleanup()
        return sequence_list

    def get_mappability_for_bin(self, row: pd.Series, mapp_bw: pyBigWig.pyBigWig) -> typing.List[float]:
        """
        For a single bin, get a list of (forward) mappability values in and around the bin based on its genomic
        coordinates and a `pyBigWig.bigWig` file that contains the genome-wide mappability.

        :param row: `pandas.Series` containing columns "chromosome","start", and "end".
        :param mapp_bw: `pyBigWig.pyBigWig` object with mappability info (i.e. result of `pyBigWig.open()`)
        :return: Mappability for the bin at base resolultion.
            Extended downstream by (:attr:`longest_fraglen` + :attr:`.readlength`) and upstream by :attr:`longest_fraglen`.
        """
        return mapp_bw.values(
            row["chromosome"],
            row["start"]-self.longest_fraglen-self.readlength,
            row["end"]+self.longest_fraglen, numpy=True
        ).astype(np.float16) #float16 saves a lot of memory and should still allow enough precision for our purposes
        # for 6000 regions, binsize 500 and extend_to 20k, memory usage of the dataframe was reduced from 5GB to 3GB.

    def get_mappability_per_bin(self) -> pd.Series:
        """
        Opens a `.bigWig` file using `pyBigWig` and calls :func:`.get_mappability_for_bin` for every bin.

        :return: A `pandas.Series` with mappability information per bin. Extended downstream by
            (:attr:`longest_fraglen` + :attr:`.readlength`) and upstream by :attr:`longest_fraglen`.
        """
        mapp_bw = pyBigWig.open(self.refgenome_mappability_bigwig_path)
        result= self.df.apply(lambda row: self.get_mappability_for_bin(row, mapp_bw), axis=1)
        mapp_bw.close()
        return result

    def get_complete_table(self) -> pd.DataFrame:
        """
        Main method to retrieve a `pandas.DateFrame` that can be used to calculate bias factors on.
        Calls :func:`.read_bed', :func:`.get_coverage_per_bin_mean_over_per_basepair_values_chunked`,
        :func:`.get_sequence_per_bin`, and
        :func:`.get_mappability_per_bin', constructs :attr:`.df`, a `pandas.DataFrame`, from the results, and
        returns it.

        :return: A `pandas.DataFrame` where rows correspond to the genomic bins in :attr:`.bins_bed_filepath`.
            This `DataFrame` has the following columns: "chromosome","start","end","bin nr.","coverage","sequence",
            "mappability".
            Entries in "sequence" and "mappability" columns are extended by :attr:`longest_fraglen`  (and downstream
            additionally by :attr:`.readlength` for "mappability").
        """
        self.df=self.read_bed()

        self.df["bin size"]=self.df["end"]-self.df["start"] # TODO remove if not required anymore

        if "coverage" not in self.skip_these_steps:
           # logging.info("Retrieving fragment coverage values ... ")
            self.df["region nr."]=((self.df["bin nr."]-1)!=self.df["bin nr."].shift()).cumsum()-1
            self.df["coverage"]=self.get_coverage_per_bin_mean_over_per_basepair_values_chunked()
            self.df=self.df.drop({"region nr."},axis=1) # clean up

        if "sequence" not in self.skip_these_steps:
            logging.info("Retrieving genomic sequences for every bin ... ")
            try:
                self.df["sequence"]=self.get_sequence_per_bin()
            except ValueError:
                sys.exit("Could not retrieve all required genomic sequences. This can happen if --tmpdir does not have"
                         "enough space left to store the temporary files produces by pybedtools. Try making some space"
                         "or specify a different tmpdir with the --tmpdir flag "
                         "(or change the value of the $TMP environment variable.")

        if "mappability" not in self.skip_these_steps:
            logging.info("Retrieving mappability information for every bin ... ")
            self.df["mappability"]=self.get_mappability_per_bin()

        return self.df


