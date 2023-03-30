import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')) # required by Ray, which is
# used by modin
from liquorice.utils.BiasFactorWeights import *
import numpy as np
import modin.pandas as modinpd
import pandas as pd
import logging
import swifter
from datetime import datetime
from typing import List,Dict

class BiasFactorHandler:
    """
    Object used for calculation of per-bin bias factors. Typically, after creation of the object a user would call
    its method :func:`.get_table_with_bias_factors` on it, and save the returned `DataFrame` for subsequent analysis
    and correction of associations of bias factors with coverage.

    :param binsize: Size of the bins. Higher values to reduce noise, lower values increase spatial
        resolution.

    :param fragments: A list containing fragment lengths that are representative of the sample's global fragment
        size distribution. Typically a few hundred fragments will suffice here.
    :param readlength: Average length of reads in the .bam file.
    :param df: `pandas.DataFrame` with one row per bin, containing columns "chromosome", "start", "end",
        "bin nr.", "coverage", "sequence", and "mappability". Suitable input is the output of the
        :func:`get_complete_table` method of the :attr:`liquorice.utils.CoverageAndSequenceTablePreparation` class
        object.
    :param n_cores: Max number of cores to use for calculations.
    :param skip_these_biasfactors: Do not calculate these bias factors. Only these entries are allowed:
        ["di and trinucleotides and GC content","mappability", "di and trinucleotides"]
    """

    def __init__(self, binsize: int, fragments: List[int], readlength: int, df: pd.DataFrame,
                 n_cores: int = 1, skip_these_biasfactors: List[str] = []) -> None:

        self.binsize=binsize
        self.readlength=readlength

        self.fragments=fragments
        self.longest_frag=max(fragments)

        self.df=df

        self.GC_weights=get_GC_weights_binwide(binsize=binsize,fragments=fragments)
        self.dinuc_weights=get_dinuc_weights_binwide(self.GC_weights)
        self.trinuc_weights=get_trinuc_weights_binwide(self.GC_weights)
        self.total_GC_weight=sum(self.GC_weights)
        self.total_dinuc_weight=sum(self.dinuc_weights)
        self.total_trinuc_weight=sum(self.trinuc_weights)

        self.fwd_mapp_weights,self.rev_mapp_weights=get_mapp_weights_binwide(binsize=binsize,fragments=fragments)
        self.total_fwd_mapp_weight,self.total_rev_mapp_weight=sum(self.fwd_mapp_weights),sum(self.rev_mapp_weights)

        self.dinucdict, self.trinucdict, self.revcompdict = get_nucleotide_dicts()

        self.n_cores=n_cores

        for item in skip_these_biasfactors:
            if item not in ["di and trinucleotides and GC content","mappability", "di and trinucleotides"]:
                raise ValueError('skip_these_biasfactors may only contain the following items: '
                                 '"di and trinucleotides and GC content",'
                                 '"mappability", "di and trinucleotides"')
        self.skip_these_biasfactors=skip_these_biasfactors

    def get_GC_and_di_and_trinuc_weights(self, sequence: str) -> Dict[str,float]:
        """
        Calculate bias factors for GC-content as well as bias factors for all di- and trinucleotides
        (reverse and forward complements merged into single factors) for a given sequence. Factors are scaled between
        0 and 1, 1 is highest (e.g. GC content: 0.. no G or C, 0.461... average GC content, 1... only G and C).

        :param sequence: Genomic sequence of the bin extended by *max(* :attr:`.fragments` *)* in both directions,
            such that its length matches :attr:`GC_weights`.
        :return: A dictionary with entries corresponding to the bias factors for GC-content as well as bias factors
            for all di- and trinucleotides (reverse and forward complements merged into single factors).
        """
        if type(sequence)!=str:  # Required because swifter calls this function with a series as sequence once, and
            # and does not handle the RunTimeError below properly.
            raise TypeError(f"The input to this function must be a str, but was {type(sequence)}.")
        GC_list=[]
        dinuc_res_dict=self.dinucdict.copy()
        trinuc_res_dict=self.trinucdict.copy()

        # start_time=datetime.now()
        for pos in range(len(sequence)):
            subseq=sequence[pos:pos+3]
            dinuc_weight=self.dinuc_weights[pos]
            trinuc_weight=self.trinuc_weights[pos]

            ### GC part ###
            nuc=subseq[0]
            if nuc=="G" or nuc=="g" or nuc=="c" or nuc=="C":
                GC_at_pos=1
            elif nuc=="A" or nuc=="a" or nuc=="t" or nuc=="T":
                GC_at_pos=0
            elif nuc=="N" or nuc=="n":
                GC_at_pos=0.461 #0.461 is the genome-wide GC content mean
            else:
                logging.error(f"Unexpected letter '{nuc}' encountered in genomic sequence.")
                raise RuntimeError
            GC_list.append(GC_at_pos)

            ### Dinucleotide part ###
            dinuc=subseq[:2].upper()
            # try:
            #     weight=(GC_weights_subsetted[0]+GC_weights_subsetted[1])/2
            try:
                dinuc_res_dict[dinuc]+=dinuc_weight
            except KeyError:  # if the dinuc. is not a key in the dict, usually its reverse complement should be
                try:
                    dinuc_revcomp=self.revcompdict[dinuc]
                    dinuc_res_dict[dinuc_revcomp]+=dinuc_weight
                except KeyError:
                    assert len(dinuc)<2 or "N" in dinuc # if at last position or an "N" is contained
                    pass
            # except IndexError:  # this happens if we are looking at the last position
            #     pass

            ### Trinucleotide part ###
            trinuc=subseq.upper()
            # try:
            #     weight=(GC_weights_subsetted[0]+GC_weights_subsetted[1]+GC_weights_subsetted[2])/3
            try:
                trinuc_res_dict[trinuc]+=trinuc_weight
            except KeyError:  # if the trinuc. is not a key in the dict, usually its reverse complement should be
                try:
                    trinuc_revcomp=self.revcompdict[trinuc]
                    trinuc_res_dict[trinuc_revcomp]+=trinuc_weight
                except KeyError:
                    assert len(trinuc)<3 or "N" in trinuc  # if at (second-to-) last position or an "N" is contained
                    pass
            # except IndexError:
            #     pass

        ## Normalize, combine and return result
        GC_bias_factor={"GC content":sum(np.array(GC_list)*np.array(self.GC_weights)) / self.total_GC_weight}
        dinuc_res_dict={key:value/self.total_GC_weight for key, value in dinuc_res_dict.items()}
        trinuc_res_dict={key:value/self.total_GC_weight for key, value in trinuc_res_dict.items()}

        res=trinuc_res_dict
        res.update(dinuc_res_dict)
        res.update(GC_bias_factor)

        return res



    def get_GC_bias_factor(self, sequence: str) -> float:
        """
        Returns a number in the range 0-1 that represents the bin's overall GC bias. Factors are scaled between
        0 and 1, 1 is highest (e.g. GC content: 0.. no G or C, 0.461... average GC content, 1... only G and C).

        :param sequence: Genomic sequence of the bin extended by *max(* :attr:`.fragments` *)* in both directions,
            such that its length matches :attr:`GC_weights`.
        :return: A number in the range 0-1 that represents the bin's overall GC bias.
        """

        if type(sequence)!=str:  # Required because swifter calls this function with a series as sequence once, and
        # and does not handle the RunTimeError below properly.
            raise TypeError(f"The input to this function must be a str, but was {type(sequence)}.")
        GC_list=[]

        # start_time=datetime.now()
        for pos in range(len(sequence)):
            subseq=sequence[pos:pos+3]

            ### GC part ###
            nuc=subseq[0]
            if nuc=="G" or nuc=="g" or nuc=="c" or nuc=="C":
                GC_at_pos=1
            elif nuc=="A" or nuc=="a" or nuc=="t" or nuc=="T":
                GC_at_pos=0
            elif nuc=="N" or nuc=="n":
                GC_at_pos=0.461 #0.461 is the genome-wide GC content mean
            else:
                logging.error(f"Unexpected letter '{nuc}' encountered in genomic sequence.")
                raise RuntimeError
            GC_list.append(GC_at_pos)

        GC_bias_factor={"GC content":sum(np.array(GC_list)*np.array(self.GC_weights)) / self.total_GC_weight}


        res=GC_bias_factor
        return res


    # if type(sequence)!=str:  # Required because swifter calls this function with a series as sequence once, and
    #                              # does not handle the RunTimeError below properly.
    #         raise TypeError(f"The input to this function must be a str, but was {type(sequence)}.")
    #
    #     GC_list=[]
    #     for nuc in sequence:
    #         if nuc=="G" or nuc=="g" or nuc=="c" or nuc=="C":
    #             GC_at_pos=1
    #         elif nuc=="A" or nuc=="a" or nuc=="t" or nuc=="T":
    #             GC_at_pos=0
    #         elif nuc=="N" or nuc=="n":
    #             GC_at_pos=0.461 #0.461 is the genome-wide GC content mean
    #         else:
    #             logging.error(f"Unexpected letter '{nuc}' encountered in genomic sequence.")
    #             raise RuntimeError
    #         GC_list.append(GC_at_pos)
    #
    #     GC_bias_factor=sum(np.array(GC_list)*np.array(self.GC_weights)) / self.total_GC_weight
    #     return GC_bias_factor

    def get_fwd_mappability_bias_factor(self, mappability: List[float]) -> float:
        """
        Returns a number representing the bin's mappability for fragments on the forward strand.

        :param mappability: A vector of values between 0 and 1, representing the mappbility of positions in and around
            the bin of interest. Must correspond to the coordinates <Bin start coord> - *max(* :attr:`.fragments` *)* -
            :attr:`.readlength` to <Bin end coord> + *max(* :attr:`.fragments` *)* + :attr:`.readlength`
        :return: A number in range 0-1 that represents the bin's overall forward mappability. 1 ... highest
        """
        mapp_fwd=mappability[self.readlength:]
        fwd_mappbias_of_bin=sum(np.array(mapp_fwd)*np.array(self.fwd_mapp_weights)) / self.total_fwd_mapp_weight
        return fwd_mappbias_of_bin

    def get_rev_mappability_bias_factor(self,mappability: List[float]) -> float:
        """
        Returns a number representing the bin's mappability for fragments on the reverse strand.

        :param mappability: A vector of values between 0 and 1, representing the mappbility of positions in and around
            the bin of interest. Must correspond to the coordinates <Bin start coord> - *max(* :attr:`.fragments` *)* -
            :attr:`.readlength` to <Bin end coord> + *max(* :attr:`.fragments` *)* + :attr:`.readlength`
        :return: A number in range 0-1 that represents the bin's overall reverse mappability. 1 ... highest
        """
        mapp_rev=mappability[:-self.readlength]
        rev_mappbias_of_bin=sum(np.array(mapp_rev)*np.array(self.rev_mapp_weights)) / self.total_rev_mapp_weight
        return rev_mappbias_of_bin

    def get_table_with_bias_factors(self) -> pd.DataFrame:
        """
        Main method to retrieve bias factor information.

        :return: A `pandas.DataFrame` with one row per bin, containing columns of the input DataFrame :attr:`df`
            ("chromosome", "start", "end", "bin nr.", "coverage", "sequence") as well as newly added
            columns for bias
            factors for mappability, GC-content, and  all di- and trinucleotides (with reverse and forward complements
            merged into single factors). Bias factors are scaled between 0 and 1. Higher values correspond to higher
            nucleotide content, mappability etc. Example: An average bin is expected to have a bias factor of 0.461
            for humans (i.e. genomic GC content).
            Note that the input columns "mappability" and "sequence" will not be part of the returned dataframe in
            order to reduce the memory usage of this function.
        """

        if "mappability" not in self.skip_these_biasfactors:
            logging.info(f"Adding forward mappability bias factors with up to {self.n_cores} cores ...")
            # alternative versions that turned out to be slower:
            # modin: 3:19 min with 25 cores
            # swifter and modin: 3:17 with 25 cores

            self.df["forward mappability"] = self.df["mappability"].swifter.progress_bar(False).set_npartitions(
                self.n_cores).apply(self.get_fwd_mappability_bias_factor)  # fast: 1:27 min with 25 cores allowed
            # (still uses 1 core only)

            logging.info(f"Adding reverse mappability bias factors with up to {self.n_cores} cores ...")
            self.df["reverse mappability"] = self.df["mappability"].swifter.progress_bar(False).set_npartitions(
                self.n_cores).apply(self.get_rev_mappability_bias_factor)  # fast: 1:27 min with 25 cores allowed
            # (still uses 1 core only)

            logging.info("Adding maximum mappability bias factors ...")
            self.df["max mappability"] = self.df[["forward mappability", "reverse mappability"]].max(axis=1)  # fast

            self.df=self.df.drop(["mappability"],axis=1)

        logging.info(f"Adding sequence-based bias factors with up to {self.n_cores} cores - this may take a while. "
                     f"(The following 'UserWarning' from modin can be ignored) ...")
        # alternative versions that turned out to be slower:

        # Loop: slow, takes probably about 10 min:
        # nuc_GC_factors_dicts = [self.get_GC_and_di_and_trinuc_weights(seq) for seq in self.df["sequence"].values]

        # multiprocessing pool.map: slow. schedules 25 tasks, but htop shows low usage.
        # with mp.Pool(processes=self.n_cores) as pool:
        #    nuc_GC_factors_dicts = pool.map(self.get_GC_and_di_and_trinuc_weights, self.df["sequence"].values)

        # modin Series.apply: 3:40 on 25 cores incl conversion to df and concat
        #nuc_GC_factors_dicts = modinpd.Series(self.df["sequence"]).apply(self.get_GC_and_di_and_trinuc_weights)

        if "di and trinucleotides and GC content" not in self.skip_these_biasfactors and "di and trinucleotides" not \
                in self.skip_these_biasfactors:
            if self.n_cores==1:
                nuc_and_gc_factors_dicts= self.df["sequence"].apply(
                    self.get_GC_and_di_and_trinuc_weights)
            else:
                nuc_and_gc_factors_dicts= modinpd.Series(self.df["sequence"]).swifter.set_npartitions(self.n_cores).apply(
                     self.get_GC_and_di_and_trinuc_weights)  # fast: 2:01 min on 25 cores incl. conversion to df and concat
            # nuc_and_gc_factors_dicts=self.df["sequence"].apply(self.get_GC_and_di_and_trinuc_weights)
            nuc_and_gc_df = pd.DataFrame(list(nuc_and_gc_factors_dicts.values))
            self.df=pd.concat([self.df,nuc_and_gc_df], axis=1)
            self.df=self.df.drop(["sequence"],axis=1)

        if "di and trinucleotides and GC content" not in self.skip_these_biasfactors and \
            "di and trinucleotides" in self.skip_these_biasfactors:
            if self.n_cores==1:
                gc_factors_dict=self.df["sequence"].apply(self.get_GC_bias_factor)
            else:
                gc_factors_dict=modinpd.Series(self.df["sequence"]).swifter.set_npartitions(self.n_cores).apply(
                    self.get_GC_bias_factor)
            gc_factor_df = pd.DataFrame(list(gc_factors_dict.values))
            self.df=pd.concat([self.df,gc_factor_df], axis=1)

                # self.df["GC content"]=modinpd.Series(self.df["sequence"]).swifter.set_npartitions(self.n_cores).apply(
                #     self.get_GC_bias_factor)
            self.df=self.df.drop(["sequence"],axis=1)


        return self.df
