import logging
import random
random.seed(42)
import typing
import numpy as np

from deeptools import bamHandler
from deeptools import mapReduce

old_settings = np.seterr(all='ignore')

# https://github.com/deeptools/deepTools/blob/ac42d29c298c026aa0c53c9db2553087ebc86b97/deeptools/getFragmentAndReadSize.py#L10
def getFragmentLength_wrapper(args):
    return getFragmentLength_worker(*args)

#
def getFragmentLength_worker(chrom: str, start: int, end: int, bamFile: str, distanceBetweenBins: int) -> np.array:
    """
    This is a function from deeptools, modified from https://github.com/deeptools/deepTools/blob/ac42d29c298c026aa0c53c9db2553087ebc86b97/deeptools/getFragmentAndReadSize.py#L14
    Queries the reads at the given region for the distance between
    reads and the read length. As opposed to the original version, does not disregard pairs that are flagged as
    "not properly paired", as this behaviour excludes fragments from the two-nucleosome peak of the cfDNA-distribution.

    :param chrom: chromosome name
    :param start: region start
    :param end: region end
    :param bamFile: BAM file name
    :param distanceBetweenBins: The number of bases at the end of each bin to ignore
    :return an np.array, where first column is fragment length, the
        second is for read length
    """
    bam = bamHandler.openBam(bamFile)
    end = max(start + 1, end - distanceBetweenBins)
    if chrom in bam.references:
        reads = np.array([(abs(r.template_length), r.infer_query_length(always=False))
                             for r in bam.fetch(chrom, start, end)
                             if r.is_read1 and not r.is_unmapped]) # this line has been modified for LIQUORICE
        # This is the original line:
        #                           if r.is_proper_pair and r.is_read1 and not r.is_unmapped])
        if not len(reads):
            # if the previous operation produces an empty list
            # it could be that the data is not paired, then
            # we try with out filtering
            reads = np.array([(abs(r.template_length), r.infer_query_length(always=False))
                                 for r in bam.fetch(chrom, start, end) if not r.is_unmapped])
    else:
        raise NameError("chromosome {} not found in bam file".format(chrom))

    if not len(reads):
        reads = np.array([]).reshape(0, 2)

    return reads


def get_read_and_fragment_length(bamFile: str, return_lengths: bool=False, blackListFileName: str=None,
                                 binSize: int=50000, distanceBetweenBins: int=1000000,
                                 numberOfProcessors: int=None, verbose: bool=False) -> typing.Tuple[dict]:
    """
    This is a function from deeptools. It was included in LIQUORICE's source code to allow it to (indirectly) call the
    modified version of :func`getFragmentLength_worker`.
    Estimates the fragment length and read length through sampling

    :param bamFile: BAM file name
    :param return_lengths:
    :param numberOfProcessors:
    :param verbose:
    :param binSize:
    :param distanceBetweenBins:
    .:return A tuple of two dictionaries, one for the fragment length and the other
        for the read length. The dictionaries summarise the mean, median etc. values
    """

    bam_handle = bamHandler.openBam(bamFile)
    chrom_sizes = list(zip(bam_handle.references, bam_handle.lengths))

    distanceBetweenBins *= 2
    fl = []

    # Fix issue #522, allow distanceBetweenBins == 0
    if distanceBetweenBins == 0:
        imap_res = mapReduce.mapReduce((bam_handle.filename, distanceBetweenBins),
                                       getFragmentLength_wrapper,
                                       chrom_sizes,
                                       genomeChunkLength=binSize,
                                       blackListFileName=blackListFileName,
                                       numberOfProcessors=numberOfProcessors,
                                       verbose=verbose)
        fl = np.concatenate(imap_res)

    # Try to ensure we have at least 1000 regions from which to compute statistics, halving the intra-bin distance as needed
    while len(fl) < 1000 and distanceBetweenBins > 1:
        distanceBetweenBins /= 2
        stepsize = binSize + distanceBetweenBins
        imap_res = mapReduce.mapReduce((bam_handle.filename, distanceBetweenBins),
                                       getFragmentLength_wrapper,
                                       chrom_sizes,
                                       genomeChunkLength=stepsize,
                                       blackListFileName=blackListFileName,
                                       numberOfProcessors=numberOfProcessors,
                                       verbose=verbose)

        fl = np.concatenate(imap_res)

    if len(fl):
        fragment_length = fl[:, 0]
        read_length = fl[:, 1]
        if fragment_length.mean() > 0:
            fragment_len_dict = {'sample_size': len(fragment_length),
                'min': fragment_length.min(),
                'qtile25': np.percentile(fragment_length, 25),
                'mean': np.mean(fragment_length),
                'median': np.median(fragment_length),
                'qtile75': np.percentile(fragment_length, 75),
                'max': fragment_length.max(),
                'std': np.std(fragment_length),
                'mad': np.median(np.abs(fragment_length - np.median(fragment_length))),
                'qtile10': np.percentile(fragment_length, 10),
                'qtile20': np.percentile(fragment_length, 20),
                'qtile30': np.percentile(fragment_length, 30),
                'qtile40': np.percentile(fragment_length, 40),
                'qtile60': np.percentile(fragment_length, 60),
                'qtile70': np.percentile(fragment_length, 70),
                'qtile80': np.percentile(fragment_length, 80),
                'qtile90': np.percentile(fragment_length, 90),
                'qtile99': np.percentile(fragment_length, 99)}
        else:
            fragment_len_dict = None

        if return_lengths and fragment_len_dict is not None:
            fragment_len_dict['lengths'] = fragment_length

        read_len_dict = {'sample_size': len(read_length),
            'min': read_length.min(),
            'qtile25': np.percentile(read_length, 25),
            'mean': np.mean(read_length),
            'median': np.median(read_length),
            'qtile75': np.percentile(read_length, 75),
            'max': read_length.max(),
            'std': np.std(read_length),
            'mad': np.median(np.abs(read_length - np.median(read_length))),
            'qtile10': np.percentile(read_length, 10),
            'qtile20': np.percentile(read_length, 20),
            'qtile30': np.percentile(read_length, 30),
            'qtile40': np.percentile(read_length, 40),
            'qtile60': np.percentile(read_length, 60),
            'qtile70': np.percentile(read_length, 70),
            'qtile80': np.percentile(read_length, 80),
            'qtile90': np.percentile(read_length, 90),
            'qtile99': np.percentile(read_length, 99)}
        if return_lengths:
            read_len_dict['lengths'] = read_length
    else:
        fragment_len_dict = None
        read_len_dict = None

    return fragment_len_dict, read_len_dict

def get_list_of_fragment_lengths_and_avg_readlength(bam_filepath: str, n_cores: int = 1, n: int = 1000,
                                                    upper_limit: int = 800) -> typing.Tuple[typing.List[int],int]:
    """
    Sample fragments from the given .bam file to obtain a representative distribution of fragment lenghts.

    :param bam_filepath: Path to the .bam file for which fragments should be sampled.
    :param n_cores: Number of cores to use by :func:`deeptools.getFragmentAndReadSize.get_read_and_fragment_length`.
    :param n: Number of randomly sampled fragment lengths to generate
    :param upper_limit: Fragment lengths exceeding this limit will be excluded. Rarely, fragment size are wrongly
        inferred and therefore huge. Sampling one of those incorrect lengths would unnecessarily blow up the sequence
        and mapping information stored for each bin. As a default, 500 is used as a reasonable upper limit of relevant
        fragment lengths for cfDNA.
    :return: A tuple, consisting of a list of the randomly samples fragment lengths, and an integer value of the
        average read length.
    """

    logging.info(f"Sampling fragment sizes, using {n_cores} cores ...")

    # Get fragment size distribution; include "not properly paired" pairs when creating the distribution
    fragment_length_d, read_length_d = get_read_and_fragment_length(
                                                                  bamFile=bam_filepath,
                                                                  return_lengths=True,
                                                                  blackListFileName=None,
                                                                  binSize=50000,
                                                                  distanceBetweenBins=1000000,
                                                                  numberOfProcessors=n_cores,
                                                                  verbose=False)

    logging.info(f"Overview of global fragment lengths: "
                 f"median={int(fragment_length_d['median'])}; "
                 f"percentiles: "
                 f"10th={int(fragment_length_d['qtile10'])}, "
                 f"25th={int(fragment_length_d['qtile25'])}, "
                 f"75th={int(fragment_length_d['qtile75'])}, "
                 f"90th={int(fragment_length_d['qtile90'])}")
    logging.info(f"Average read length is {int(read_length_d['mean'])}.")

    sampled_fraglens = random.sample([abs(int(x)) for x in fragment_length_d["lengths"] if (upper_limit > x != 0)], n)

    return sampled_fraglens,int(read_length_d["mean"])
