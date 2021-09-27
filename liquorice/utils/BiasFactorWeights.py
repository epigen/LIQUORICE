import numpy as np
from Bio import Seq
from typing import Dict,List,Tuple

def get_GC_weights_binsize1(fragments: List[int]) -> List[float]:
    """
    For a given fragment size distribution, returns a list corresponding to the GC weight vector for a bin of size 1

    :param fragments: A list of fragment lengths that is representative of the sample's global fragment length
        distribution
    :return: A list corresponding to the GC weight vector for a bin of size 1. Its length is `max(fragments)*2`, the
        weight corresponding to the bin itself is at index `max(fragments)`. Values represent influence of a nucleotide
        at a given position on the bin's coverage.
    """
    longest_fraglen=max(fragments)
    weights_GC_binsize1=[]
    for pos in range(-longest_fraglen,longest_fraglen):
        weight=sum([fraglen-abs(pos) for fraglen in fragments if fraglen>=abs(pos)])
        weights_GC_binsize1.append(weight)
    return weights_GC_binsize1


def get_GC_weights_binwide(binsize: int, fragments: List[int], dont_slide: bool = False) -> List[float]:
    """
    For a given binsize, calculate the influence of a nucleotide at a given position on the bin's coverage

    :param binsize: Size of the bin
    :param fragments: A list of fragment lengths that is representative of the samples global fragment length
        distribution
    :param dont_slide: Set to False to return weights of 1 in the bin area and 0 outside of it
    :return: A list of length `(2*max(fragments)+binsize)`, the weight corresponding to the first position within the
        bin itself is at index `max(fragments)`. Values represent influence of a nucleotide at a given position on the
        bin's coverage.
    """
    weights_GC_binsize1=get_GC_weights_binsize1(fragments)
    longest_fraglen=max(fragments)

    # needs to be binsize longer than the weights if center is just a single position wide
    GC_weights_for_bin=[ [] for i in range (len(weights_GC_binsize1)+binsize)]
    # run through all postitions ("centers") within the bin
    for center in range(binsize):
        # the leftmost center just appends its weights to the list, the next appends them shifted by 1 pos and so on
        for pos in range(len(weights_GC_binsize1)):
            GC_weights_for_bin[center+pos].append(weights_GC_binsize1[pos])
        # to make the calculation of the mean meaningful, zeros need to be appended at positions where a center is not reached
        # by any fragment
        # do this for the positions that are too far downstream
        for pos in range(center):
            GC_weights_for_bin[pos].append(0)
        # and too far upstream
        for pos in range(binsize-center):
            GC_weights_for_bin[len(GC_weights_for_bin)-1-pos].append(0)
    GC_weights_for_bin=[np.mean(x) for x in GC_weights_for_bin]
    if not dont_slide:
        return GC_weights_for_bin
    else:
        return [0 for i in range(longest_fraglen)]+[1 for i in range(binsize)]+[0 for i in range(longest_fraglen)]


def get_dinuc_weights_binwide(GC_weights: List[float]) -> List[float]:
    """
    For a given binsize, calculate the influence of a given dinucleotide starting at a given position on the bin's
    coverage. This simply averages the GC weight of the position of interest and the following position. For the last
    position, a weight of 0 is returned.

    :param GC_weights: result of :func:`get_GC_weights_binwide` for the appropriate binsize and fragments of interest.
    :return:  A list of length `(2*max(fragments)+binsize)`, the weight corresponding to the first position within the
        bin itself is at index `max(fragments)`. Values represent influence of a dinucleotide starting at a given
        position on the  bin's coverage.
    """
    return [(GC_weights[i] + GC_weights[i + 1]) / 2 for i in range(len(GC_weights) - 1)] + [0]


def get_trinuc_weights_binwide(GC_weights: List[float]) -> List[float]:
    """
    For a given binsize, calculate the influence of a given trinucleotide starting at a given position on the bin's
    coverage. This simply averages the GC weight of the position of interest and the following 2 positions. For the last
    two positions, a weight of 0 is returned.

    :param GC_weights: result of get_GC_weights_binwide for the binsize and fragments of interest.
    :return:  A list of length `(2*max(fragments)+binsize)`, the weight corresponding to the first position within the
        bin itself is at index `max(fragments)`. Values represent influence of a trinucleotide starting at a given
        position on the  bin's coverage.
    """
    return [(GC_weights[i] + GC_weights[i + 1] + GC_weights[i + 2]) / 3 for i in range(len(GC_weights) - 2)] + [0,0]

def get_mapp_weights_binwide(binsize: int, fragments: List[int], dont_slide: bool = False) -> Tuple[List[float]]:
    """
    For a given binsize, calculate the influence a fragment starting/ending at a given position on the bin's coverage
    
    :param binsize: Size of the bin
    :param fragments: A list of fragment lengths that is representative of the samples global fragment length 
        distribution
    :param dont_slide: Set to False to return weights of 1 in the bin area and 0 outside of it
    :return: Two lists of length `(2*max(fragments)+binsize)`. Values represent influence a fragment
        starting/ending at a given position on the bin's coverage. First list is for forward mappability
        (fragments starting at the position), second list is for reverse mappability (fragments ending at the position)
    """
    longest_frag=max(fragments)
    weights_fwd={key:0 for key in range(-longest_frag,longest_frag+binsize)}
    weights_rev={key:0 for key in range(-longest_frag,longest_frag+binsize)}
    for pos in range(-longest_frag,binsize):
        first_binpos=0
        last_binpos=binsize-1
        for fraglen in fragments:
            if pos <=first_binpos:
                #overlap of the fragment with the bin
                weight=max(min(binsize,fraglen-abs(pos)),0)
                weights_fwd[pos]+=weight
                weights_rev[pos+fraglen-1]+=weight
            if pos > first_binpos and pos <= last_binpos:
                weight=max(min(binsize-abs(pos), fraglen),0)
                weights_fwd[pos]+=weight
                weights_rev[pos+fraglen-1]+=weight
    if not dont_slide:
        return list(weights_fwd.values()),list(weights_rev.values())
    else:
        return [0 for i in range(longest_frag)]+[1 for i in range(binsize)]+[0 for i in range(longest_frag)], \
            [0 for i in range(longest_frag)]+[1 for i in range(binsize)]+[0 for i in range(longest_frag)]


def get_nucleotide_dicts() -> Tuple[Dict[str,int]]:
    """
    Prepare dictionaries with di-and trinucleotides, as well as a dictionary for forward/reverse complement translation.

    :return: Three dictionaries: All dinucleotides excluding reverse complements, all trinucleotides excluding reverse
        complements, and a dictonary that allows translation of reverse complements to their forward complement counterparts
        (in this order). Keys are all 0.
    """
    # dictionary of all di- and trinucleotides:
    dinucdict={}
    trinucdict={}
    nucdict={}

    nucleotides=["A","T","G","C"]

    for nuc1 in nucleotides:
        for nuc2 in nucleotides:
            dinuc=nuc1+nuc2
            dinuc_seq=Seq.Seq(dinuc)
            if not str(dinuc_seq.reverse_complement()) in dinucdict.keys():
                dinucdict[dinuc]=0
            nucdict[dinuc]=0

    for nuc1 in nucleotides:
        for nuc2 in nucleotides:
            for nuc3 in nucleotides:
                trinuc=nuc1+nuc2+nuc3
                trinuc_seq=Seq.Seq(trinuc)
                if not str(trinuc_seq.reverse_complement()) in trinucdict.keys():
                    trinucdict[trinuc]=0
                nucdict[trinuc]=0

    revcompdict={key:str(Seq.Seq(key).reverse_complement()) for key in list(nucdict.keys())}
    return dinucdict,trinucdict,revcompdict