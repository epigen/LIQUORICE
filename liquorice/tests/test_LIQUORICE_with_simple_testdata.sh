N_CPUS=1

# download and unzip the reference genome and reference mappability file
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/p12/hg38.p12.fa.gz
gunzip hg38.p12.fa.gz
wget https://github.com/epigen/LIQUORICE/raw/master/liquorice/data/hg38.p12.fa.fai
wget https://github.com/epigen/LIQUORICE/raw/master/liquorice/data/hg38.fa.mappability_100bp.subsetted_for_testdata.bw

# download .bam file of a healthy control liquid biopsy sample (pre-processed to keep the size small)
wget https://github.com/epigen/LIQUORICE/raw/master/liquorice/data/Ctrl_17_testdata.bam
wget https://github.com/epigen/LIQUORICE/raw/master/liquorice/data/Ctrl_17_testdata.bam.bai


# download .bed file for universally accessible DHSs
wget https://github.com/epigen/LIQUORICE/raw/master/liquorice/data/universal_DHSs.bed

# run LIQUORICE
LIQUORICE --bamfile Ctrl_17_testdata.bam --refgenome_fasta "hg38.p12.fa" \
        --mappability_bigwig "hg38.fa.mappability_100bp.subsetted_for_testdata.bw" \
        --bedpathlist "universal_DHSs.bed" \
        --blacklist "hg38" --n_cpus "${N_CPUS}"
LIQUORICE_summary