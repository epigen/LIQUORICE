#!/usr/bin/env bash

# Code from: https://evodify.com/gem-mappability/

GENOME_FASTA="$1"
READLENGTH="$2"
NCORES="$3"

FASTA_NAME="$(basename ${GENOME_FASTA})"
# dowload the GEM library binary file:
#wget https://sourceforge.net/projects/gemlibrary/files/gem-library/Binary%20pre-release%203/GEM-binaries-Linux-x86_64-core_i3-20130406-045632.tbz2
#bzip2 -d GEM-binaries-Linux-x86_64-core_i3-20130406-045632.tbz2
#tar -xvf GEM-binaries-Linux-x86_64-core_i3-20130406-045632.tar
#rm GEM-binaries-Linux-x86_64-core_i3-20130406-045632.tar
export PATH=$(realpath GEM-binaries-Linux-x86_64-core_i3-20130406-045632/bin):$PATH
#
#wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/wigToBigWig
#chmod 744 wigToBigWig
#
## Create GEM index and mappability tracks
#gem-indexer -T $NCORES -i $GENOME_FASTA -o ${FASTA_NAME}.gem_index
gem-mappability -T $NCORES -I "${FASTA_NAME}.gem_index.gem" -l $READLENGTH -o "${FASTA_NAME}.mappability_${READLENGTH}bp.gem"

# Convert the mappability track to bigwig format
gem-2-wig -I "${FASTA_NAME}.gem_index.gem" -i "${FASTA_NAME}.mappability_${READLENGTH}bp.gem.mappability" -o ${FASTA_NAME}.mappability_${READLENGTH}bp
./wigToBigWig "${FASTA_NAME}.mappability_${READLENGTH}bp.wig" "${FASTA_NAME}.mappability_${READLENGTH}bp.sizes" "${FASTA_NAME}.mappability_${READLENGTH}bp.bw"

## clean up
rm "${FASTA_NAME}.mappability_${READLENGTH}bp.gem.mappability"
rm "${FASTA_NAME}.mappability_${READLENGTH}bp.wig"