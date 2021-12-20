FROM condaforge/mambaforge:4.9.2-5 as conda

RUN mamba install -c bioconda -c conda-forge liquorice==0.5.4 ray-core
