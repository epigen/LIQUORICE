import sys
import os
import pandas as pd
import pathlib
import argparse
import logging
from datetime import datetime
import json
import tempfile
import urllib
import numpy as np

class FullPaths(argparse.Action):
    """
    Expand user- and relative-paths. From: https://gist.github.com/brantfaircloth/1252339. Does not convert the path
    to the blacklist if the value is 'hg38' or '10k_random, which is interpreted as an instruction to use the shipped
    list instead.
    """

    def __call__(self, parser, namespace, values,option_string=None,):
        if not values=="hg38" and not values=="10k_random":
            if type(values)==list:
                setattr(namespace, self.dest, [os.path.abspath(os.path.expanduser(x)) for x in values])
            else:
                setattr(namespace, self.dest, os.path.abspath(os.path.expanduser(values)))
        else:
            setattr(namespace, self.dest, values)

def parse_args():
    """
    Parses the arguments from the command line. For a full list of arguments,
    see the documentation of the LIQUORICE command line tool.

    :return: An `argparse.ArgumentParser` object storing the arguments.
    """
    parser = argparse.ArgumentParser(description="LIQUORICE: A tool for bias correction and quantification of changes "
                                                 " in coverage around regions of interest in cfDNA WGS datasets. Documentation: https://liquorice.readthedocs.io; Publication: https://doi.org/10.1093/bioadv/vbac017.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required_keyword_args = parser.add_argument_group('Required named arguments')

    required_keyword_args.add_argument(
        '--bamfile', help='.bam file containing the mapped reads of the sample. Used to infer coverage, fragment'
                          ' size, and read length.', required=True,action=FullPaths)

    required_keyword_args.add_argument(
        '--refgenome_fasta', help='Path to a .fa file of the reference genome. Must have a .fa.fai index in the same '
                                  'directory.',required=True,action=FullPaths)

    required_keyword_args.add_argument(
        '--mappability_bigwig', help='Path to a bigWig file that contains (forward) mappability values for every base '
                                     'in the reference genome. Can be calculated with gem-mappability for the '
                                     'appropriate read length.', required=True,action=FullPaths)


    optional_keyword_args = parser.add_argument_group('Optional named arguments - General settings')

    optional_keyword_args.add_argument(
        '--bedpathlist', help='List of paths to BED files, one for each region-set of interest. If unspecified, '
                              'only the biasmodel will be trained (if indicated by the --bedpath_biasmodel, '
                              '--detect_exisiting_biasmodel, and --use_this_biasmodel settings).',nargs="+",
        default=[],action=FullPaths)

    optional_keyword_args.add_argument(
        '--bedpath_biasmodel', help=".bed file containing regions that are used for generating the bias model for the "
                                    "sample. E.g. random regions should work well. Incompatible with "
                                    "use_provided_biasmodel. "
                                    "If '10k_random' is specified, a set of 10k random regions for hg38 shipped with "
                                    "the package is used for training unless an existing "
                                    "biasmodel can be used. "
                                    "If not specified / None (default), and if --use_this_biasmodel is also not "
                                    "specified, train a seperate biasmodel for each region-set "
                                    "that is specified in --bedpathlist, using the flanking regions (+- extend_to) "
                                    "for each region in the set.",
        action=FullPaths,default=None)

    optional_keyword_args.add_argument(
        '--binsize', help="Bin size is important for the resolution of the output plots & data, and for the bias model "
                          "itself. Smaller bin sizes give higher resolution, but take longer to calculate and may "
                          "result in more noise", default=500, type=int)

    optional_keyword_args.add_argument(
        '--extend_to', help="Size of the flanking region, in bp. Must be devidable by --binsize and must be a multiple "
                            "of 2. The regions will be extended by this value in both directions. The most upstream bin"
                            " starts <extend_to> bp upstream of the core region start, and the most downstream bin ends"
                            " <extend_to> bp downstream of the core region end. If --all_bins_same_size is set, instead"
                            "the outmost bins will have their center at <center of the region>+-<extend_to>. "
                            , default=20000, type=int)

    optional_keyword_args.add_argument(
        '--blacklist', help="Exclude regions if they overlap with the regions in this .bed file "
                            "(after extension by --extend_to). Default: None. Set to 'hg38' to use"
                            " the Boyle lab's hg38-blacklist.v2 that is shipped with LIQUORICE."
        ,type=str, default=None,action=FullPaths)

    optional_keyword_args.add_argument(
        '--cna_seg_file', help="If specified, use this .seg file to correct the coverage by the values specified in "
                               "this file prior to model training or bias correction. Use this if you want to normalize"
                               " out the effects of copy number aberrations (CNAs) on the coverage. File must be "
                               "tab-separated, with column names as first line. The second,third,and fourth column "
                               "must be chromosome, start, and end of the segment, and the last column must be the "
                               "log2-ratio of observed / expected read depth. This file can be generated e.g. by "
                               "running ichorCNA on the sample first."
        ,type=str, default=None,action=FullPaths)

    optional_keyword_args.add_argument(
        '--detect_existing_biasmodel', help="Check if a bias-model has already been built and saved under "
                                            "./<samplename>/biasmodel/trained_biasmodel.joblib. If so, "
                                            "use it, otherwise build one using --on a "
                                            "new biasmodel, overwriting files with the same name.",
        action="store_true")

    optional_keyword_args.add_argument(
        '--use_this_biasmodel', help="Use this bias model instead of training a model. IMPORTANT: This model has "
                                     "to come from the same sample/patient as the current one, otherwise the "
                                     "bias correction makes no sense.", default=None,action=FullPaths)

    optional_keyword_args.add_argument(
        '--extend_to_biasmodel', help="Ignored unless --bedpath_biasmodel is set. "
                                      "Size of the flanking region, in bp, to be used for the bias-model. "
                                      "Must be devidable by --binsize and must be a multiple "
                                      "of 2. The regions will be extended by this value in both directions. "
                                      "Outmost bins will have their center at <center of the region>+-<extend_to>. "
                                      "Default 0: Only place a single bin at the center of each provided region to "
                                      "speed up the training process.", default=0, type=int)

    optional_keyword_args.add_argument(
        '--no_chr_prefix', help='Specify this if the reference genome your .bam files are aligned to uses a chromosome '
                                'naming scheme such as "1,2,3,..,X,Y" instead of "chr1,chr2,chr3,..,chrX,chrY", which '
                                'is the default. Note that if your chromosomes are not named like the default, you '
                                'must not use the "10k_random" setting for --bedpath_biasmodel or the "hg38" setting '
                                'for --blacklist. Also, all other input files (refgenome_fasta, mappability_bigwig, '
                                'bedpathlist, and cna_seg_file must follow the same notation.',
        action="store_true")

    optional_keyword_args.add_argument(
        "--use_this_roi_biasfactortable", help="If set, use the specified biasfactor table "
                                               "and only train/apply the biasmodel, skipping the calculation of "
                                               "coverage and bias factors.",
        default=None,action=FullPaths)


    optional_keyword_args.add_argument(
        "--speed_mode", help="Only perform GC correction, don't correct using mappability or di/trinucleotides. "
                             "Setting this flag makes LIQUORICE considerably faster, but may lead to less accurate "
                             "results. Currently respected only if --bedpath_biasmodel is not specified.",
        action="store_true")

    optional_keyword_args.add_argument(
        "--all_bins_same_size", help="If set, use always the same bin size (for both the "
                                     "core region provided with --bedpath_list and the flanking regions "
                                     "defined by --extend_to), instead of splitting the core region into bins "
                                     "with sizes corresponding to 10,15,25,15,and 10%% of the core region's length. "
        ,action="store_true")
    optional_keyword_args.add_argument(
        "--dont_crossvalidate_if_train_on_rois", help="Unless set, if --train_on_rois is specified, train two seperate "
                                                      "bias models, each on half of the dataset, and use the trained"
                                                      "model to predict the other half.",action="store_true")

    optional_keyword_args_technical = parser.add_argument_group('Optional named arguments - Technical settings')

    optional_keyword_args_technical.add_argument(
        '--n_cpus', help="Number of processors to be used whereever multiprocessing/multithreading is used. ",
        default=1,type=int)

    optional_keyword_args_technical.add_argument(
        "--tmpdir", help="Use this directory as a temporary directory. "
                         "Default None: search environment variables $TMPDIR,$TEMP,$TMP, and paths /tmp,/var/tmp and "
                         "/usr/tmp, as well as the current working directory (in this order) until a suitable directory"
                         " is found.",
        default=None,action=FullPaths)

    optional_keyword_args_output = parser.add_argument_group('Optional named arguments - Output settings')

    optional_keyword_args_output.add_argument(
        '--samplename', help='Name of the sample that is being processed. This will be used for output plots and the '
                             'names of directories. Default None: Infer from --bamfile by removing .bam extension',
        default=None)

    optional_keyword_args_output.add_argument('--quiet', help='If set, the log level is set to "warning", making '
                                                              'LIQUORICE less chatty.', action="store_true")

    optional_keyword_args_output.add_argument(
        '--save_training_table', help="If set, save the training DataFrame of the bias model under"
                                      " ./<samplename>/biasmodel/training_table.csv "
                                      "(or ./<samplename>/<region-set name>/training_table.csv if --bedpath_biasmodel "
                                      "is not specified)",action="store_true")

    optional_keyword_args_output.add_argument(
        '--save_biasfactor_table', help="If set, for each region-set, save a table of bin coordinates, bin number, coverage, corrected coverage and biasfactor values"
                                        " per bin under "
                                        "./<samplename>/<region-set name>/coverage_and_biasfactors_per_bin.csv. (Filesize can get quite large)",
        action="store_true")

    optional_keyword_args_output.add_argument(
        '--save_corrected_coverage_table', help="If set, for each region-set, save a table of bin coordinates, bin number, coverage, and corrected coverage "
                                        " per bin under "
                                        "./<samplename>/<region-set name>/coverage_per_bin.csv",
        action="store_true")

    return parser


def main():
    """
    Main function for the LIQUORICE command line tool. Performs the following steps: Sanity checks for user input,
    setup of environment variables and general setup-steps, handles file paths, selects which steps need to be
    performed based on user input, logs the progress and performs the actual LIQUORICE analysis by calling
    :func:`liquorice.utils.GlobalFragmentSize.get_list_of_fragment_lengths_and_avg_readlength`,
    :func:`liquorice.utils.MeanSequencingDepth.sample_bam_to_get_sequencing_depth`,
    :func:`liquorice.utils.Workflows.train_biasmodel_for_sample`,
    :func:`liquorice.utils.Workflows.run_liquorice_on_regionset_with_pretrained_biasmodel`, and/or
    :func:`liquorice.utils.Workflows.run_liquorice_train_biasmodel_on_same_regions` after generating the
    corresponding objects.
    """
    parser=parse_args()
    args = parser.parse_args()

    # Set up logging
    if args.quiet:
        loglevel="WARNING"
    else:
        loglevel="INFO"
    logging.basicConfig(level=loglevel, format='%(levelname)s: %(asctime)s\t %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

    #  Sanity checks for user input:
    if args.bedpath_biasmodel and args.use_this_biasmodel:
        parser.error("ERROR: Please specify EITHER a .bed file to use for generation of the bias model "
                     "(--bedpath_biasmodel) OR a pre-calculated error model (--use_this_biasmodel), not both")
    if args.extend_to%2:
            sys.exit(f"ERROR: --extend_to (set to {args.extend_to}) must be a multiple of 2.")
    if args.extend_to%args.binsize:
        sys.exit(f"ERROR: --extend_to (was set to {args.extend_to}) must be a multiple of --binsize "
                 f"(was set to {args.binsize}).")
    if args.extend_to<args.binsize:
        sys.exit(f"ERROR: --extend_to (was set to {args.extend_to}) must be larger or equal to --binsize "
                 f"(was set to {args.binsize}).")
    if args.extend_to_biasmodel !=0 and args.bedpath_biasmodel is None:
        logging.warning("You have specified --extend_to_biasmodel but not --bedpath_biasmodel. "
                        "The --extend_to_biasmodel parameter will be ignored.")
    if args.bedpathlist==[] and args.bedpath_biasmodel is None:
        sys.exit(f"ERROR: Neither --bedpathlist nor --bedpath_biamodel was specified. "
                 f"Please specify one or both parameters.")

    check_these_paths=[args.bamfile,args.refgenome_fasta,args.mappability_bigwig,
        args.use_this_biasmodel,args.cna_seg_file,args.use_this_roi_biasfactortable]+args.bedpathlist
    for file in check_these_paths:
        if file is not None:
            if not os.path.isfile(file):
                sys.exit(f"ERROR: Input file '{file}' does not exist or is a directory - please correct.")
    if args.blacklist is not None and args.blacklist!="hg38":
        if not os.path.isfile(args.blacklist):
            sys.exit(f"ERROR: Input file '{args.blacklist}' does not exist or is a directory - please correct.")
    if args.bedpath_biasmodel is not None and args.bedpath_biasmodel!="10k_random":
        if not os.path.isfile(args.bedpath_biasmodel):
            sys.exit(f"ERROR: Input file '{args.bedpath_biasmodel}' does not exist or is a directory - please correct.")
    if args.tmpdir is not None:
        if not os.path.isdir(args.tmpdir):
            sys.exit(f"ERROR: The specified tmpdir '{args.tmpdir}' does not exist or is not a directory - "
                     f"please correct.")
    if args.samplename == "":
        sys.exit(f"ERROR: --samplename must not be set to an empty string - please either don't specify the flag at all"
                 f"or provide a proper string.")

    if args.no_chr_prefix:
        if args.blacklist == "hg38":
            sys.exit(f"ERROR: --blacklist hg38 is not compatible with --no_chr_prefix.")
        if args.bedpath_biasmodel == "10k_random":
            sys.exit(f"ERROR: --bedpath_biasmodel 10k_random is not compatible with --no_chr_prefix.")

    #  misc set-up
    np.random.seed(42)
    if args.samplename is None:
        args.samplename = args.bamfile.split("/")[-1].replace(".bam","")
    if args.blacklist == "hg38":
        args.blacklist=os.path.abspath(os.path.join(os.path.dirname(__file__),'data/hg38.blacklist.v2.0.bed'))
    detected_valid_biasmodel=False
    if args.detect_existing_biasmodel and pathlib.Path(f"{args.samplename}/biasmodel/trained_biasmodel.joblib").exists():
        detected_valid_biasmodel=True
    args.train_on_rois= args.bedpath_biasmodel is None and args.use_this_biasmodel is None and not detected_valid_biasmodel
    args.percentile_split_core_regions= not args.all_bins_same_size
    args.crossvalidated_predictions_if_train_on_rois = not args.dont_crossvalidate_if_train_on_rois

    if args.tmpdir is None:
        args.tmpdir=tempfile.gettempdir()
    os.environ["TMP"]=args.tmpdir
    os.environ["TMPDIR"]=args.tmpdir
    os.environ["TEMP"]=args.tmpdir

    #  Multi-processing set-up and dependent imports:
    os.environ["MODIN_CPUS"] = str(args.n_cpus)
    os.environ["OMP_NUM_THREADS"]= str(args.n_cpus)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    os.environ["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    try: # Can fail if OS==macOS
        import ray
        os.environ["MODIN_ENGINE"] = "ray"
        # activate if ray version ==1.1.0 or higher
        if ray.__version__.split(".")[0]=="1" and int(ray.__version__.split(".")[1])>=1:
            is_mac = sys.platform.startswith("darwin")
            maxlen_tmp_dir_path_in_bytes = (104 if is_mac else 108) - 1 - \
                                  len("/session_2021-09-13_18-13-19_866740_61434/sockets/plasma_store".encode('utf-8'))
            len_path_to_tmpdir_in_bytes=len(args.tmpdir.split("://", 1)[-1].encode('utf-8'))
            if len_path_to_tmpdir_in_bytes > maxlen_tmp_dir_path_in_bytes:
                logging.warning(f"The path to the specified tmpdir ({args.tmpdir}) is too long for the library ray which"
                                f" is"
                                f" used for parallel processing ({len_path_to_tmpdir_in_bytes} bytes observed, "
                                f"{maxlen_tmp_dir_path_in_bytes} bytes allowed). "
                                f"Will attempt to use ray's default, '/tmp', instead.")
                ray.init(num_cpus=args.n_cpus,include_dashboard=False)
            else:
                ray.init(num_cpus=args.n_cpus,_temp_dir=args.tmpdir,include_dashboard=False)
        else:
            logging.warning(f"Failed to initialize ray with the specified tmpdir '{args.tmpdir}' and "
                            f"num_cpus {args.n_cpus}."
                            f" This happens if the ray version is lower than 1.1.0. Setup will be done by modin "
                            f"instead, but this can mean the tmpdir will not be respected.")
        n_cpus_workflows=args.n_cpus
    except ModuleNotFoundError:
        #os.environ["MODIN_ENGINE"] = "dask"
        if args.n_cpus>1:
            logging.warning("Could not import ray library - this happens if running on macOS. Parallelization is "
                            "therefore only "
                            "available for a subset of analysis tasks - have a look at the 'Parallelization` section of"
                            " the documentation for more information: "
                            "https://liquorice.readthedocs.io/en/latest/liquorice_commandline_tool.html#parallelization")
        n_cpus_workflows=1
        pass
    from liquorice.utils import GlobalFragmentSize
    from liquorice.utils import MeanSequencingDepth
    from liquorice.utils import Workflows

    # Start the actual LIQUORICE workflow:
    logging.info(f"\n\n"
                 f"#######################################################"+
                 "".join(["#" for i in range(len(args.samplename))])+"\n"
                 f"############  Running LIQUORICE on sample {args.samplename}  ###########\n"
                 f"#######################################################"+
                 "".join(["#" for i in range(len(args.samplename))])+"\n")

    os.makedirs(args.samplename,exist_ok=True)
    os.chdir(args.samplename)

    if args.use_this_roi_biasfactortable and args.train_on_rois:
        logging.info(f"Using pre-calculated table of biasfactors: {args.use_this_roi_biasfactortable}")
        sampled_fragment_lengths, avg_readlength,mean_seq_depth=(None,None,None)
    else:
        # Get a sample of representative fragment lengths from the .bam file, and determine the read length:
        sampled_fragment_lengths, avg_readlength = GlobalFragmentSize.get_list_of_fragment_lengths_and_avg_readlength(
            args.bamfile,n_cores=args.n_cpus)

        # Get the mean sequencing depth of the .bam file:
        mean_seq_depth = MeanSequencingDepth.sample_bam_to_get_sequencing_depth(
           bam_filepath=args.bamfile,
           n_cores=args.n_cpus,
           n_sites_to_sample=10000,
           chromosomes_list=None if not args.no_chr_prefix else [str(i) for i in range(1,23)])

    # Determine wether a bias model shall be trained
    train_model=True
    if args.use_this_biasmodel:
        logging.info(f"Using existing biasmodel {args.use_this_biasmodel}. Biasmodel training will be skipped.")
        if args.detect_existing_biasmodel:
            logging.info(f"--detect_existing_biasmodel will be ignored because --use_this_biasmodel was specified.")
        train_model=False
    elif args.detect_existing_biasmodel:
        if pathlib.Path("biasmodel/trained_biasmodel.joblib").exists():
            logging.info(f"Detected existing bias model under {os.path.realpath('biasmodel/trained_biasmodel.joblib')}."
                     f" Biasmodel training will be skipped.")
            train_model=False
        else:
            logging.info(f"Did not detect an existing biasmodel under "
                         f"{os.path.realpath('biasmodel/trained_biasmodel.joblib')}.")
    elif not args.detect_existing_biasmodel and pathlib.Path("biasmodel/trained_biasmodel.joblib").exists():
        if not args.train_on_rois:
            logging.warning(f"Found an existing biasmodel under "
                        f"{os.path.realpath('biasmodel/trained_biasmodel.joblib')}"
                        f". This model will be overwritten as --detect_existing_biasmodel was not specified.")
        else:
            logging.warning(f"Found an existing biasmodel under "
                            f"{os.path.realpath('biasmodel/trained_biasmodel.joblib')}"
                            f". This model will be ignored as --detect_existing_biasmodel was not specified."
                            f"Instead, a seperate model will be trained on each region-set in --bedpathlist.")

    # Version in which a single bias-model is trained for the sample, based on a set of seperate training regions
    if not args.train_on_rois:
        if train_model:
            if args.bedpath_biasmodel == "10k_random":
                args.bedpath_biasmodel=os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                                    'data/10000_random_training_regions_hg38.bed'))
                logging.info("Using LIQUORICE's default region-set of 10k random regions for hg38 to train a biasmodel.")

            logging.info(f"###### Training a biasmodel for this sample using {args.bedpath_biasmodel.split('/')[-1]} #####")
            os.makedirs("biasmodel",exist_ok=True)
            os.chdir("biasmodel")
            Workflows.train_biasmodel_for_sample(
                bam_filepath=args.bamfile,
                bed_filepath=args.bedpath_biasmodel,
                refgenome_filepath=args.refgenome_fasta,
                samplename=args.samplename,
                refgenome_chromsizes_filepath=args.refgenome_fasta+".fai",
                sampled_fragment_lengths=sampled_fragment_lengths,
                avg_readlength=avg_readlength,
                refgenome_mappability_bigwig_path=args.mappability_bigwig,
                blacklist_bed_filepath=args.blacklist,
                cna_seg_filepath=args.cna_seg_file,
                mean_seq_depth=mean_seq_depth,
                binsize=args.binsize,
                n_cores=n_cpus_workflows,
                biasmodel_output_path="trained_biasmodel.joblib",
                extend_to=args.extend_to_biasmodel,
                save_training_table=args.save_training_table,
                no_chr_prefix=args.no_chr_prefix,
                percentile_split_core_rois=args.percentile_split_core_regions)
            os.chdir("..")

        for bed_filepath in args.bedpathlist:
            regionset_name=bed_filepath.split("/")[-1].replace(".bed","")
            # if pathlib.Path(regionset_name).exists():
            #     raise FileExistsError(f"A directory or file already exists under {os.path.abspath(regionset_name)}. "
            #                           f"Aborting.")
            os.makedirs(regionset_name,exist_ok=True)
            os.chdir(regionset_name)

            with open('binning_settings.json', 'w') as f:
                json.dump({"binsize":args.binsize, "extend_to":args.extend_to}, f)

            Workflows.run_liquorice_on_regionset_with_pretrained_biasmodel(
                bam_filepath=args.bamfile,
                bed_filepath=bed_filepath,
                samplename=args.samplename,
                biasmodel_path=args.use_this_biasmodel if args.use_this_biasmodel is not None else "../biasmodel/trained_biasmodel.joblib",
                regionset_name=regionset_name,
                refgenome_filepath=args.refgenome_fasta,
                refgenome_chromsizes_filepath=args.refgenome_fasta+".fai",
                refgenome_mappability_bigwig_path=args.mappability_bigwig,
                blacklist_bed_filepath=args.blacklist,
                sampled_fragment_lengths=sampled_fragment_lengths,
                avg_readlength=avg_readlength,
                cna_seg_filepath=args.cna_seg_file,
                mean_seq_depth=mean_seq_depth,
                binsize=args.binsize,
                extend_to=args.extend_to,
                n_cores=n_cpus_workflows,
                save_biasfactor_table=args.save_biasfactor_table,
                save_corrected_coverage_table=args.save_corrected_coverage_table,
                use_default_fixed_sigma_values=True,
                no_chr_prefix=args.no_chr_prefix,
                percentile_split_core_rois=args.percentile_split_core_regions,
                use_this_roi_biasfactortable=args.use_this_roi_biasfactortable)

            os.chdir("..")
        os.chdir("..")

    # Version in which a bias-model is trained for each region-set, using the flanking regions
    else:
        for bed_filepath in args.bedpathlist:
            regionset_name=bed_filepath.split("/")[-1].replace(".bed","")
            # if pathlib.Path(regionset_name).exists():
            #     raise FileExistsError(f"A directory or file already exists under {os.path.abspath(regionset_name)}."
            #                           f" Aborting.")
            os.makedirs(regionset_name,exist_ok=True)
            os.chdir(regionset_name)

            with open('binning_settings.json', 'w') as f:
                json.dump({"binsize":args.binsize, "extend_to":args.extend_to}, f)

            Workflows.run_liquorice_train_biasmodel_on_same_regions(
                bam_filepath=args.bamfile,
                bed_filepath=bed_filepath,
                samplename=args.samplename,
                regionset_name=regionset_name,
                refgenome_filepath=args.refgenome_fasta,
                refgenome_chromsizes_filepath=args.refgenome_fasta+".fai",
                refgenome_mappability_bigwig_path=args.mappability_bigwig,
                blacklist_bed_filepath=args.blacklist,
                sampled_fragment_lengths=sampled_fragment_lengths,
                avg_readlength=avg_readlength,
                cna_seg_filepath=args.cna_seg_file,
                mean_seq_depth=mean_seq_depth,
                binsize=args.binsize,
                extend_to=args.extend_to,
                n_cores=n_cpus_workflows,
                save_biasfactor_table=args.save_biasfactor_table,
                save_corrected_coverage_table=args.save_corrected_coverage_table,
                use_default_fixed_sigma_values=True,
                no_chr_prefix=args.no_chr_prefix,
                save_training_table=args.save_training_table,
                nr_of_bins_for_training_and_testing=None, # use all bins,
                skip_central_n_bins_for_training=5 if args.percentile_split_core_regions else 1, #Skip exactly the core
                # region if percentile_split_core_regions, or skip the center-most bin if all bins have the same size.
                percentile_split_core_rois=args.percentile_split_core_regions,
                use_cross_validated_predictions=args.crossvalidated_predictions_if_train_on_rois,
                use_this_roi_biasfactortable=args.use_this_roi_biasfactortable,
                speed_mode=args.speed_mode
            )

            os.chdir("..")
        os.chdir("..")

    logging.info(f"LIQUORICE analysis for sample {args.samplename} completed successfully!")


if __name__ == "__main__":
    sys.exit(main())
