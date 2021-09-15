#!/usr/bin/env bash

### NOTE: Currently, this file is for internal testing purposes only.


import subprocess

param_dict={
#Note: The last entry in each of the dics is used as the default further below
# If False, the flag will not be specified
# If empty string, the flag will be specified without arguments
# If 0: Expect successfull termination with return code  0
# If 1: Expect failed run with return code >0

"bamfile":[["dummy",False,"","/home/peter/ctDNA/src/LIQUORICE/liquorice/data/Ctrl_17_testdata.bam"],[1,1,1,0]]# /data_synology_rg3/peter/ctDNA_ewing/results/mapped_bam_to_decomposed_signal/ln_to_treated_bams_allsamples_incl_ns7/MAP054.bam"],[]]

,"refgenome_fasta":[["dummy",False,"","/tmp/hg38.p12.fa", "/data2/peter/ctDNA_ewing/data/reference_data/hg38/hg38.fa"],[1,1,1,0,0]]

,"mappability_bigwig":[["dummy",False,"","/data2/peter/ctDNA_ewing/data/reference_data/hg38/hg38_mappability_75bp.bigwig",
    "/data_synology_rg3/peter/ctDNA_ewing/results/gem_mappability/hg38.fa.mappability_100bp.bw"],[1,1,1,0,0]]

,"bedpathlist":[[False,"dummy","","/data2/peter/ctDNA_ewing/data/reference_data/ewing_dnaseI_hypersensitivity/sknmc_specific_only10_totest.bed /data2/peter/ctDNA_ewing/data/reference_data/differential_histone_peaks/try_sort_by_worst_rank/hg38/diff_H3K27ac_peaks_neg_sorted_by_worstrank_top50_LOhg38.bed",
    "/data2/peter/ctDNA_ewing/data/reference_data/differential_histone_peaks/try_sort_by_worst_rank/hg38/diff_H3K27ac_peaks_neg_sorted_by_worstrank_top50_LOhg38.bed",
    "/data2/peter/ctDNA_ewing/data/reference_data/ewing_dnaseI_hypersensitivity/sknmc_specific_only10_totest.bed"],[1,1,1,0,0,0]]

,"bedpath_biasmodel":[["dummy","","10k_random",
    "/data2/peter/ctDNA_ewing/data/reference_data/random_regions/500_rand_regions_150bp.bed",False],[1,1,0,0]]

,"binsize":[["dummy","",10,1000,False],[1,1,0,0]]

,"extend_to":[["dummy","",100,40000,False],[1,1,1,0,0]]

,"blacklist":[["dummy", "",False,"hg38"],[1,1,0,0]]

,"cna_seg_file":[["dummy","","/data2/peter/ctDNA_ewing/results/linearmodel/input_no_pdx_novaseqs_complete_incl7/seg_files/CTR17PE001x.seg",False],[1,1,0,0]]

,"detect_existing_biasmodel":[["dummy","",False],[1,0,0]]

,"use_this_biasmodel":[["/data_synology_rg3/peter/ctDNA_ewing/results/linearmodel/test_LIQUORICEv.5.0/2021-06-17__extend_nonPP/RF_w_standardscaling/repaired_maxfragsize800/avoid_RF_overfitting_tests/sklearn_like_top_h20_leaderbord_gbm/trainonRandom_allbinssamesize/CTRPE001_trainonRandom_allBinSameSize_extto15k_bs150_autoML_800maxrepaired/biasmodel/trained_biasmodel.joblib",
        "","dummy",False],[0,1,1,0],]

,"extend_to_biasmodel":[["","dummy",100,False],[1,1,0,0]]

,"use_this_roi_biasfactortable":[["/data_synology_rg3/peter/ctDNA_ewing/results/linearmodel/test_LIQUORICEv.5.0/2021-06-17__extend_nonPP/RF_w_standardscaling/repaired_maxfragsize800/avoid_RF_overfitting_tests/sklearn_like_top_h20_leaderbord_gbm/NO2foldCV_trainOnROI_500bp_EXTD20k_and_percentilebins_ML_samples/Ost_8_1_trainonROI_percentileBins_extto20k_bs500_autoML_800maxrepaired/sknmc_specific/coverage_and_biasfactors_per_bin.csv",
        "","dummy",False],[0,1,1,0]]

,"speed_mode":[["",False],[0,0]]

,"all_bins_same_size":[["",False],[0,0]]

,"dont_crossvalidate_if_train_on_rois":[["",False],[0,0]]

,"n_cpus":[["","dummy",1,False,10],[1,1,0,0,0]]

,"tmpdir":[["","dummy","/tmp",".",False],[1,1,0,0,0]]

,"samplename":[["","testsamplename",False],[1,0,0]]

,"quiet":[["",False],[0,0]]

,"save_training_table":[[False,""],[0,0]]

,"save_biasfactor_table":[["",False],[0,0]]
}

#TODO: Test on a hg19 dataset with no_chr_prefix

def get_flag(key,value):
    if (type(value)==bool and not value):
        return []
    if type(value)==str and value=="":
        return [f"--{key}"]
    else:
        return [f"--{key}"]+[str(entry) for entry in (value.split(" ") if type(value)==str else [value])]

counter=0

failed_tests_against_expectations=[]
passed_tests_against_expectations=[]
passed_tests_as_expected=[]
failed_tests_as_expected=[]

for param_key_to_test in param_dict.keys():
    for param_value_to_test,expected_return_value in zip(param_dict[param_key_to_test][0],param_dict[param_key_to_test][1]):
        counter+=1

        cmd=["LIQUORICE"]
        flag_cmd_to_test=get_flag(param_key_to_test,param_value_to_test)
        cmd+=(flag_cmd_to_test)

        for other_param_key in [x for x in param_dict.keys() if not x==param_key_to_test]:
            other_param_value=param_dict[other_param_key][0][-1]
            cmd+=get_flag(other_param_key,other_param_value)

        if not "--samplename" in cmd:
            cmd+=["--samplename",f"test_{counter}"]
        
        print(f"\n\n\n############################# Test #{counter}: Testing {param_key_to_test}:{param_value_to_test}"
              f" #####################")

        result=subprocess.run(cmd)
        print(result)

        if result.returncode ==0:
            if expected_return_value==0:
                passed_tests_as_expected.append(counter)
            else:
                passed_tests_against_expectations.append(counter)
        else:
            if expected_return_value==0:
                failed_tests_against_expectations.append(counter)
            else:
                failed_tests_as_expected.append(counter)

print(f"The following tests failed even though they should have passed: {failed_tests_against_expectations}")
print(f"The following tests passed even though they should have failed: {passed_tests_against_expectations}")
print(f"The following tests passed as expected: {passed_tests_as_expected}")
print(f"The following tests failed as expected: {failed_tests_as_expected}")