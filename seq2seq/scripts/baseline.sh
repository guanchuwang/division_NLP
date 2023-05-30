# This scripts trains full finetuning method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3. 
folder_name=all_output_logs/
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi


source scripts/env.sh

config_file_name=configs/baseline.json
update_file_name=configs/baseline/baseline_$2_$3.json

for seed in 0 # 1 2
do
    # rm -r outputs/full_finetuning/
    python scripts/update_scripts_for_given_input.py $config_file_name "" $update_file_name
    
    python scripts/update_scripts_for_given_input.py $update_file_name task_name str $2  $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name eval_dataset_name str $2  $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name test_dataset_name str $2  $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name split_validation_test bool true $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name pad_to_max_length bool true $update_file_name

    python scripts/update_scripts_for_given_input.py $update_file_name learning_rate float 3e-5 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name num_train_epochs int ${num_epochs[$2]}   $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name seed int $seed   $update_file_name

    python scripts/update_scripts_for_given_input.py $update_file_name output_dir   str "outputs/full_finetuning_$2_$3" $update_file_name

    CUDA_VISIBLE_DEVICES=$1 python run_seq2seq.py  $update_file_name

    # cp outputs/full_finetuning/all_results.json  all_output_logs/full_finetuning_$2@${seed}.json

done