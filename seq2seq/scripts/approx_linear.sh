# This scripts trains full finetuning method.
# For smaller datasets of GLUE (mrpc, cola, and stsb), we set the `num_train_epochs` to 20,
# for other larger datasets in GLUE we used `num_train_epochs` of 3. 
folder_name=all_output_logs/
if [ ! -d ${folder_name} ] ; then
    mkdir -p ${folder_name}
fi

echo $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11}

source scripts/env_approx.sh

config_file_name=configs/approx_linear.json
update_file_name=configs/approx_linear/approx_linear_$2_${11}_B$9_Q${10}.json

for seed in 2 # 7 # 0 1 2
do
    # rm -r outputs/full_finetuning/
    python scripts/update_scripts_for_given_input.py $config_file_name "" $update_file_name

    python scripts/update_scripts_for_given_input.py $update_file_name task_name str $2 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name eval_dataset_name str $2 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name test_dataset_name str $2 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name split_validation_test bool false $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name pad_to_max_length bool true $update_file_name

    python scripts/update_scripts_for_given_input.py $update_file_name learning_rate float ${lr[$2]} $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name num_train_epochs int ${num_epochs[$2]} $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name seed int $seed $update_file_name

    python scripts/update_scripts_for_given_input.py $update_file_name k_sampling    int 0 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name q_sampling    int 0 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name v_sampling    int 0 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name o_sampling    int $3 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name wi_0_sampling int $4 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name wi_1_sampling int $5 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name wo_sampling   int $6 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name score_sampling   int $7 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name attout_sampling   int $8 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name softmax_prune_ratio   float 0 $update_file_name

    python scripts/update_scripts_for_given_input.py $update_file_name lfc_block   int $9 $update_file_name
    python scripts/update_scripts_for_given_input.py $update_file_name hfc_bit   int ${10} $update_file_name

    python scripts/update_scripts_for_given_input.py $update_file_name output_dir   str "outputs/full_finetuning_$2_${11}_B$9_Q${10}" $update_file_name

    CUDA_VISIBLE_DEVICES=$1 python run_seq2seq.py  $update_file_name

    # cp outputs/full_finetuning/all_results.json  all_output_logs/full_finetuning_$2@$7.json

done