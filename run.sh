python main.py --gpu 7 \
            --config_file configs/htm.yaml \
            --run_name round3_1200k_20epoch

# python finetune.py --gpu 2 \
#             --config_file configs/mr.yaml \
#             --run_name 1200k_10epoch_1e-4

# python eval_mr.py --gpu 2 \
#             --config_file configs/mr_eval.yaml \
#             --run_name eval_train_nopre