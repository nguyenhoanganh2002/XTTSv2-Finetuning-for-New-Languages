CUDA_VISIBLE_DEVICES=1 python train_dvae_xtts.py \
--output_path=checkpoints/ \
--train_csv_path=datasets/metadata_train.csv \
--eval_csv_path=datasets/metadata_eval.csv \
--language="vi" \
--num_epochs=5 \
--batch_size=512 \
--lr=5e-6