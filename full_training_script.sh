python classifier_train.py --epochs 100 --num_emos 4
python train_main.py --recon_only --config ./configs/config_step1_extension.yaml
python train_main.py --checkpoint ./checkpoints/model_step1/200000.ckpt --load_emo ./checkpoints/cls_checkpoint.ckpt --config ./configs/config_step2_extension.yaml
