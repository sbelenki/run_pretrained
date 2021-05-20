ssl=swav
log="${ssl}_fine_tune_logs.txt"
checkpoint=/home/ec2-user/vissl_orig/checkpoints_swav/model_final_checkpoint_phase29.torch
out_dir=./fine_tuning_$ssl
epochs=10

rm -rf $out_dir
rm -rf lightning_logs

python train_unet_demo.py \
	--data_path /home/ec2-user/mri \
	--max_epochs $epochs \
	--gpus 4 \
	--chans 64 \
	--unet_module nestedunet \
	--output_path $out_dir \
	--opt_lr 0.0007 \
	--opt_lr_step_size 3 \
	--opt_lr_gamma 0.5 \
	--state_dict_file $checkpoint >& $log

zip_out=fine_tune_"${ssl}_${epochs}ep_`date +"%m.%d.%Y-%H.%M.%S"`".zip
echo Compressing to: $zip_out
zip -r $zip_out $out_dir $log lightning_logs fine_tune.sh
aws s3 cp $zip_out s3://ylichman-dl-bucket/$ssl/$zip_out

echo DONE DONE DONE
sudo shutdown now
