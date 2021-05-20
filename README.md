# Self-MRI Subrepo

## Used as a part of:

### Fine-tuning

Run modified train_unet_demo.py from the `Self-MRI/run_pretrained` folder and start fine-tuning using the following line as an example for SwAV:

```
python train_unet_demo.py \
--data_path /path/to/data/ \
--max_epochs 10 \
--gpus 4 \
--output_path /output/of/finetuned/model \
--unet_module nestedunet \
--state_dict_file path/to/your/model \
```
This procedure will fine-tune your pre-trainded model.


## Testing and Inferenece 

* Navigate to `run_pretrained` directory

To run the inference on the final models (U-Net or U-Net++), use `run_pretrained_unet_inference.py` script.  The output is a directory of H5 files, each corresponding to the appropriate input test file.

For example:

```
python run_pretrained_unet_inference.py \
  --device cpu \
  --unet_module unet \
  --state_dict_file pirl.torch \
  --output_path ./results \
  --data_path /path/to/data/test \
```

To generate test image(s) for the final models, use `createImageFromH5.py`:

```
python createImageFromH5.py  --h5file ./results/reconstructions/file1000000.h5
```
