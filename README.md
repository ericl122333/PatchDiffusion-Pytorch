# Patch Diffusion

**UPDATE (Mar 2024): Unfortunately, the model checkpoints were lost.** They were accidentally deleted when I was clearing my personal google drive storage. Hopefully this doesnt cause too much of a detriment. (At this point the patching technique we use here has become pretty commonplace among diffusion transformers. For those interested in ImageNet-scale models with open weights the [DiT repo](https://github.com/facebookresearch/DiT) might be a good starting point.)

Code for the paper "Improving Diffusion Model Efficiency Through Patching". The core idea of the paper is to insert a ViT-style patching operation at the beginning of the U-Net, letting it operate on data with smaller height and width. We show in our paper that the optimal prediction for **x** is quite blurry for most timesteps, and therefore convolutions at the original resolution are usually not necessary. This causes a considerable reduction in compute cost: For example, when using a patch size of 4 (P = 4), generating 256x256 images costs only as much as generating 64x64 images normally (with P = 1). 

# Pretrained Models

**UPDATE: per the above message, the links in this section are broken**

We include our models for ImageNet 256x256 and FFHQ 1024x1024, as well as 3 LSUN models with P=2, P=4, and P=8. 

You can download them from Google Drive:

 * ImageNet 256x256, Split #0: [imagenet_weights_0.pt](https://drive.google.com/file/d/1--FE31CNDsCqa_ihGaJIwVSoELdwIAfC/view?usp=sharing)
 * ImageNet 256x256, Split #1: [imagenet_weights_1.pt](https://drive.google.com/file/d/1-9kmLKUR1fDVckHY0i_83xzV3QHzuDDC/view?usp=sharing)
 * FFHQ 1024x1024: [ffhq_weights.pt](https://drive.google.com/file/d/1-4Len8DL1ZzBv---oNurw5UQQrS0tVuQ/view?usp=sharing)
 * LSUN 256x256, P=2: [lsun_weights_p2.pt](https://drive.google.com/file/d/1pjQzsyiNWSlyp2HcxSUBf9Hh0EQXr2ES/view?usp=sharing)
 * LSUN 256x256, P=4: [lsun_weights_p4.pt](https://drive.google.com/file/d/1-4-e9M2xzmGd2tCTDwcZd6B0m36AqKvz/view?usp=sharing)
 * LSUN 256x256, P=8: [lsun_weights_p8.pt](https://drive.google.com/file/d/1-7wvb5coEdoKEmtixBPg_kZbcNXpnApN/view?usp=sharing)

# Sampling Instructions

First, clone our repository and change directory into it. 

Then do 

```
pip install -e .
```

Assuming you have downloaded the relevant models in ./models, run the following code to sample from our models. It will save the images (in a PNG file) and npz arrays to ./results

FFHQ:
```
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing 250"
MODEL_FLAGS="--channel_mult '1,2,2,4,4,4' --class_cond False --patch_size 4 --image_size 1024 --learn_sigma True --noise_schedule linear0.025 --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --use_fp16 True --use_scale_shift_norm True --use_new_attention_order True"
python scripts/image_sample.py $MODEL_FLAGS --save_dir './results' --model_path ./models/ffhq_weights.pt $SAMPLE_FLAGS 
```

ImageNet256:

For ImageNet, we used two techniques to reduce computation cost and boost sample quality: [classifier-free guidance](https://openreview.net/forum?id=qw8AKxfYbI) and splitting. When increasing the guidance scale, classifier-free guidance improves the visual quality of samples, at the expense of sample diversity. 1.5 is a good default value, although larger values such as 2.25 work as well. Unguided sampling (guidance_scale 1.0) is faster, but generally doesn't lead to as good results.

Splitting uses 2 (or more) different diffusion models during the generation process, where each model learns to denoise data for part of the diffusion process. In our case, one diffusion model denoises data where the signal-to-noise (SNR) ratio of the data is above 0.25, while the other denoises data below 0.25 SNR. Therefore, we set snr_splits to '0.25', and pass in the checkpoint paths to *two* different models in the ```model_path``` argument. Both models need to be downloaded to run ImageNet sampling.

```
SAMPLE_FLAGS="--batch_size 4 --num_samples 4 --guidance_scale 1.5 --timestep_respacing 250"
MODEL_FLAGS="--snr_splits '0.25' --channel_mult '1,2,2,2' --class_cond True --patch_size 4 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 3 --use_fp16 True --use_scale_shift_norm True --use_new_attention_order True"
!python scripts/image_sample.py $MODEL_FLAGS --save_dir './results' --model_path './models/imagenet_weights_0.pt,./models/imagenet_weights_1.pt' $SAMPLE_FLAGS 
```

LSUN, with P=2:
```
SAMPLE_FLAGS="--batch_size 4 --num_samples 4 --timestep_respacing 250"
MODEL_FLAGS="--channel_mult '1,2,2,4,4' --class_cond False --patch_size 2 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 128 --num_head_channels 64 --num_res_blocks 2 --use_fp16 True --use_scale_shift_norm True --use_new_attention_order True"
!python scripts/image_sample.py $MODEL_FLAGS --save_dir './results' --model_path ./models/lsun_weights_p2.pt $SAMPLE_FLAGS 
```

LSUN, with P=4:
```
SAMPLE_FLAGS="--batch_size 4 --num_samples 4 --timestep_respacing 250"
MODEL_FLAGS="--channel_mult '1,1,2,2' --class_cond False --patch_size 4 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --use_fp16 True --use_scale_shift_norm True --use_new_attention_order True"
!python scripts/image_sample.py $MODEL_FLAGS --save_dir './results' --model_path ./models/lsun_weights_p4.pt $SAMPLE_FLAGS 
```

LSUN, with P=8:
```
SAMPLE_FLAGS="--batch_size 4 --num_samples 4 --timestep_respacing 250"
MODEL_FLAGS="--channel_mult '1,1.5,2' --class_cond False --patch_size 8 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 3 --use_fp16 True --use_scale_shift_norm True --use_new_attention_order True"
!python scripts/image_sample.py $MODEL_FLAGS --save_dir './results' --model_path ./models/lsun_weights_p8.pt $SAMPLE_FLAGS 
```

# Training:
Make sure all your training images are in the directory data_dir in png/jpg format (they can be in subdirectories). Then, define MODEL_FLAGS, DIFFUSION_FLAGS, and TRAIN_FLAGS. 

For example: 
```
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 2 --channel_mult '1,1,2,2' --patch_size 4 --learn_sigma True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
```

Through the ```--weight_schedule``` argument, we also support different weight schedules of $\lambda_t$, where the objective is $\lambda_t\lVert\textbf{x} - \textbf{x}_\theta(\textbf{z}_t, t)  \rVert$. The default is "sqrt_snr", where $\lambda_t$ = $\sqrt{\frac{\alpha}{1-\alpha}}$. However we also include support for the [P2 loss weighting](https://arxiv.org/abs/2204.00227) ("p2") , as well as "snr", "snr+1" and "truncated_snr" schedules from the [progressive distillation](https://arxiv.org/abs/2202.00512) paper.

To train a model with splitting, add ``` "--snr_splits '{snr_split_values}'" ``` to MODEL_FLAGS and add ```--schedule_sampler uniform_split_{num}``` where num is the split index starting from 0. Model splitting is described in Section 4.1 of the paper. Note: the SNR is defined as alpha/(1-alpha).



Then use:
```
python scripts/image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```


We trained our models for a relatively short duration: our ImageNet models trained for a combined 32 V-100 days (approximately), while our FFHQ model trained for roughly 14 V-100 days. Our LSUN models trained for about 5 V-100 days each. In general, longer training is recommended if you have the budget for it - it improves results.

# Acknowledgements:

Our repository builds on top of ADM's [guided diffusion](https://github.com/openai/guided-diffusion) repository - Thanks for sharing!

# Citation:

If you find this work helpful to your research, please cite us:

@misc{https://doi.org/10.48550/arxiv.2207.04316,
  doi = {10.48550/ARXIV.2207.04316},
  
  url = {https://arxiv.org/abs/2207.04316},
  
  author = {Luhman, Troy and Luhman, Eric},
  
  keywords = {Machine Learning (cs.LG), Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Improving Diffusion Model Efficiency Through Patching},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
