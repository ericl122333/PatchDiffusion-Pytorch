def make_th_model_ffhq():
    defaults = dict(
        clip_denoised=True,
        batch_size=8,
        use_ddim=False,
        model_path=None,
        timestep_respacing=1000,
        num_samples=1000
    )
    defaults.update(model_and_diffusion_defaults())
    
    args = ConfigDict(defaults)

    args.attention_resolutions = '16,8'
    args.channel_mult = '1,2,2,4,4,4'
    args.use_conv=False
    args.class_cond=False
    args.image_size=1024

    args.patch_size=4
    args.classifier_free=True #doesnt matter for unconditional models.
    args.output_type='x0' #doesnt matter for unet, move arg to diffusion object
    args.modified_arch=True
    
    args.learn_sigma=True
    args.num_channels=128
    args.resblock_updown=False
    args.use_scale_shift_norm=True
    args.use_fp16 = False
    args.use_new_attention_order = True
    args.num_head_channels = 64
    
    print(args_to_dict(args, model_and_diffusion_defaults().keys()))
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.eval()
    return model

def make_th_model_net():
    defaults = dict(
        clip_denoised=True,
        batch_size=8,
        use_ddim=False,
        model_path=None,
        timestep_respacing=1000,
        num_samples=1000
    )
    defaults.update(model_and_diffusion_defaults())
    
    args = ConfigDict(defaults)

    args.attention_resolutions = '16,8'
    args.channel_mult = '1,2,2,2'
    args.use_conv=False
    args.class_cond=True
    args.image_size=256

    args.patch_size=4
    args.classifier_free=True #doesnt matter for unconditional models.
    args.output_type='x0' #doesnt matter for unet, move arg to diffusion object
    args.modified_arch=True
    
    args.learn_sigma=True
    args.num_channels=256
    args.num_res_blocks=3
    args.resblock_updown=False
    args.use_scale_shift_norm=True
    args.use_fp16 = False
    args.use_new_attention_order = True
    args.num_head_channels = 64
    
    print(args_to_dict(args, model_and_diffusion_defaults().keys()))
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.eval()
    return model