model = dict(
    type = 'IcesatMAEMamba',
    mask_ratio = 0.6,
    mask_type = 'rand',
    trans_dim = 384,
    depth = 12,
    drop_path_rate = 0.1,
    num_heads= 6,
    decoder_depth = 4,
    decoder_num_heads = 6,
    rms_norm = False,
    drop_path = 0.1,
    drop_out = 0.1,
    drop_out_in_block = 0.1,
    fused_add_norm = False
)