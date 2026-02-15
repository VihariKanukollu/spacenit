"""SpaceNit neural-network architecture modules.

New modules (deep rewrite):
- :mod:`attention` -- GQA, RMSNorm, SwiGLU, post-norm transformer
- :mod:`embed` -- Adaptive patch embed, spatial/temporal RoPE, cyclic month, sensor embed
- :mod:`encoder` -- Multi-sensor tokenizer, encoder, decoder
- :mod:`heads` -- Pixel, projection, and pooling heads
- :mod:`models` -- LatentPredictor, AutoEncoder, DualBranch, SpatioTemporalEncoder

Legacy modules (band_tokenization, helpers) remain for backward compatibility.
"""
