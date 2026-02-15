"""Data ingestion utilities for SpaceNit.

New modules (deep rewrite):
- :mod:`geotiff_reader` -- Read GeoTIFFs from HuggingFace dataset layout
- :mod:`hf_dataset` -- PyTorch Dataset/DataLoader for HF geospatial data
- :mod:`augmentations` -- Dihedral, CutMix, SensorDropout transforms

Legacy modules (sensors, standardizer, modalities shim) remain for compatibility.
"""
