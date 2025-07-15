from valis import registration, preprocessing, feature_detectors, affine_optimizer
import pyvips
from valis.micro_rigid_registrar import MicroRigidRegistrar
from valis.affine_optimizer import AffineOptimizerMattesMI
import time
import os
import numpy as np


slide_src_dir = "/data/he"
results_dst_dir = "/data/af"
registered_slide_dst_dir = "/data/afr"
reference_slide = "he.tiff"




# Create a Valis object and use it to register the slides in slide_src_dir

registrar = registration.Valis(slide_src_dir, results_dst_dir, reference_img_f=reference_slide, align_to_reference=True, check_for_reflections=True, non_rigid_registrar_cls=None,micro_rigid_registrar_cls=MicroRigidRegistrar)

rigid_registrar, non_rigid_registrar, error_df = registrar.register(brightfield_processing_cls=preprocessing.HEDeconvolution)

#registrar.register_micro(brightfield_processing_cls=preprocessing.HEDeconvolution, max_non_rigid_registration_dim_px=2000, align_to_reference=True)

# Perform high resolution non-rigid registration using 25% full resolution
# micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px=500, align_to_reference=True)

# Save all registered slides as ome.tiff
registrar.warp_and_save_slides(registered_slide_dst_dir, crop="reference", non_rigid=False)

# Kill the JVM
registration.kill_jvm()


