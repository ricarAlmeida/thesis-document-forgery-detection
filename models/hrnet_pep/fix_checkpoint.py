"""
Utility script to fix PyTorch checkpoints saved with DataParallel.

It removes the leading 'module.' prefix from state_dict keys so that
the checkpoint can be loaded without nn.DataParallel.
"""

from collections import OrderedDict
import torch

# ---------------------------------------------------------------------
# Paths (adjust if needed)
# ---------------------------------------------------------------------

INPUT_CHECKPOINT = (
    "./weights_100/"
    "hrnet-pep-batchnorm-q2_100_97-q3_90/"
    "2025-12-22 10:11:25.295153-checkpoint19.pth"
)

OUTPUT_CHECKPOINT = (
    "./weights_100/"
    "hrnet-pep-batchnorm-q2_100_97-q3_90/"
    "2025-12-22 10:11:25.295153-checkpoint19_fixed.pth"
)


# ---------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------

checkpoint = torch.load(INPUT_CHECKPOINT, map_location="cpu")
state_dict = checkpoint["state_dict"]


# ---------------------------------------------------------------------
# Remove 'module.' prefix from DataParallel checkpoints
# ---------------------------------------------------------------------

fixed_state_dict = OrderedDict()

for key, value in state_dict.items():
    new_key = key
    if new_key.startswith("module."):
        new_key = new_key.replace("module.", "", 1)
    fixed_state_dict[new_key] = value

checkpoint["state_dict"] = fixed_state_dict


# ---------------------------------------------------------------------
# Save fixed checkpoint
# ---------------------------------------------------------------------

torch.save(checkpoint, OUTPUT_CHECKPOINT)


# ---------------------------------------------------------------------
# Debug / sanity check
# ---------------------------------------------------------------------

print("✅ Fixed checkpoint saved to:", OUTPUT_CHECKPOINT)

old_key = next(iter(state_dict.keys()))
new_key = next(iter(fixed_state_dict.keys()))
print("Example key mapping:")
print(f"  {old_key}  ->  {new_key}")