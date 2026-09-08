"""Importing this package registers every SFT source module.

Adding a dataset: write sources/your_source.py with a @register("name")
class, then add one import line below -- scripts/prepare_sft_data.py picks
it up automatically from there via src.sft.registry.
"""

from src.sft.sources import aya_hi, euroblocks, indic_instruct_hi, smoltalk2pt

__all__ = ["aya_hi", "euroblocks", "indic_instruct_hi", "smoltalk2pt"]
