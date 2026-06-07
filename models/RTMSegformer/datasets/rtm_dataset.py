# ASCFormer/mmseg/datasets/rtm_dataset.py

from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset


@DATASETS.register_module()
class RTMDataset(BaseSegDataset):
    """
    Dataset RTM (binary segmentation).
    Espera:
      - img_path: pasta com imagens
      - seg_map_path: pasta com máscaras
      - split: ficheiro .txt com nomes (um por linha)
    """

    METAINFO = dict(
        classes=("background", "tamper"),
        palette=[[0, 0, 0], [255, 255, 255]],
    )

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix=".jpg",        # ajusta se fores usar .jpg
            seg_map_suffix=".png",
            reduce_zero_label=False,  # para binário (0/1) não precisas reduzir
            **kwargs,
        )