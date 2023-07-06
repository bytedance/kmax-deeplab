from . import data  # register all new datasets
from . import modeling

# config
from .config import add_kmax_deeplab_config

# dataset loading
from .data.dataset_mappers.panoptic_kmaxdeeplab_dataset_mapper import PanoptickMaXDeepLabDatasetMapper



# models
from .kmax_model import kMaXDeepLab

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
