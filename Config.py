import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu";
LEARNING_RATE = 1e-5
BATCH_SIZE = 1
NUM_WORKERS = 4
IMAGE_SIZE = 512
NUM_EPOCHS = 40
EPSILON = 1e-5
NUM_CLASSES = 1;
PROJECT_NAME = '';
PROJECT_ROOT = '';
MAX_PEN_SIZE = 300;
FINISH_TRAINING = False;
PREDEFINED_COLORS = [[0, 0, 255], [255, 0, 0], [0, 255, 0], [102, 0, 204], [255, 102, 102], [255, 128, 0], [255, 255, 0]];
PREDEFINED_NAMES = ['Vertebra', 'Spinous process', 'Ribs', 'Thoracic Limbs', 'Pulmonary Arteries', 'Mediastinum', 'Trachea', 'Bronchi', 'Abdomen', 'Lung', 'Spine'];
PROJECT_PREDEFINED_NAMES = [];
NEXT_SAMPLE_SELECTION = 'Similarity';
 #Initialize transforms for training and validation
train_transforms = A.Compose(
[
    #A.PadIfNeeded(min_height = 512, min_width = 512),
    #A.RandomCrop(Config.IMAGE_SIZE, Config.IMAGE_SIZE, always_apply = False, p = 0.5),
    #A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
    #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.5),
    A.HorizontalFlip(p=0.5),
    #A.RandomBrightnessContrast(p=0.5),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2(),
],
additional_targets={'mask': 'mask'}
)

valid_transforms = A.Compose(
    [
    #A.PadIfNeeded(min_height = 512, min_width = 512),
    #A.RandomCrop(Config.IMAGE_SIZE, Config.IMAGE_SIZE, always_apply = True),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2()
    ]
)

