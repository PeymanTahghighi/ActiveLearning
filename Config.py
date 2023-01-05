import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu";
LEARNING_RATE = 1e-5
BATCH_SIZE = 2
NUM_WORKERS = 4
IMAGE_SIZE = 1024
VIRUTAL_BATCH = 2;
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
NEXT_SAMPLE_SELECTION = 'In order';
MUTUAL_EXCLUSION = True;


 #Initialize transforms for training and validation
train_transforms = A.Compose(
[
    A.Resize(IMAGE_SIZE,IMAGE_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2(),
],
additional_targets={'mask': 'mask'}
)

valid_transforms = A.Compose(
    [
    A.Resize(IMAGE_SIZE,IMAGE_SIZE),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2()
    ]
)


predict_transforms = A.Compose(
    [
    A.LongestMaxSize(max_size = IMAGE_SIZE),
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ToTensorV2()
    ]
)