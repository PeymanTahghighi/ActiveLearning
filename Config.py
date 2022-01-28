import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu";
LEARNING_RATE = 5e-5
BATCH_SIZE = 2
NUM_WORKERS = 1
IMAGE_SIZE = 1024
NUM_EPOCHS = 40
EPSILON = 1e-6
NUM_CLASSES = 1;
PROJECT_NAME = '';
PROJECT_ROOT = '';
MAX_PEN_SIZE = 300;
FINISH_TRAINING = False;
PREDEFINED_COLORS = [[0, 0, 255], [255, 0, 0], [0, 255, 0], [102, 0, 204], [255, 102, 102], [255, 128, 0], [255, 255, 0]];
PREDEFINED_NAMES = ['Vertebra', 'Spinous process', 'Ribs', 'Thoracic Limbs', 'Pulmonary Arteries', 'Mediastinum', 'Trachea', 'Bronchi', 'Abdomen', 'Lung', 'Spine'];
PROJECT_PREDEFINED_NAMES = [];