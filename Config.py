import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
NUM_WORKERS = 1
IMAGE_SIZE = 1024
NUM_EPOCHS = 500
EPSILON = 1e-6
NUM_CLASSES = 1;
PROJECT_NAME = '';
PROJECT_ROOT = '';
MAX_PEN_SIZE = 300;
FINISH_TRAINING = False;
PREDEFINED_COLORS = ['blue', 'red', 'purple', 'cyan', 'magenta', 'orange', 'yellow'];
PREDEFINED_NAMES = ['Vertebra', 'Spinous process', 'Ribs', 'Thoracic Limbs', 'Pulmonary Arteries', 'Mediastinum', 'Trachea', 'Bronchi', 'Abdomen', 'Lung', 'Spine'];
PROJECT_PREDEFINED_NAMES = [];