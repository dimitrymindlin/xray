from typing import List

from torchvision.transforms import Compose
from src.data import MURASubset
from src.data.transforms import GrayScale, MinMaxNormalization, ToTensor
from src.data.transforms import AdaptiveHistogramEqualization

from src.models.run_params import COMMON_PARAMS

run_params = {**COMMON_PARAMS}
seed = run_params['random_seed'][0]
composed_transforms = Compose([GrayScale(),
                               AdaptiveHistogramEqualization(
                                   active=run_params['pipeline']['adaptive_hist_equilization']),
                               MinMaxNormalization(*run_params['pipeline']['normalisation']),
                               ToTensor()])

data_path = "dataset/MURA-v1.1/train"


def filenames(parts: List[str], train=True):
    root = '../tensorflow_datasets/downloads/cjinny_mura-v11/'
    # root = '/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/'
    if train:
        csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
        # csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/train_image_paths.csv"
    else:
        csv_path = "../tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"
        # csv_path = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1/valid_image_paths.csv"

    with open(csv_path, 'rb') as F:
        d = F.readlines()
        imgs = [root + str(x, encoding='utf-8').strip() for x in d if
                str(x, encoding='utf-8').strip().split('/')[2] in parts]

    # imgs= [x.replace("/", "\\") for x in imgs]
    labels = [x.split('_')[-1].split('/')[0] for x in imgs]
    return imgs, labels


train_filenames, labels = filenames(["XR_HAND"])
valid_filenamse, labels = filenames(["XR_HAND"], train=False)
train = MURASubset(filenames=train_filenames, transform=composed_transforms)
validation = MURASubset(filenames=valid_filenamse, transform=composed_transforms)

train.save_transformed_samples()
validation.save_transformed_samples()
