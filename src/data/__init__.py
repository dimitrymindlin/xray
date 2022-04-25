import glob
import re
import pandas as pd
from numpy import uint8
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import imutils
import os
import sys
from glob import glob
from matplotlib import pyplot as plt
from tqdm import tqdm

PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range
SHOW = True


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def find_squares(img, min_area=100000, max_skew=0.45):
    """
    A method to find inner square images on bigger images
    :param min_area: specifies minimal square area in pixels
    :param max_skew: specifies maximum skewness of squares
    :param img: numpy array representation of an image
    :return: list of found squares
    """
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)
            bin = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(bin)
            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if len(cnt) >= 4 and cv.contourArea(cnt) >= min_area \
                        and cv.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4],
                                                cnt[(i + 2) % 4])
                                      for i in xrange(4)])
                    if max_cos < max_skew:
                        squares.append(cnt)
    return squares


def crop_squares(squares, img):
    rect = cv.minAreaRect(squares[0])
    box = cv.boxPoints(rect)
    box = np.int0(box)

    if SHOW:
        cv.drawContours(img, [box], 0, (0, 0, 255), 2)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]],
                       dtype="float32")

    # the perspective transformation matrix
    M = cv.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv.warpPerspective(img, M, (width, height))

    if width > height:
        warped = imutils.rotate_bound(warped, 90)

    if SHOW:
        cv.imshow("crop_img.jpg", warped)

    return warped


class TrainValTestSplitter:

    def __init__(self, path_to_data, show_labels_dist=False, random_state=42):
        """
        Train-validation-test splitter, stores all the filenames
        :param path_to_data: for glob.glob to find all the images path
        :param show_labels_dist: show plot of distributions of labels
        """
        path_to_data = f'{path_to_data}/*/*/*'
        self.random_state = random_state
        self.data = pd.DataFrame()
        self.data['path'] = glob.glob(path_to_data)
        self.data['label'] = self.data['path'].apply(lambda path: len(re.findall('positive', path)))
        self.data['patient'] = self.data['path'].apply(lambda path: re.findall('[0-9]{5}', path)[0])
        if show_labels_dist:
            self.data['label'].hist()
            plt.title('Labels distribution')
            plt.show()
        self._split_data()

    def _split_stats(self, df):
        print(f'Size: {len(df)}')
        print(f'Percentage from original data: {len(df) / len(self.data)}')
        print(f'Percentage of negatives: {len(df[df.label == 0]) / len(df)}')
        print(f'Number of patients: {len(df.patient.unique())}')

    def _split_data(self):
        """
        Creates data_train, data_val, data_test dataframes with filenames
        """
        # train | validate test split
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=self.random_state)
        negative_data = self.data[self.data.label == 0]
        generator = splitter.split(negative_data.label, groups=negative_data['patient'])
        idx_train, idx_validate_test = next(generator)

        print('=================Train subset=================')
        self.data_train = negative_data.iloc[idx_train, :].reset_index(drop=True)
        self._split_stats(self.data_train)

        # validate | test split
        data_val_test = pd.concat(
            [self.data[self.data.label == 1], self.data.iloc[negative_data.iloc[idx_validate_test, :].index]])
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=self.random_state)
        generator = splitter.split(data_val_test.label, groups=data_val_test['patient'])
        idx_val, idx_test = next(generator)

        print('=============Validation subset===============')
        self.data_val = data_val_test.iloc[idx_val, :]
        self.data_val = self.data_val.sample(len(self.data_val)).reset_index(drop=True)
        self._split_stats(self.data_val)

        print('=================Test subset=================')
        self.data_test = data_val_test.iloc[idx_test, :]
        self.data_test = self.data_test.sample(len(self.data_test)).reset_index(drop=True)
        self._split_stats(self.data_test)


class MURASubset(Dataset):

    def __init__(self, filenames, transform=None, n_channels=1, true_labels=None, patients=None):
        """Initialization
        :param filenames: list of filenames, e.g. from TrainValTestSplitter
        :param true_labels: list of true labels (for validation and split)
        """
        self.transform = transform
        self.filenames = list(filenames)
        self.n_channels = n_channels
        self.true_labels = true_labels
        self.patients = patients

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        return len(self.filenames)

    def __getitem__(self, index) -> np.array:
        """Reads sample"""
        image = cv.imread(self.filenames[index])
        label = self.true_labels[index] if self.true_labels is not None else None
        patient = self.patients[index] if self.true_labels is not None else None
        filenames = self.filenames[index]

        sample = {'image': image, 'label': label, 'patient': patient,
                  'filename': filenames}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def save_transformed_samples(self):
        mura_dir = "/Users/dimitrymindlin/tensorflow_datasets/downloads/cjinny_mura-v11/MURA-v1.1"
        for filename in tqdm(self.filenames, maxinterval=len(self.filenames)):
            img = cv.imread(filename)
            if img is None:
                print(f"Couldn't load {filename}")
                continue

            sample = {'image': img, 'label': None, 'patient': None}
            sample = self.transform(sample)
            img = np.squeeze(sample['image'].numpy(), axis=0)

            img = cv.cvtColor(img, cv.COLOR_GRAY2RGB) * 255.0
            img = img.astype(uint8)
            squares = find_squares(img) # Needs rgd img
            cv.drawContours(img, squares, 0, (0, 255, 0), 3)
            if len(squares) > 0:
                img = crop_squares(squares, img)

            sample = {'image': img, 'label': None, 'patient': None}
            sample = self.transform(sample)
            img = np.squeeze(sample['image'].numpy(), axis=0)
            full_img_dir = mura_dir + "_transformed" + filename.split("MURA-v1.1")[-1]
            file_folder = "/".join(full_img_dir.split("/")[:-1])
            try:
                os.makedirs(file_folder)
            except FileExistsError:
                pass
            cv.imwrite(full_img_dir, img * 255.0)
