import numpy as np
import h5py, os, pickle, torch
import time
from framework.utilities import calculate_scalar, scale
import framework.config as config



class DataGenerator(object):
    def __init__(self, batch_size=config.batch_size, seed=42, dataset_path=None, normalization=False, split_type=None):

        ############### data split #######################################################
        test_file = os.path.join(dataset_path, 'test_audio_id.txt')
        if not os.path.exists(test_file):
            print('data spliting... :', split_type)
            data_split(dataset_path, dir_name=split_type)

        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        self.validate_random_state = np.random.RandomState(0)
        self.test_random_state = np.random.RandomState(0)

        # Load data
        load_time = time.time()

        file_path = os.path.join(dataset_path, 'training.pickle')
        print('using: ', file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.train_audio_ids, self.train_rates, self.train_event_label = \
            data['audio_ids'], data['rates'], data['event_label']
        self.train_x = data['x']

        file_path = os.path.join(dataset_path, 'validation.pickle')
        print('using: ', file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.val_audio_ids, self.val_rates, self.val_event_label = \
            data['audio_ids'], data['rates'], data['event_label']
        self.val_x = data['x']

        file_path = os.path.join(dataset_path, 'test.pickle')
        print('using test: ', file_path)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        self.test_audio_ids, self.test_rates, self.test_event_label = \
            data['audio_ids'], data['rates'], data['event_label']
        self.test_x = data['x']

        ############################################################################################

        print('Loading data time: {:.3f} s'.format(time.time() - load_time))

        print('Split development data to {} training and {} '
              'validation data and {} '
              'test data. '.format(len(self.train_audio_ids),
                                         len(self.val_audio_ids),
                                   len(self.test_audio_ids)))

        self.normal = normalization
        if self.normal:
            (self.mean, self.std) = calculate_scalar(self.train_x)


    def load_x(self, audio_ids):
        if os.path.exists(r'D:\Yuanbo\Code\GPULab\Code\UCL\meta_data'):
            root = r'D:\Yuanbo\Code\GPULab\Code\UCL\meta_data'
        elif os.path.exists(r'E:\Yuanbo\UCL\DeLTA\meta_data'):
            root = r'E:\Yuanbo\UCL\DeLTA\meta_data'
        elif os.path.exists(r'D:\Yuanbo\Code\UCL\meta_data'):
            root = r'D:\Yuanbo\Code\UCL\meta_data'
        elif os.path.exists('/project_antwerp/yuanbo/Code/UCL/meta_data'):
            root = '/project_antwerp/yuanbo/Code/UCL/meta_data'
        elif os.path.exists('/project_scratch/yuanbo/Code/UCL/meta_data'):
            root = '/project_scratch/yuanbo/Code/UCL/meta_data'

        x_list = []
        for each_id in audio_ids:
            idfile = os.path.join(root, 'DeLTA_mp3_boost_8dB_mel64', each_id.replace('.mp3', '.npy'))
            x_list.append(np.load(idfile))
        return np.array(x_list)


    def load_rate_event_id(self, sub_set, dataset_path):
        audio_id_file = os.path.join(dataset_path, sub_set + '_audio_id.txt')
        rate_file = os.path.join(dataset_path, sub_set + '_annoyance_rate.txt')
        event_label_file = os.path.join(dataset_path, sub_set + '_sound_source.txt')

        audio_ids = []
        with open(audio_id_file, 'r') as f:
            for line in f.readlines():
                part = line.split('\n')[0]
                if part:
                    audio_ids.append(part)
        rates = np.loadtxt(rate_file)[:, None]
        event_label = np.loadtxt(event_label_file)
        return audio_ids, rates, event_label

    def generate_train(self):
        audios_num = len(self.train_audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        self.random_state.shuffle(audio_indexes)

        iteration = 0
        pointer = 0

        while True:
            if pointer >= audios_num:
                pointer = 0
                self.random_state.shuffle(audio_indexes)

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_x = self.train_x[batch_audio_indexes]
            batch_y = self.train_rates[batch_audio_indexes]
            batch_y_event = self.train_event_label[batch_audio_indexes]
            if self.normal:
                batch_x = self.transform(batch_x)

            yield batch_x, batch_y, batch_y_event


    def generate_validate(self, data_type, max_iteration=None):
        audios_num = len(self.val_audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        self.validate_random_state.shuffle(audio_indexes)

        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_x = self.val_x[batch_audio_indexes]
            batch_y = self.val_rates[batch_audio_indexes]
            batch_y_event = self.val_event_label[batch_audio_indexes]

            if self.normal:
                batch_x = self.transform(batch_x)

            yield batch_x, batch_y, batch_y_event


    def generate_testing(self, data_type, max_iteration=None):
        audios_num = len(self.test_audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        self.test_random_state.shuffle(audio_indexes)

        print('Number of {} audios in {}'.format(len(audio_indexes), data_type))

        iteration = 0
        pointer = 0

        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            batch_audio_indexes = audio_indexes[pointer: pointer + self.batch_size]
            pointer += self.batch_size

            iteration += 1
            batch_x = self.test_x[batch_audio_indexes]
            batch_y = self.test_rates[batch_audio_indexes]
            batch_y_event = self.test_event_label[batch_audio_indexes]

            if self.normal:
                batch_x = self.transform(batch_x)

            yield batch_x, batch_y, batch_y_event


    def transform(self, x):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        return scale(x, self.mean, self.std)


