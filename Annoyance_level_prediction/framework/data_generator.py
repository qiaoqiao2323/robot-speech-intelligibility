import numpy as np
import h5py, os
from framework.utilities import scale


class DataGenerator(object):
    def __init__(self, pretrained_model_dir, mel_path):

        # Load data
        #############################################################################################################
        npy_path = mel_path
        self.val_audio_ids = []
        self.val_x = []
        for each in os.listdir(npy_path):
            if each.endswith('.npy'):
                self.val_audio_ids.append(each)
                eachpath = os.path.join(npy_path, each)
                self.val_x.append(np.load(eachpath))

        self.val_x = np.array(self.val_x)
        # print(len(self.val_audio_ids))
        # print(self.val_x.shape)  # (954, 480, 64)
        # 3
        # (3, 480, 64)
        ############################################################################################
        scalar_fn = os.path.join(pretrained_model_dir, 'scalar.h5')
        with h5py.File(scalar_fn, 'r') as hf:
            self.mean = hf['mean'][:]
            self.std = hf['std'][:]
        # print('self.mean shape: ', self.mean.shape)
        # print('self.std shape: ', self.std.shape)


    def generate_evaluate_clips(self):
        audios_num = len(self.val_audio_ids)
        audio_indexes = [i for i in range(audios_num)]

        iteration = 0
        pointer = 0

        while True:
            # Reset pointer
            if pointer >= audios_num:
                break

            batch_size = 1
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            iteration += 1
            batch_x = self.val_x[batch_audio_indexes]
            batch_ids = [self.val_audio_ids[each] for each in batch_audio_indexes]

            batch_x = self.transform(batch_x)
            yield batch_x, batch_ids

    def transform(self, x):
        """Transform data.

        Args:
          x: (batch_x, seq_len, freq_bins) | (seq_len, freq_bins)

        Returns:
          Transformed data.
        """

        return scale(x, self.mean, self.std)















