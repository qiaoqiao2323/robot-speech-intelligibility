from scipy import signal
import numpy as np
import os
from multiprocessing import Pool
import multiprocessing as mp


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


class LogMelExtractor():
    def __init__(self, sample_rate, window_size, overlap, mel_bins):
        self.window_size = window_size
        self.overlap = overlap
        self.ham_win = np.hamming(window_size)

        self.melW = librosa.filters.mel(sr=sample_rate,
                                        n_fft=window_size,
                                        n_mels=mel_bins,
                                        fmin=50.,
                                        fmax=sample_rate // 2).T

    def transform(self, audio):
        ham_win = self.ham_win
        window_size = self.window_size
        overlap = self.overlap

        # print(len(audio), len(audio)/44100)
        # print(window_size, overlap)
        # # 663024 15.034557823129251
        # # 2048 672

        [f, t, x] = signal.spectral.spectrogram(
            audio,
            window=ham_win,
            nperseg=window_size,
            noverlap=overlap,
            detrend=False,
            return_onesided=True,
            mode='magnitude')
        x = x.T
        # print(x.shape)


        x = np.dot(x, self.melW)
        x = np.log(x + 1e-8)
        x = x.astype(np.float32)

        return x



import librosa
def read_audio(path, target_fs=None):
    # print(path)
    audio, fs = librosa.load(path)  # mp3
    # print(audio.shape)
    # (331512,)

    if audio.ndim > 1:
        # 之前默认都是 平均的
        audio = np.mean(audio, axis=1)

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    # print(len(audio), fs)  # 663024 44100
    return audio, fs


def calculate_logmel(audio_path, sample_rate, feature_extractor):
    # Read audio
    (audio, fs) = read_audio(audio_path, target_fs=sample_rate)

    '''We do not divide the maximum value of an audio here because we assume 
    the low energy of an audio may also contain information of a scene. '''

    feature = feature_extractor.transform(audio)

    return feature


def run_jobs():
    source_dir = os.path.join(os.getcwd(), 'AR_test')

    sample_rate = 44100
    window_size = 2048
    overlap = 672  # So that there are 320 frames in an audio clip
    seq_len = 320
    mel_bins = 64

    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate,
                                        window_size=window_size,
                                        overlap=overlap,
                                        mel_bins=mel_bins)

    ################################### start ###################################################################

    audio_path = os.path.join(source_dir, '0_audios')

    output_mel = audio_path + '_mel64'
    create_folder(output_mel)

    seq_len = int(15*32)

    # 这里出错了，原来是这样，因为os.listdir(audio_path)中的audio_name和audio_names中的不一样！！！！
    for num, audio_name in enumerate(os.listdir(audio_path)):
        # print(num, audio_name)
        if audio_name.endswith('.mp3'):
            feature_file = os.path.join(output_mel, audio_name.replace('.mp3', '.npy'))

            if not os.path.exists(feature_file):
                audio_file = os.path.join(audio_path, audio_name)
                #########################################################################################
                feature = calculate_logmel(audio_path=audio_file,
                                           sample_rate=sample_rate,
                                           feature_extractor=feature_extractor)
                '''(seq_len, mel_bins)'''

                feature = feature[:seq_len]

                print('n, audio_name: ', num, audio_name, ' shape: ', feature.shape)
                # n, audio_name:  0 airport-barcelona-0-0-a.wav  shape: 10s (320, 64)
                # n, audio_name:  0 2cv11_1.mp3  shape: 15s (481, 64)

                np.save(feature_file, feature)
            else:
                print('Done: ', feature_file)




def main():
    cpu_num = mp.cpu_count()
    print('cpu_num: ', cpu_num)

    cpu_num = 1
    pool = Pool(cpu_num)
    # pool=Pool(最大的进程数)
    # 然后添加多个需要执行的进程，可以大于上面设置的最大进程数，会自动按照特定的队列顺序执行

    if cpu_num==1:
        pool.apply_async(func=run_jobs, args=())
    else:
        for i in range(cpu_num):
            pool.apply_async(func=run_jobs, args=())

    pool.close()
    pool.join()
#     join(): 等待工作进程结束。调用 join() 前必须先调用 close() 或者 terminate() 。



if __name__ == '__main__':
    main()









