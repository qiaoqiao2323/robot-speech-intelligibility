import sys, os

# 这里的0是GPU id
import time

gpu_id = -1 # 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.data_generator import *
from framework.processing import *
from framework.models_pytorch import *
from framework.extract_features import *
import framework.config as config


def extract_feature(audio_path, output_mel):

    sample_rate = 44100
    window_size = 2048
    overlap = 672  # So that there are 320 frames in an audio clip
    mel_bins = 64

    # Feature extractor
    feature_extractor = LogMelExtractor(sample_rate=sample_rate,
                                        window_size=window_size,
                                        overlap=overlap,
                                        mel_bins=mel_bins)

    ################################### start ###################################################################



    create_folder(output_mel)

    seq_len = int(15 * 32)

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

def main(argv):
    start_time = time.time()
    x = 1
    # print(x)
    root_dir = os.path.join(os.getcwd(), 'AR_test')
    mp3_audio_dir_name = '0_audios'
    audio_path = os.path.join(root_dir, mp3_audio_dir_name)
    output_mel = audio_path + '_mel64'

    extract_feature(audio_path, output_mel)

    feature_end_time = time.time()
    # print('feature time: {:.5f} s'.format(feature_end_time-start_time))
    ############################################# model ########################################################
    pretrained_model_dir = os.path.join(os.getcwd(), 'model')
    using_model = PANN
    event_class = len(config.event_labels)
    model = using_model(event_num=event_class)
    # print(model)

    model_name = 'rate_best_pytorchv2.pth'
    model_path = os.path.join(pretrained_model_dir, model_name)

    model_event = torch.load(model_path, map_location='cpu')
    model.load_state_dict(model_event['state_dict'])

    cuda = config.cuda
    if config.cuda:
        model.cuda()

    ###########################################################################################################
    generator = DataGenerator(pretrained_model_dir, output_mel)
    generate_func = generator.generate_evaluate_clips()

    # Forward
    dict = forward_asc_aec(model=model, generate_func=generate_func, cuda=cuda)

    # rate loss
    annoyance_rate = dict['output']
    all_ids = dict['all_ids']

    event = dict['outputs_events']

    top_event = 5
    for i, rate in enumerate(annoyance_rate):
        audio_id = all_ids[i]
        f_event = [config.event_labels[k] for k in np.argsort(event[i])[::-1]][:top_event]
        # print('audio id: ', audio_id, ' , annoyance rate: ', rate[0],)
        # print('Max Pro fine event: ', f_event, '\n')

    predict_time = time.time() - feature_end_time

    # print('predict_time: {:.5f} s'.format(predict_time))

    sys.stdout.write(str(rate[0]))
    sys.stdout.flush()



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















