import numpy as np
import os
from sklearn.model_selection import train_test_split
import framework.config as config


def create_folder(feature_dir):
    """ 如果目录有多级，则创建最后一级。如果最后一级目录的上级目录有不存在的，则会抛出一个OSError。   """
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)


def get_all_data(root_dir):
    filename = 'DeLTA-collapsed-majority_2022-08-09.txt'
    label_file = os.path.join(root_dir, filename)

    with open(label_file, 'r') as f:
        event_label = f.readlines()[0].split('\n')[0].split('\t')[1:-3]

    audio_id_list = []
    label_matrix = []
    annoyance_rate_list = []
    with open(label_file, 'r') as f:
        for line in f.readlines()[1:]:
            part = line.split('\n')[0].split('\t')
            audio_id = part[0]
            events = [int(each) for each in part[1:25]]
            annoyance_rate = float(part[-1])

            audio_id_list.append(audio_id)
            label_matrix.append(events)
            annoyance_rate_list.append(annoyance_rate)

    label_matrix = np.array(label_matrix)
    audio_id_list = np.array(audio_id_list)
    annoyance_rate_list = np.array(annoyance_rate_list)

    return event_label, label_matrix, audio_id_list, annoyance_rate_list


def save_data(dir_path, data_type, audio_id_list, sub_id, annoyance_rate_list, label_matrix):
    filename = os.path.join(dir_path, data_type + 'audio_id.txt')
    sub_audio_id_list = audio_id_list[sub_id]
    with open(filename, 'w') as f:
        for i in range(len(sub_audio_id_list)):
            f.write(sub_audio_id_list[i] + '\n')
    filename = os.path.join(dir_path, data_type + 'annoyance_rate.txt')
    sub_annoyance_rate_list = annoyance_rate_list[sub_id]
    np.savetxt(filename, sub_annoyance_rate_list, fmt='%.2f')
    filename = os.path.join(dir_path, data_type + 'sound_source.txt')
    sub_label_matrix = label_matrix[sub_id]
    np.savetxt(filename, sub_label_matrix, fmt='%d')
    # print(len(sub_audio_id_list), len(sub_annoyance_rate_list), sub_label_matrix.shape)
    # 1936 1936 (1936, 24)




def data_split(root_dir, dir_name, test_size=0.1538, val_size=0.1):
    event_label, label_matrix, audio_id_list, annoyance_rate_list = get_all_data(root_dir)

    all_id = [i for i in range(len(label_matrix))]
    train_val_id, test_id = train_test_split(all_id, test_size=test_size, random_state=42)

    train_id, val_id = train_test_split(train_val_id, test_size=val_size, random_state=42)
    # print(len(train_id), len(val_id), len(test_id))
    # print(len(train_id) + len(val_id) + len(test_id))
    # # 2200 245 445
    # # 2890

    dir_path = os.path.join(root_dir, dir_name)
    create_folder(dir_path)

    sub_id = train_id
    data_type = 'train_'
    save_data(dir_path, data_type, audio_id_list, sub_id, annoyance_rate_list, label_matrix)

    sub_id = val_id
    data_type = 'validation_'
    save_data(dir_path, data_type, audio_id_list, sub_id, annoyance_rate_list, label_matrix)

    sub_id = test_id
    data_type = 'test_'
    save_data(dir_path, data_type, audio_id_list, sub_id, annoyance_rate_list, label_matrix)




