import torch, os

if os.path.exists(r'D:\Yuanbo\Code\GPULab\Code\13_interspeech2023\0_meta_data'):
    root = r'D:\Yuanbo\Code\GPULab\Code\13_interspeech2023\0_meta_data'
elif os.path.exists(r'D:\Yuanbo\Code\13_interspeech2023\0_meta_data'):
    root = r'D:\Yuanbo\Code\13_interspeech2023\0_meta_data'
elif os.path.exists('/project_antwerp/yuanbo/Code/13_interspeech2023/0_meta_data'):
    root = '/project_antwerp/yuanbo/Code/13_interspeech2023/0_meta_data'
elif os.path.exists('/project_scratch/yuanbo/Code/13_interspeech2023/0_meta_data'):
    root = '/project_scratch/yuanbo/Code/13_interspeech2023/0_meta_data'
elif os.path.exists('/project_antwerp/qq/Code/13_interspeech2023/0_meta_data'):
    root = '/project_antwerp/qq/Code/13_interspeech2023/0_meta_data'
elif os.path.exists('/project_scratch/qq/Code/13_interspeech2023/0_meta_data'):
    root = '/project_scratch/qq/Code/13_interspeech2023/0_meta_data'
####################################################################################################

cuda = 0 # 1

training = 1
testing = 1

if cuda:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')

mel_bins = 64
batch_size = 64
epoch = 1000
lr_init = 1e-3



event_labels = ['Aircraft', 'Bells', 'Bird tweet', 'Bus', 'Car', 'Children', 'Construction',
                'Dog bark', 'Footsteps', 'General traffic', 'Horn', 'Laughter', 'Motorcycle', 'Music',
                'Non-identifiable', 'Other', 'Rail', 'Rustling leaves', 'Screeching brakes', 'Shouting',
                'Siren', 'Speech', 'Ventilation', 'Water']

sort_object_event_labels = ['Aircraft', 'Bus', 'Car', 'General traffic', 'Motorcycle', 'Rail', 'Screeching brakes',
                'Bells', 'Music',
                'Bird tweet', 'Dog bark',
                'Children', 'Laughter',  'Speech', 'Shouting', 'Footsteps',
                'Siren', 'Horn',
                'Rustling leaves', 'Water',
                'Construction', 'Non-identifiable', 'Other', 'Ventilation', ]

source_to_sort_indices = [sort_object_event_labels.index(each) for each in event_labels]
# print(source_to_sort_indices)

subject_labels = ['Vehicle',
                  'Music',
                  'Animals',
                  'Human sounds',
                  'Alarm',
                  'Natural sounds',
                  'Other'
                  ]


endswith = '.pth'

cuda_seed = None
