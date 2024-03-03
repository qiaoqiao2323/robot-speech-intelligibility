import sys, os, argparse

# 这里的0是GPU id
import numpy as np

gpu_id = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.data_generator import *
from framework.processing import *
from framework.models_pytorch import *


def cal_auc(targets_event, outputs_event):
    aucs = []
    for i in range(targets_event.shape[0]):
        test_y_auc, pred_auc = targets_event[i, :], outputs_event[i, :]
        if np.sum(test_y_auc):
            test_auc = metrics.roc_auc_score(test_y_auc, pred_auc)
            aucs.append(test_auc)
    final_auc_event_branch = sum(aucs) / len(aucs)
    return final_auc_event_branch


def main(argv):
    Model = DNN

    event_class = len(config.event_labels)
    model = Model(event_class)

    batch_size = 64

    result_dir = os.path.join(os.getcwd(), 'pretrained_models')
    file = 'DNN.pth'

    print('\nusing model: ', file)

    event_model_path = os.path.join(result_dir, file)
    model_event = torch.load(event_model_path, map_location='cpu')
    model.load_state_dict(model_event['state_dict'])

    cuda = 1
    if config.cuda:
        model.cuda()

    # Generate function
    data_type = 'test'
    dataset_path = os.path.join(os.getcwd(), 'Dataset')
    generator = DataGenerator(dataset_path=dataset_path, normalization=True)
    generate_func = generator.generate_testing(data_type=data_type)
    dict = forward_asc_aec(model=model, generate_func=generate_func, cuda=cuda)

    # -----------------------------------------------------------------------------------------------------------
    targets = dict['output']
    predictions = dict['target']

    mae_loss = metrics.mean_absolute_error(targets, predictions)
    mse_loss = metrics.mean_squared_error(targets, predictions)

    print("MSE : ", mse_loss)
    print("MAE : ", mae_loss)







if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















