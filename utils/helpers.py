import os


def check_related_path(current_path):
    assert os.path.exists(current_path)
    
    checkpoints_path = os.path.join(current_path, 'checkpoints')
    logs_path = os.path.join(checkpoints_path, 'logs')
    weights_path = os.path.join(current_path, 'weights')
    prediction_path = os.path.join(current_path, 'predictions')
    
    if not os.path.exists(checkpoints_path):
        os.mkdir(checkpoints_path)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    if not os.path.exists(prediction_path):
        os.mkdir(prediction_path)
        
    paths = {'checkpoints_path': checkpoints_path,
             'logs_path': logs_path,
             'weights_path': weights_path,
             'prediction_path': prediction_path}
    return paths



















