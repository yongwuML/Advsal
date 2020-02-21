from test.utils import TrackerParams
from train.admin.environment import env_settings

def parameters():
    params = TrackerParams()

    params.description = 'SODGAN testing with default settings.'
    params.ckpt = env_settings().workspace_dir
    params.crf_refine = True
    params.save_results =True
    params.snapshot = '5000'
    params.num_workers = 12                                    # Number of workers for image loading
    params.normalize_mean = [0.485, 0.456, 0.406]             # Normalize mean (default pytorch ImageNet values)
    params.normalize_std = [0.229, 0.224, 0.225]   

    return params
