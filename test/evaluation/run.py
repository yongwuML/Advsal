import importlib
import numpy as np
import torch
import os
import time
from PIL import Image
from torchvision import transforms
from test.evaluation.environment import env_settings
from train.dataset import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from train.model.sodgan import Net


class Running:

    def __init__(self, name: str, dataset_name: str):
        self.name = name
        self.env = env_settings()
        if dataset_name == 'ecssd':
            self.to_test = {'ecssd': self.env.ecssd_path}
        elif dataset_name == 'hkuis':
            self.to_test =  {'hkuis': self.env.hkuis_path}
        elif dataset_name == 'pascal':
            self.to_test = {'pascal': self.env.pascals_path}
        elif dataset_name == 'dutomron':
            self.to_test = {'dutomron': self.env.dutomron_path}
        elif dataset_name == 'dutste':
            self.to_test = {'dutste': self.env.dutste_path} 
        else:
            raise ValueError('Unknown dataset name')

        self.dataset_name = dataset_name

        
    def infer(self):
        start_time=time.time()
        param_module = importlib.import_module('test.parameter.{}'.format(self.name))
        params = param_module.parameters()
        img_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=params.normalize_mean, std=params.normalize_std)])
        to_pil = transforms.ToPILImage()
        net = Net().cuda()

        print ('load snapshot \'%s\' for testing' % params.snapshot)
        net.load_state_dict(torch.load(os.path.join(self.env.network_path, params.snapshot + '.pth')))
        net.eval()

        results = {}

        with torch.no_grad():

            for name, root in self.to_test.items():

                precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
                mae_record = AvgMeter()

                if params.save_results:
                    check_mkdir(os.path.join(self.env.results_path, '%s_%s' % (name, params.snapshot)))

                img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.jpg')]
                for idx, img_name in enumerate(img_list):
                    print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))

                    img = Image.open(os.path.join(root, img_name + '.jpg')).convert('RGB')
                    img_var = img_transform(img).unsqueeze(0).cuda()
                    prediction = net(img_var)
                    prediction = np.array(to_pil(prediction.data.squeeze(0).cpu()))

                    if params.crf_refine:
                        prediction = crf_refine(np.array(img), prediction)

                    gt = np.array(Image.open(os.path.join(root, img_name + '.png')).convert('L'))
                    precision, recall, mae = cal_precision_recall_mae(prediction, gt)
                    #precision, recall, mae = cal_precision_recall_mae(prediction, gt, img_name)
                    for pidx, pdata in enumerate(zip(precision, recall)):
                        p, r = pdata
                        precision_record[pidx].update(p)
                        recall_record[pidx].update(r)
                    mae_record.update(mae)

                    if params.save_results:
                        Image.fromarray(prediction).save(os.path.join(self.env.results_path, '%s_%s' % (
                                name, params.snapshot), img_name + '.png'))
                    end_time = time.time()
                T = '%d' % (end_time-start_time)
                print(T)

                fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                    [rrecord.avg for rrecord in recall_record])

                results[name] = {'F-measure': fmeasure, 'MAE': mae_record.avg}

        print ('test results:')
        print (results)
