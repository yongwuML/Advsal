import importlib
import os

class EnvSettings:
    def __init__(self):
        testing_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        self.results_path = '{}/results/'.format(testing_path)
        self.network_path = '{}/networks/'.format(testing_path)
        self.pascals_path = '/home/yongwu/文档/YWuwork/R3Net/PASCAL850'
        self.ecssd_path = '/home/yongwu/文档/YWuwork/R3Net/ECSSD'
        self.hkuis_path = '/home/yongwu/文档/YWuwork/R3Net/HKU-IS'
        self.dutste_path = '/home/yongwu/文档/YWuwork/R3Net/DUTS-TE'
        self.dutomron_path = '/home/yongwu/文档/YWuwork/R3Net/DUT-OMRON'
        
        
        


def create_default_local_file():
    comment = {'results_path': 'Where to store tracking results',
               'network_path': 'Where tracking networks are stored.'}

    path = os.path.join(os.path.dirname(__file__), 'local.py')
    with open(path, 'w') as f:
        settings = EnvSettings()

        f.write('from test.evaluation.environment import EnvSettings\n\n')
        f.write('def local_env_settings():\n')
        f.write('    settings = EnvSettings()\n\n')
        f.write('    # Set your local paths here.\n\n')

        for attr in dir(settings):
            comment_str = None
            if attr in comment:
                comment_str = comment[attr]
            attr_val = getattr(settings, attr)
            if not attr.startswith('__') and not callable(attr_val):
                if comment_str is None:
                    f.write('    settings.{} = \'{}\'\n'.format(attr, attr_val))
                else:
                    f.write('    settings.{} = \'{}\'    # {}\n'.format(attr, attr_val, comment_str))
        f.write('\n    return settings\n\n')

def env_settings():
    env_module_name = 'test.evaluation.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.local_env_settings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')
        print(env_file)
        create_default_local_file

        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. '
                           'Then try to run again.'.format(env_file))
