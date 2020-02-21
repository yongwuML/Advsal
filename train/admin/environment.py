import importlib
import os

def env_settings():
    env_module_name = 'train.admin.local'
    try:
        env_module = importlib.import_module(env_module_name)
        return env_module.EnvironmentSettings()
    except:
        env_file = os.path.join(os.path.dirname(__file__), 'local.py')
        
        raise RuntimeError('YOU HAVE NOT SETUP YOUR local.py!!!\n Go to "{}" and set all the paths you need. Then try to run again.'.format(env_file))
