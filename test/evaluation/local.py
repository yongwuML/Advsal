from test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.dutomron_path = ''
    settings.dutste_path = ''
    settings.ecssd_path = ''
    settings.hkuis_path = ''
    settings.network_path = ''    # Where tracking networks are stored.
    settings.pascal_path = ''
    settings.results_path = ''    # Where to store tracking results

    return settings

