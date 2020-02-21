from test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.dutomron_path = '/home/yongwu/文档/YWuwork/R3Net/DUT-OMRON'
    settings.dutste_path = '/home/yongwu/文档/YWuwork/R3Net/DUTS-TE'
    settings.ecssd_path = '/home/yongwu/文档/YWuwork/R3Net/ECSSD'
    settings.hkuis_path = '/home/yongwu/文档/YWuwork/R3Net/HKU-IS'
    settings.network_path = '/home/yongwu/文档/YWuwork/Advsal/test/networks/'    # Where tracking networks are stored.
    settings.pascal_path = '/home/yongwu/文档/YWuwork/R3Net/PASCAL850'
    settings.results_path = '/home/yongwu/文档/YWuwork/Advsal/test/results/'    # Where to store tracking results

    return settings

