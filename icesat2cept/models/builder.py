from icesat2cept.utils.registroy import Registry

MODELS = Registry("models")

def build_model(config):
    return MODELS.build(config)