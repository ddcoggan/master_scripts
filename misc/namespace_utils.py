
import yaml as yaml
from types import SimpleNamespace
from json import JSONEncoder


class Loader(yaml.Loader):
    pass


class Dumper(yaml.Dumper):
    pass


def _construct_mapping(loader, node):
    loader.flatten_mapping(node)
    return SimpleNamespace(**dict(loader.construct_pairs(node)))


def _ns_representer(dumper, data):
    return dumper.represent_mapping(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.__dict__.items()
    )

Dumper.add_representer(SimpleNamespace, _ns_representer)

Loader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


class nsEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, SimpleNamespace):
            return obj.__dict__
        return super(nsEncoder, self).default(obj)


def object_hook(d):
    return SimpleNamespace(**d)

