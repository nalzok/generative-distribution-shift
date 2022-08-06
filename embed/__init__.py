from dataclasses import dataclass, asdict
from urllib.parse import quote
from json import dumps


@dataclass
class EmbeddingConfig:
    model: str
    global_pool: bool
    mask_ratio: float

    def __str__(self):
        config = asdict(self)
        config_string = quote(dumps(config, sort_keys=True, separators=(',', ':')), safe='')
        return f'EMBEDDING_{config_string}'
