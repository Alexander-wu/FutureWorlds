"""Small tensor container for the upstream video reward's batch-only API."""


class TensorBatch:
    def __init__(self, batch, meta_info=None):
        self.batch = batch
        self.meta_info = meta_info or {}

    @classmethod
    def from_single_dict(cls, batch, meta_info=None):
        return cls(batch, meta_info)

    def __len__(self):
        return len(next(iter(self.batch.values())))

    def __getitem__(self, index):
        return type(self)({k: v[index] for k, v in self.batch.items()}, self.meta_info)
