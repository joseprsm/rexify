import pickle
from pathlib import Path

from rexify.utils import make_dirs


class SavableTransformer:
    def save(self, output_dir: str, filename: str):
        make_dirs(output_dir)
        output_path = Path(output_dir) / filename
        with open(output_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str):
        with open(path, "rb") as f:
            feat = pickle.load(f)
        return feat
