import os

os.environ.setdefault("USE_TF", "0")
from .cli import main

main()
