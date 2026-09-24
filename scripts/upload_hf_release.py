"""Verify a staged bundle; upload to a private HF model repo only on --upload."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from futureworlds.checkpoints import verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    root = Path(args.folder)
    manifest = verify_bundle(root)
    allowed = {x["path"] for x in manifest["files"]} | {"manifest.json"}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != allowed:
        raise ValueError(
            "Staging directory contains files outside the verified manifest"
        )
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "files": len(allowed),
                "bytes": sum(x["bytes"] for x in manifest["files"]),
                "upload": args.upload,
            },
            indent=2,
        )
    )
    if not args.upload:
        return
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    api = HfApi()  # Existing local login; never embed a token in a config or command.
    try:
        info = api.repo_info(args.repo_id, repo_type="model")
        if not info.private:
            raise ValueError("This staging uploader only accepts private repositories")
    except RepositoryNotFoundError:
        api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=False)
    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=root,
        allow_patterns=sorted(allowed),
        commit_message="Add verified FutureWorlds research bundle",
    )
    print(result)


if __name__ == "__main__":
    main()
