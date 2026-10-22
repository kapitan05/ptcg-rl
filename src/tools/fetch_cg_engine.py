"""Fetch the official C++ engine, verify its bytes, and install without patches."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DISTRIBUTION = {
    "competition": "pokemon-tcg-ai-battle",
    "source_prefix": "ptcg_engine/ptcgProgram 22/",
    "files": {
        "ActivateInfo.h": {
            "size": 1714,
            "sha256": "a4c327b6c8fc6be3a80e67eae90f07e13ab799af65b3f6bb4456b387aeacedf8",
        },
        "AddLog.h": {
            "size": 6906,
            "sha256": "70c4b980fd7b64fd53e4add3269de076546b8f3284304d6cca84b6b070aef869",
        },
        "AddOption.h": {
            "size": 4268,
            "sha256": "dfe24ced984da07aaf50cab7835744064c19de87e24c14a943fb01584c15117f",
        },
        "All.h": {
            "size": 748,
            "sha256": "b176e6c5d1d3dde03e103f67b26117cab8e89b607bcfea26e01147be6e9dd41f",
        },
        "Api.h": {
            "size": 8427,
            "sha256": "786acae884631bdcbab0311471316bc75d62acb465b8011bf09dad05628bcafd",
        },
        "ApiData.h": {
            "size": 933,
            "sha256": "b21025cb1027556746f11fd73d6851ef231fa6492d82dd8074e80d70cbf93da9",
        },
        "ApiJson.h": {
            "size": 13028,
            "sha256": "fa405a0421ad09fbae6e3922bd73a0a07333b17790dea7e8fd90c1b97f326851",
        },
        "ApiType.h": {
            "size": 3130,
            "sha256": "d270162140ce1e518a9f2ea6a4a73ed10e9585b8a212eb9a3eca0ba1c220d655",
        },
        "BattleData.h": {
            "size": 8844,
            "sha256": "6aa12c3cdec1b3e3aca0b4c102952ab5713a4ba10c1a0a8df66be3b42db5db8f",
        },
        "Binary.h": {
            "size": 4934,
            "sha256": "ab42f329818f19e4a03d76edf8066312fc0060bbc1b5524a17897a1636b12257",
        },
        "Card.h": {
            "size": 11962,
            "sha256": "32a3b5fc5b1cb2eb7826b97faade4ec36ed5aa83082bab32e6ad8fbb394c844d",
        },
        "CardImpl.h": {
            "size": 878383,
            "sha256": "286a51820d36f9b60b5b13c58bc2edff352eb4050581ddadf1960f13fd6f21a9",
        },
        "CardMove.h": {
            "size": 16529,
            "sha256": "7252d27c63c539c30899e3c7136e34506e9b3be3f09e974cfc4b2f3139e26a0d",
        },
        "Core.h": {
            "size": 1524,
            "sha256": "fdc29455564073b865cc8ac0e0429e66a1df4a245594b26f939d64bc1d0fd7a3",
        },
        "CreateCard.h": {
            "size": 57440,
            "sha256": "54bf6a7fc374f740454116b5a291460192287b8d1ba8791cf223ecda925f78d0",
        },
        "EffectContinual.h": {
            "size": 16472,
            "sha256": "5ac571b4ed6e5b324098cd9ca0377f53eba4788172ab8d9c891648d4c5b09c83",
        },
        "EffectInstant.h": {
            "size": 65977,
            "sha256": "e1729b7f5ad4f928fd7b0cb6053be3b935065711483f7018797ba822679e46f8",
        },
        "EffectProc.h": {
            "size": 36883,
            "sha256": "ab5aa982826910d372b5983308f995675ff2c9b40823b444fbff6f1a26f0f975",
        },
        "Export.cpp": {
            "size": 5117,
            "sha256": "1269f64671527d50b8560540473438e9781926fdbdb0c9383b344f0fb91bf82f",
        },
        "FixedList.h": {
            "size": 4907,
            "sha256": "2e1de1418edf279040ba7dc9e8ec9fb1608880f6a66d742b5036b74f1f31836f",
        },
        "Framework.h": {
            "size": 903,
            "sha256": "f0ad603d615ba1769c622982e791d042a900d4dfa0b2d53f6ce48b84eced19a2",
        },
        "Game.h": {
            "size": 2857,
            "sha256": "1dd43b6a46f225ff2a238d92e8d718cf0d7414773bfe52388f5325bf02b0a097",
        },
        "GameFunction.h": {
            "size": 1536,
            "sha256": "b88525cdd5a760d7c8871d99897453a2433fdd024f76a67ea8568806f4cfcf57",
        },
        "GameProc.h": {
            "size": 30790,
            "sha256": "cf1537be5c439bed44fbf08b91a6b066f3878ee5ae8a88a828f110bce01b3260",
        },
        "GameUtil.h": {
            "size": 6201,
            "sha256": "ae42efb5ec81b9292eeeb51742de9661af8a6cc0a7e0e0999ff0dea347d55aab",
        },
        "InitializeCard.h": {
            "size": 1211,
            "sha256": "130b20dbedeb8de09e4cab1b3cba0b12fe1d0654788b134e29d2245dd0f36da8",
        },
        "JsonBuilder.h": {
            "size": 3500,
            "sha256": "4a6c1604e4fce3669a7734c0c80a41227448af9ad7f9c5834757d7d1fb143fe7",
        },
        "LICENSES/LicenseRef-PTCG-ABC-Competition-Use-Only.txt": {
            "size": 1590,
            "sha256": "0bc3060e440432fa9481071ee9531d0d34e7212b1b6ffe595e38b9c9cd9c081b",
        },
        "PlayerState.h": {
            "size": 3713,
            "sha256": "269e7c4a2c1522245fc2ddc1e89cf83412d1b00f1deed4b673f3275906a412a1",
        },
        "PullTrigger.h": {
            "size": 5310,
            "sha256": "620076e485690bdab1f82327bc862374803794b54c5cccce0339a5abc0f8db99",
        },
        "README.md": {
            "size": 2293,
            "sha256": "7e49513e5704a95f13772d87e0e955eba20903cd7184367928e3a5f08540c91d",
        },
        "REUSE.toml": {
            "size": 464,
            "sha256": "9bcfaf05c5d0a6abfe923bcd918c8fd9524c297e44592085db70a67639f9ae5b",
        },
        "SatisfyCondition.h": {
            "size": 9988,
            "sha256": "5a26309ff23e616b6a27c66e5a9125d6062763cd5dbf0e903f589a3f85a59127",
        },
        "Search.h": {
            "size": 6321,
            "sha256": "d73553ee85dd993f2c1d7e4606e3e38f988f5390b88080148f4ef98a926762f9",
        },
        "SelectProc.h": {
            "size": 14926,
            "sha256": "cefae973a571d00a7c34a94f613e27c5048080c371ee62824dca390519099c46",
        },
        "SetProperty.h": {
            "size": 14367,
            "sha256": "a563762c4a8f6c1f9c77048b89f770cd2e7a5d4801cd29256196dda9e40fcb81",
        },
        "SetupProc.h": {
            "size": 8266,
            "sha256": "fa1e3290d45796fa802b50f3005c657202653173183ca0abb024656b0ed7e1a9",
        },
        "Skill.h": {
            "size": 8400,
            "sha256": "53415b8143e0b23dbfe6b79997924effd90afa6104022334c6e8aeaa3c23991e",
        },
        "State.h": {
            "size": 49492,
            "sha256": "a12bc5669b4b79c122899142f86616f8d9926684d9029392220c5c886de2a33c",
        },
        "TargetList.h": {
            "size": 26064,
            "sha256": "c93bea62ea3de7b733e3cbbc8d75cde9a612c3dd956516597a8fe8858dc8eac7",
        },
        "ToJson.h": {
            "size": 9993,
            "sha256": "84ee63939863493520ebe29e8ca717217ebf90191829ea80d1349942aa867602",
        },
        "Types.h": {
            "size": 36076,
            "sha256": "c5a9ad65b1221fc9b6ed09e5f56c6bd7ec325117cc2247592164abe698bad0a8",
        },
        "game.sln": {
            "size": 945,
            "sha256": "05a0d8ad94f1aef75733d18dc354534aa974437e405825eef5bfa9cc2dfec3cc",
        },
        "game.vcxproj": {
            "size": 6161,
            "sha256": "afb079173a8305e0441b95cef7a304ddd799a88fabb656a05ed541764841e6eb",
        },
        "game.vcxproj.filters": {
            "size": 5100,
            "sha256": "f32e46fa571c00c29051f90ba08f16f06812a24d40c889dd4ac0a6747eac22d7",
        },
    },
}

PLAYGROUND = {
    "competition": "the-pokemon-company-ptcg-ai-battle-challenge-playground",
    "source_prefix": "ptcg_engine_playground/ptcgProgram 22/",
    "files": {
        "ActivateInfo.h": {
            "size": 1769,
            "sha256": "9f4449af9dad01f6e244cd3b26c750f4be154580cbaa8760e970c51c53c241e9",
        },
        "AddLog.h": {
            "size": 6940,
            "sha256": "4b64a7bd38b7c2570a15a5b9db8e5ddc77c6e15bbd6f89154ba4840ea0e400ca",
        },
        "AddOption.h": {
            "size": 4302,
            "sha256": "ad99ee765f505ab4f0569decc2b0ab7c2b6cf74da60cee5ddf11e4b2d98cf778",
        },
        "All.h": {
            "size": 782,
            "sha256": "62a18bc4846a4616aab527a33a750a1ad1427df31e0cd2470ef07fa1badd329b",
        },
        "Api.h": {
            "size": 8408,
            "sha256": "faa89394b3b2740a4d519c8db04676b87d69d7aaf1801ac1265b46499ce94875",
        },
        "ApiData.h": {
            "size": 967,
            "sha256": "e9f517376d7378c98b6302f2b533ae8d39ead1ce8bd0925a3fb248a8daa9babe",
        },
        "ApiJson.h": {
            "size": 13062,
            "sha256": "aa9858c99f6258c6c4ad817d124c9ff182548ca2fb477202f467b3a18163b406",
        },
        "ApiType.h": {
            "size": 3164,
            "sha256": "60987a5858779bebaa365e67c9e8b2e45f2360ead9b2543e85a3f669e5421090",
        },
        "BattleData.h": {
            "size": 9106,
            "sha256": "3fcc39786299e3ee83fcd8907027a43d63778c0037a0a1d2f1d9a9a6e6ecfcc4",
        },
        "Binary.h": {
            "size": 4968,
            "sha256": "f8cc840300628d6c0b7229929cef70697e7ed867865512c5dfe57a625063a35b",
        },
        "Card.h": {
            "size": 12788,
            "sha256": "c721ba102ed3ab0a1ea5860230bbbc84e7478c3d9fc21b664130a2bd89c611bf",
        },
        "CardImpl.h": {
            "size": 997516,
            "sha256": "35a0ad47ee90b542a6556e25c2cf4563253a41c0fbe75e06e81990a5fcbd9252",
        },
        "CardMove.h": {
            "size": 16659,
            "sha256": "b2d8ab6db9a73c9b3781faf824516c63e06653571d1b1fab620b525256f9bd9f",
        },
        "Core.h": {
            "size": 1558,
            "sha256": "a043860c651e93afae8101541359be9600e907fd3dd07cb53b27bac4164b3ae8",
        },
        "CreateCard.h": {
            "size": 58270,
            "sha256": "6a064d3354f00b729a2f65ba95306f36168f7c3f21626a1c15d958e38c896d84",
        },
        "EffectContinual.h": {
            "size": 16715,
            "sha256": "8b5ea32ff9b950095132cebf4d545c028bd9cdd3acce8f1286801ae925467933",
        },
        "EffectInstant.h": {
            "size": 68691,
            "sha256": "77800fac405a981ca927187e36f62302ecfdd012c42d8e202213e7f6226da775",
        },
        "EffectProc.h": {
            "size": 37296,
            "sha256": "e9704b4e3a7e289c29d103f4b3bd4d86b3319ebd66b30b16badae2548eb5b265",
        },
        "Export.cpp": {
            "size": 5254,
            "sha256": "01d66dae52ad6f5b2e8cee3550b43163019fcc9d5a7ab3e0720198affa02fdee",
        },
        "FixedList.h": {
            "size": 4941,
            "sha256": "98e1e49c03ff278d5c3dbb0140b367bb73c7adac45ed4e297f3485bbfc243a72",
        },
        "Framework.h": {
            "size": 937,
            "sha256": "406fce1a7b885a928137ebf44d86feba5c31d139084325ce2da199afe1a7be0f",
        },
        "Game.h": {
            "size": 2913,
            "sha256": "8ad1bff43e406f98065274591ce20425ed30d4de41590e9f55e33ae5d6e7595f",
        },
        "GameFunction.h": {
            "size": 1570,
            "sha256": "3b575d337567b8d8532086046fbd31cea294bf46c62bc56bd48b68c7398567b0",
        },
        "GameProc.h": {
            "size": 32615,
            "sha256": "8293c27f920dafd12c86531a4f5e984e8df934f58b3204996ab6adf75258af9f",
        },
        "GameUtil.h": {
            "size": 6880,
            "sha256": "79778edb943eb0058fe40ecc24a34f75a4b5d1515212d7b7cd73a36886b2be03",
        },
        "InitializeCard.h": {
            "size": 1245,
            "sha256": "e32bbe3f4aaf05164f02ec7b5bac8709f303f4ea23ffc7ffadcc6a22b6512a7b",
        },
        "JsonBuilder.h": {
            "size": 3534,
            "sha256": "fa022e28a9aea83738e20219955fd37e15a49ed6b53b41eec2eacbe4a4ca094c",
        },
        "LICENSES/LicenseRef-PTCG-ABC-Competition-Use-Only.txt": {
            "size": 1624,
            "sha256": "2a936d4c4ac8a1c80bf4b7733d2174d7730e8513428585d867e67b208cb03c48",
        },
        "PlayerState.h": {
            "size": 3792,
            "sha256": "d758d1c7a5a3f062edba09750ae8875ba3bc7b5effdbf506efeff86ee2d5f153",
        },
        "PullTrigger.h": {
            "size": 5612,
            "sha256": "2ebc6b5751c29cc9c465d503efe1cede604634bd783b4b4318e6fce6d7c96010",
        },
        "README.md": {
            "size": 2361,
            "sha256": "c0284008a1c7641516347bb1f9ad92618a349396abb875a8e4a8e1c31096a739",
        },
        "REUSE.toml": {
            "size": 464,
            "sha256": "9bcfaf05c5d0a6abfe923bcd918c8fd9524c297e44592085db70a67639f9ae5b",
        },
        "SatisfyCondition.h": {
            "size": 10022,
            "sha256": "b03d5f255314c268315ea334277f95f874f13d8f8561448bdef333c27a86a644",
        },
        "Search.h": {
            "size": 6355,
            "sha256": "f8dcacc58cf8782d9bcab53b1a95df6c27d38bc27506c55fbbd121683ddc0dd5",
        },
        "SelectProc.h": {
            "size": 16739,
            "sha256": "133b80b167362ecdb38af2affa538944330e6bcd4f2350d9a1ade0a139a52263",
        },
        "SetProperty.h": {
            "size": 15265,
            "sha256": "438f2869f0211bbbeaf2de9705e1107abe02312ecbe0c0f67497206187f70587",
        },
        "SetupProc.h": {
            "size": 8339,
            "sha256": "6daeae1550116adf929fdfefe78ea79c6ca344b61dbe9678114b88ccd727f13d",
        },
        "Skill.h": {
            "size": 8634,
            "sha256": "cf290d5ab92cd9ee73a2dbb20e2da3bc7bf152551134b78dedc0d75dd199257a",
        },
        "State.h": {
            "size": 49934,
            "sha256": "c4d559c4b84a1e98aa69f7c92ddace69f45c070d5adfee6aa91fa34ca916b54b",
        },
        "TargetList.h": {
            "size": 26344,
            "sha256": "d611b97361f8209bdc3c63be422cf01133e5c6a39746f4e3aa1dd569b69bc03d",
        },
        "ToJson.h": {
            "size": 10205,
            "sha256": "c01021d1aff19ffabcf67d7f174f884f1d6915f6930df70325b3923d80ad1dea",
        },
        "Types.h": {
            "size": 37084,
            "sha256": "46ccfd9a3544d496f0a457950636a144118ec3ef6e509ecc2bebd51c32b34a73",
        },
        "game.sln": {
            "size": 945,
            "sha256": "05a0d8ad94f1aef75733d18dc354534aa974437e405825eef5bfa9cc2dfec3cc",
        },
        "game.vcxproj": {
            "size": 6161,
            "sha256": "afb079173a8305e0441b95cef7a304ddd799a88fabb656a05ed541764841e6eb",
        },
        "game.vcxproj.filters": {
            "size": 5100,
            "sha256": "f32e46fa571c00c29051f90ba08f16f06812a24d40c889dd4ac0a6747eac22d7",
        },
    },
}
DISTRIBUTIONS = {"original": DISTRIBUTION, "playground": PLAYGROUND}


def safe_relative(name: str) -> PurePosixPath:
    """Reject absolute paths, parent traversal, and ambiguous archive separators."""
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe distribution path: {name}")
    return Path(*path.parts)


def verify_file(path: Path, spec: dict[str, Any]) -> None:
    """Verify a distribution file's size and SHA-256 digest."""
    data = path.read_bytes()
    if len(data) != spec["size"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
        raise ValueError(f"Official file changed or corrupted: {path}")


def verify_tree(directory: Path, manifest: dict[str, Any]) -> None:
    """Require exactly the original distribution files and reject symlinks."""
    expected = set(manifest["files"])
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink in engine: {path}")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if actual != expected:
        raise ValueError(
            f"Engine file list mismatch: missing={expected - actual}, extra={actual - expected}"
        )
    for name, spec in manifest["files"].items():
        verify_file(directory / safe_relative(name), spec)


def unpack_one(download_dir: Path, target: Path, relative: PurePosixPath, remote_name: str) -> None:
    """Kaggle may serve an individual file either directly or as a ZIP."""
    direct = download_dir / relative.name
    if direct.is_file():
        shutil.copyfile(direct, target)
        return
    archives = list(download_dir.glob("*.zip"))
    if len(archives) != 1:
        raise ValueError(f"Unexpected Kaggle download layout: {download_dir}")
    with zipfile.ZipFile(archives[0]) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if len(members) != 1:
            raise ValueError("Expected exactly one file in Kaggle download ZIP")
        member = members[0]
        safe_relative(member.filename)
        if member.filename not in (relative.name, relative.as_posix(), remote_name):
            raise ValueError(f"Unexpected ZIP entry: {member.filename}")
        # Read bytes only; never apply archive paths, permissions, or symlinks.
        target.write_bytes(archive.read(member))


def install(
    manifest: dict[str, Any], destination: Path, cache: Path, verify_only: bool = False
) -> None:
    """Download into staging and install the verified engine atomically."""
    for name in manifest["files"]:
        safe_relative(name)
    if destination.is_symlink():
        raise ValueError(f"Engine destination must not be a symlink: {destination}")
    if destination.exists():
        verify_tree(destination, manifest)
        print(f"Verified {len(manifest['files'])} unchanged engine files: {destination}")
        return
    if verify_only:
        raise FileNotFoundError(f"Engine has not been downloaded: {destination}")
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cg-staging-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        for index, (name, spec) in enumerate(manifest["files"].items(), 1):
            relative = safe_relative(name)
            cached = cache / "files" / relative
            if cached.exists():
                verify_file(cached, spec)
            else:
                cache.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=cache) as download:
                    remote = manifest["source_prefix"] + relative.as_posix()
                    api.competition_download_file(
                        manifest["competition"], remote, path=download, quiet=True
                    )
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    candidate = Path(download) / "verified-payload"
                    unpack_one(Path(download), candidate, relative, remote)
                    verify_file(candidate, spec)
                    candidate.replace(cached)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cached, target)
            print(f"{index}/{len(manifest['files'])} {name}", flush=True)
        verify_tree(staging, manifest)
        if destination.exists():
            raise FileExistsError(f"Destination appeared during download: {destination}")
        staging.rename(destination)
    print(f"Installed unmodified competition engine: {destination}")


def main() -> None:
    """Parse command-line arguments and run this tool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "src/engine/cg")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "external/engine_download")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--distribution", choices=sorted(DISTRIBUTIONS), default="playground")
    args = parser.parse_args()
    install(DISTRIBUTIONS[args.distribution], args.destination, args.cache_dir, args.verify_only)


if __name__ == "__main__":
    main()
