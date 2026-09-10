"""Build an explicit manuscript/evidence ZIP, excluding runtimes and raw assets."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    dest = OUT / "ConPath_paper_working_bundle_v1.zip"
    manifest_path = OUT / "release_manifest.json"
    if dest.exists() or manifest_path.exists():
        raise FileExistsError("Refuse to overwrite an existing paper release")
    mapping = {}

    def add(path, archive_name=None):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        assert path.is_file(), path
        if path.is_symlink():
            # Portable ZIPs materialize only these two local vendor links.
            assert path in {ROOT / "paper/IEEEtran.cls", ROOT / "paper/IEEEtran.bst"}, path
            assert path.resolve() == ROOT / "paper/vendor" / path.name, path
        rel = str(path.relative_to(ROOT))
        assert path.suffix.lower() not in {".npz", ".npy", ".pt", ".pth", ".tar", ".zip"}, path
        name = archive_name or rel
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        assert name not in mapping
        mapping[name] = path

    for name in ["PAPER_DRAFT.md", "PAPER_EVIDENCE.md", "PAPER_TABLES.md", "PAPER_FIGURES.md",
                 "PAPER_READING_ZH.md", "PAPER_DATA_PACKAGE.md", "PAPER_CLAIM_AUDIT.md",
                 "results/FINAL_MODEL_SELECTION.md", "results/paper_validation_snapshot.json",
                 "results/paper_validation_snapshot.csv"]:
        add(name)
    add(OUT / "README_ZH.md", "START_HERE_ZH.md")
    add(Path(__file__))
    add("results/paper_submission_v1/verify_submission.py")
    for name in ("main", "supplement"):
        pdf = ROOT / "paper/build" / (name + ".pdf")
        assert pdf.read_bytes().startswith(b"%PDF-"), pdf
    paper = ROOT / "paper"
    for p in sorted(paper.rglob("*")):
        if not p.is_file():
            continue
        parts = p.relative_to(paper).parts
        if any(v in parts for v in (".cache", "tools", "__pycache__")):
            continue
        if parts[0] == "build" and p.suffix not in {".pdf", ".json", ".log"}:
            continue
        if parts[0] == "review" and p.suffix != ".json" and not (p.suffix == ".jpg" and "_final_contact_" in p.name):
            continue
        if p.suffix.lower() in {".tex", ".bib", ".bst", ".cls", ".md", ".py", ".sh", ".json", ".pdf", ".txt", ".png", ".jpg", ".log"} or p.name in {".gitignore", "IEEEtran_README"}:
            add(p)
    for folder in ("evidence", "analysis", "review"):
        for p in sorted((ROOT / "results/paper_submission_v1" / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                add(p)
    figures = subprocess.check_output(["git", "ls-files", "results/paper_figures"], cwd=ROOT, text=True).splitlines()
    for name in figures:
        add(name)
    add("results/paper_submission_v1/portability_verification.json")
    add("results/paper_submission_v1/submission_verification.json")
    add("results/paper_submission_v1/root_visual_check.json")
    items = []
    with zipfile.ZipFile(dest, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(mapping.items()):
            data = path.read_bytes()
            info = zipfile.ZipInfo("ConPath_paper_working_v1/" + name, date_time=(2026, 9, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
            items.append(dict(archive_path=info.filename, source_path=str(path.relative_to(ROOT)),
                              materialized_local_vendor_symlink=path.is_symlink(),
                              sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    with zipfile.ZipFile(dest) as archive:
        assert archive.testzip() is None
        assert len(archive.infolist()) == len(items)
        for item in items:
            assert hashlib.sha256(archive.read(item["archive_path"])).hexdigest() == item["sha256"]
    manifest = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                    git_base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    archive=dest.name, sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
                    compressed_bytes=dest.stat().st_size, uncompressed_bytes=sum(x["bytes"] for x in items),
                    files=items, all_member_bytes_verified=True,
                    purpose="Author-review manuscript PDFs and portable existing scalar evidence",
                    final_test_results=False, new_training=False, new_inference=False,
                    exclusions=["TeX runtimes/cache", "weights", "raw maps", "final-test assets"])
    with manifest_path.open("x") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({k: manifest[k] for k in ("archive", "sha256", "compressed_bytes", "uncompressed_bytes")}, ensure_ascii=False))
    print("Files:", len(items))


if __name__ == "__main__":
    main()
