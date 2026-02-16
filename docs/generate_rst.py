import os
from pathlib import Path

SRC = Path("../include/saltro")
DST = Path("source/api")

for path in SRC.rglob("*"):
    if path.is_dir():
        rel = path.relative_to(SRC)
        out_dir = DST / rel
        out_dir.mkdir(parents=True, exist_ok=True)

        index = out_dir / "index.rst"

        subheaders = sorted(p.stem for p in path.glob("*.h"))
        subdirs = sorted(p.name for p in path.iterdir() if p.is_dir())

        with open(index, "w") as f:
            title = rel.name if rel != Path(".") else "API"
            f.write(title + "\n")
            f.write("=" * len(title) + "\n\n")

            if subdirs or subheaders:
                f.write(".. toctree::\n")
                f.write("   :maxdepth: 1\n\n")

                for d in subdirs:
                    f.write(f"   {d}/index\n")
                for h in subheaders:
                    f.write(f"   {h}\n")

    elif path.suffix == ".h":
        rel = path.relative_to(SRC)
        out_file = DST / rel.with_suffix(".rst")
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w") as f:
            title = path.stem
            f.write(title + "\n")
            f.write("=" * len(title) + "\n\n")
            f.write(f".. doxygenfile:: saltro/{rel}\n")
            f.write("   :project: saltro\n")