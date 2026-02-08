build-stubs:
    uv run scripts/stubgen.py 'ome:formats-gpl:RELEASE' --prefix loci --prefix ome --prefix java

test:
    uv run pytest --allow-cache

check:
    uv run prek -a --hook-stage=manual
