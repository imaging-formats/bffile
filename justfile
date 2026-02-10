@_default:
    @just --list

# build java stubs into ./typings
build-stubs:
    uv run scripts/stubgen.py 'ome:formats-gpl:RELEASE' --prefix loci --prefix ome --prefix java

# run tests quickly with coverage
test:
    uv run pytest --allow-cache -n 6 --cov

# run linting and type checking
check:
    uv run prek -a --hook-stage=manual

# clone bioformats repository
clone-bioformats:
    git clone https://github.com/ome/bioformats
