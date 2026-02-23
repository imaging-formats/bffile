# Contributing to bffile

Some quick notes/tips for contributors:

## Setup

1. Fork the repo and clone your fork locally.
1. Install uv: <https://docs.astral.sh/uv/getting-started/installation/>
1. `uv sync`
1. All work that you intend to submit as a PR should be done on a new
   branch (not `main`)... even for your personal fork.

## Testing

```sh
uv run pytest
```

The first run will download test data from a Cloudflare R2 bucket. 
To re-fetch the latest test data at any point, either delete `tests/data`, or use
[the fetch test data script](https://github.com/psobolewskiPhD/bffile/blob/main/scripts/fetch_test_data.py):

```sh
uv run python scripts/fetch_test_data.py
```

Note: To use an isolated [jgo](https://github.com/apposed/jgo) cache directory,
pass `--no-jgo-cache` to the test command. This will force jgo to resolve dependencies
from scratch each time, without using any cached artifacts from `~/.jgo`.
This is slower (since it needs to download all maven endpoints
each time), but ensures that tests are not affected by any existing Java
artifacts on your system. See [understanding the Java
setup](#understanding-the-java-setup-and-dependency-management) below for more
details.

```sh
uv run pytest --no-jgo-cache
```

## Understanding the Java setup and dependency management

A goal of bffile is for the entire java setup and dependency management to be
fully automatic and transparent to users. It should "just work" on any system
with Python and pip, without requiring users to have Java or Bio-Formats already
installed, or to manually configure classpaths or environment variables.

There is some very useful (but deep) java magic going on under the hood here,
which is worth understanding:

When someone `pip installs bffile` onto a clean system that perhaps doesn't even
have Java installed, and then imports `bffile` and creates a `BioFile` object,
the following happens:

1. **JVM Startup** (`bffile._java_stuff.start_jvm()`): Triggers
   `scyjava.start_jvm()`, then redirects Java logging to Python.

2. **Java Detection**: Inside of the `scyjava.start_jvm` function,
   `ensure_jvm_available()` checks if Java is available via
   `jpype.getDefaultJVMPath()`. If not found (or if forced), it will use
   [`cjdk`](https://github.com/cachedjdk/cjdk) to download and cache a JDK/JRE
   (default: zulu-jre) to `~/.cache/cjdk`, then adds it to PATH. Similarly
   downloads Maven if needed.  These are all configurable via scyjava:
   <https://github.com/scijava/scyjava?tab=readme-ov-file#bootstrap-a-java-installation>

3. **Dependency Resolution**: Next, `jgo.resolve_dependencies()` takes Maven
   coordinates (e.g., `ome:formats-gpl:RELEASE`) that have been specified using
   `scyjava.config.endpoints`, generates a temporary POM file, runs `mvn
   dependency:resolve` to download all JARs to `~/.m2/repository`, then links
   them into a jgo cache directory (`~/.jgo`) based on a hash of the
   coordinates.

4. **JVM Launch**: JPype starts the JVM with all resolved JARs on the classpath,
   making Bio-Formats classes available to Python.

This happens automatically on first import, with subsequent imports reusing
cached JDK and JARs for fast startup.

## Using Claude

If (like me) you don't have a *ton* of familiarity with the Bio-Formats java codebase,
it can be helpful to use an LLM to navigate the code. The `CLAUDE.md` file explains
to Claude how to clone the bioformats repo into a local dir and where to find key files.
