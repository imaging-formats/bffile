# Tips for Claude and LLM Agents

This repo wraps the Bio-Formats Java library.

## Understanding the underlying Java codebase

When you need to understand the underlying Java implementation details,
you can clone it locally if it's not already here:

```bash
git clone https://github.com/ome/bioformats
cd bioformats
```

### Directory Structure

The bioformats repo uses a multi-module structure under `components/`:

```text
bioformats/components/
├── formats-api/          # Core API interfaces and base classes
├── formats-bsd/          # BSD-licensed implementations
├── formats-gpl/          # GPL-licensed implementations
├── bio-formats-tools/    # Command-line tools
├── bio-formats-plugins/  # ImageJ/Fiji plugins
└── test-suite/           # Test files
```

### Key Java Classes We Wrap

Here's exactly where each Java class used in `src/bffile/_biofile.py` is located:

#### 1. `loci.formats.ImageReader`

- **Location**: `bioformats/components/formats-api/src/loci/formats/ImageReader.java`
- **Purpose**: Master file format reader that auto-detects and delegates to specific format readers

#### 2. `loci.formats.IFormatReader`

- **Location**: `bioformats/components/formats-api/src/loci/formats/IFormatReader.java`
- **Purpose**: Interface defining the reader contract
- **Note**: This is the interface that `ImageReader` implements

#### 3. `loci.formats.Memoizer`

- **Location**: `bioformats/components/formats-bsd/src/loci/formats/Memoizer.java`
- **Purpose**: Caching wrapper that serializes initialized readers to `.bfmemo` files
- **How it works**: Wraps another reader and caches the initialized state after `setId()` to speed up subsequent opens
- **Constructor we use**: `Memoizer(IFormatReader, long millisThreshold, Path memoDir)`

#### 4. `loci.formats.in.DynamicMetadataOptions`

- **Location**: `bioformats/components/formats-api/src/loci/formats/in/DynamicMetadataOptions.java`
- **Purpose**: Configuration object for reader behavior
- **Key method we use**: `set(String name, String value)` - Sets format-specific options
- **Example usage**: `options={"nativend2.chunkmap": False}` to disable ND2 chunkmap reading

#### 5. `loci.formats.services.OMEXMLService`

- **Location**: `bioformats/components/formats-api/src/loci/formats/services/OMEXMLService.java`
- **Purpose**: Service interface for creating and manipulating OME-XML metadata
- **Key method we use**: `createOMEXMLMetadata()` - Creates metadata store for readers

#### 6. `loci.formats.FormatTools`

- **Location**: `bioformats/components/formats-api/src/loci/formats/FormatTools.java`
- **Purpose**: Utility class with static helper methods
- **What we use**: `VERSION` constant to get Bio-Formats version

#### 7. `loci.common.services.ServiceFactory`

- **Location**: ⚠️ **NOT in bioformats repo** - from separate `loci-common` dependency
- **Purpose**: Factory pattern for creating service instances
- **Documentation**: <https://downloads.openmicroscopy.org/bio-formats/latest/api/>

### Underlying Implementation Details

#### CoreMetadata Structure

- **Location**: `bioformats/components/formats-api/src/loci/formats/CoreMetadata.java`
- Contains: `sizeX`, `sizeY`, `sizeZ`, `sizeC`, `sizeT`, pixel type, RGB channel count, dimension order, etc.
- We parse this in our `_core_metadata.py` module

#### Reader Workflow

1. **ImageReader** acts as a multiplexer - it maintains instances of all format-specific readers
2. When `setId()` is called, it tests each reader until one accepts the file
3. That specific reader becomes the active delegate for all subsequent operations
4. **Memoizer** wraps this entire setup and caches the initialized state

#### Key Java Package Structure

```text
loci.formats/              # Main reader API
├── ImageReader            # Master reader
├── IFormatReader          # Reader interface
├── FormatReader           # Base implementation
├── FormatTools            # Utilities
├── CoreMetadata           # Metadata container
├── Memoizer               # Caching (in formats-bsd)
├── in/
│   └── DynamicMetadataOptions  # Reader options
└── services/
    └── OMEXMLService      # OME metadata service

loci.common/               # Separate dependency
└── services/
    └── ServiceFactory     # Service factory pattern
```

### Quick Reference for Finding Java Code

When you need to understand the Java implementation:

1. **Reader behavior**: Check `formats-api/src/loci/formats/FormatReader.java` for base implementation
2. **Specific format readers**: Look in `formats-bsd/src/loci/formats/in/` (e.g., `ND2Reader.java`)
3. **Metadata handling**: See `formats-api/src/loci/formats/CoreMetadata.java`
4. **Caching logic**: Review `formats-bsd/src/loci/formats/Memoizer.java`
5. **Options system**: Check `formats-api/src/loci/formats/in/DynamicMetadataOptions.java`
