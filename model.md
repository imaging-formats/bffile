Bio-Formats Data Model (Final - All Statements Verified)

File → Series → Resolution Hierarchy

1. A file contains 1+ series (minimum 1 after initialization)
2. Each series contains 1+ resolution levels (minimum 1)
3. Different series CAN have different numbers of resolutions ✅ CONFIRMED

Dimension Model (Per Series/Resolution)

1. Every series/resolution is conceptually 5D: TCZYX
2. Dimensions can be 0 during parsing, but readers normalize to ≥1:

  - During metadata parsing, sizeX, sizeY, sizeZ, sizeC, sizeT CAN temporarily be 0
  - Format readers defensively check and reset to 1: if (ms.sizeZ == 0) ms.sizeZ = 1;
  - After setId() completes, dimensions should be ≥1 (by convention, not enforced)

1. ALL planes within a series/resolution share: ✅ CONFIRMED

  - Same X,Y dimensions (sizeX, sizeY)
  - Same pixel type/dtype (pixelType)
  - No raggedness - strictly rectangular

1. RGB/interleaved channels:

  - getRGBChannelCount() > 1 for interleaved data
  - sizeC = logical channel count
  - openBytes() returns multiple channels together for interleaved

Core Invariant (with Edge Case)

1. effectiveSizeC × sizeZ × sizeT = imageCount - DOCUMENTED as definition, but:

  - When sizeZ * sizeT == 0, getEffectiveSizeC() returns 0 (not calculated)
  - After proper initialization, the relationship holds
  - This is the intended model, but not strictly enforced during parsing

CoreMetadata Organization

1. When flattenedResolutions = true (DEFAULT):

  - Each entry in getCoreMetadataList() = one (series, resolution) pair
  - Direct indexing works: core[i] for series i
  - BioFile relies on this mode

1. When flattenedResolutions = false:

  - 2D structure: navigate with setSeries(s) + setResolution(r)

Resolution Levels Within a Series

1. Across resolution levels of the same series: ✅ CONFIRMED

  - SAME: sizeC, sizeT, sizeZ, pixelType, endianness
  - DIFFERENT: sizeX, sizeY (decreasing in pyramid)
  - Resolutions created by copying base CoreMetadata, then modifying only X,Y

---
Key Insights for BioFile API Design

Based on this verified model, here's what you can rely on:

✅ Safe Assumptions (Always True After Init)

- Direct indexing: core_meta_list[series] works (when flattenedResolutions=true)
- All planes in a series have same shape (no raggedness)
- Resolutions preserve C,T,Z dimensions (only X,Y change)
- Series can have different resolution counts

⚠️ Edge Cases to Handle

- During reader initialization: Dimensions might temporarily be 0 (readers fix this)
- Flattened vs non-flattened: BioFile assumes flattened mode (safe, it's the default)
- The imageCount invariant: Documented relationship, but edge case when sizeZ or sizeT is 0

💡 API Design Recommendation

For a clean, stateless Python API:

# ✅ Make series explicit everywhere

```python
def read_plane(self, t=0, c=0, z=0, series=0, ...) -> np.ndarray:
    self._java_reader.setSeries(series)
    meta = self._core_meta_list[series]  # Direct indexing safe!
    ...

def core_meta(self, series: int = 0) -> CoreMetadata:
    return self._core_meta_list[series]  # Direct lookup
```

# Property for convenience (single-series files)

```python
@property
def shape(self) -> OMEShape:
    return self.core_meta(0).shape
```

No hidden state, no bugs, clear semantics!
