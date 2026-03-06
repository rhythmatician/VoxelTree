# Voxy On-Disk & In-Memory Format

> **Source:** Audited from `MCRcortex/voxy` source code (v0.2.11-alpha), cloned to
> `reference-code/voxy/`. All claims below are grounded in specific Java files.

---

## 1. Section Geometry

| Concept | Value | Source |
|---|---|---|
| Voxels per section (in-memory) | `32 × 32 × 32 = 32,768` | `WorldSection.java` |
| In-memory index formula | `(y<<10)\|(z<<5)\|x` | `WorldSection.java:getIndex()` |
| Vanilla chunk section size | `16 × 16 × 16` | MC standard |
| LOD 0 WorldSection = | 32 world-blocks per axis @ 1:1 | derived |
| LOD 1 WorldSection = | 32 voxels covering 64 world-blocks per axis | derived |
| LOD _n_ WorldSection = | 32 voxels covering `32 × 2ⁿ` world-blocks per axis | derived |

Two vanilla chunk sections fit along each axis per LOD-0 WorldSection (2×2×2 = 8 sections total fill one WorldSection).

---

## 2. Section Key Encoding (64-bit)

From `WorldEngine.java`, method `getWorldSectionId(int lvl, int x, int y, int z)`:

```java
return ((long)lvl<<60)
     | ((long)(y&0xFF)<<52)
     | ((long)(z&((1<<24)-1))<<28)
     | ((long)(x&((1<<24)-1))<<4);
// NOTE: bits 3-0 are spare/unused
```

| Field | Bits | Width | Notes |
|---|---|---|---|
| `lvl` | 63–60 | 4 bits | LOD level 0–15 |
| `y` | 59–52 | 8 bits | Signed, -128 to +127 (section units) |
| `z` | 51–28 | 24 bits | Signed 24-bit (via arithmetic shift in getter) |
| `x` | 27–4 | 24 bits | Signed 24-bit (via arithmetic shift in getter) |
| spare | 3–0 | 4 bits | Unused |

Decoder helpers (also in `WorldEngine.java`):
```java
getLevel(id) = (id>>60)&0xF
getX(id) = (int)((id<<36)>>40)   // sign-extends 24-bit
getY(id) = (int)((id<<4)>>56)    // sign-extends 8-bit
getZ(id) = (int)((id<<12)>>40)   // sign-extends 24-bit
```

---

## 3. Per-Voxel `long` Encoding

From `Mapper.java` constants and `composeMappingId()`:

```
Bit 63–56  (8 bits) : light  — packed as (sky<<4 | block), each 4 bits
Bit 55–47  (9 bits) : biome ID  — Voxy's internal biome index (up to 512)
Bit 46–27  (20 bits): block state ID — Voxy's **internal** mapped ID (not MC registry ID)
Bit 26–0   (27 bits): unused / lower flags (zero in practice)
```

Special values:
- `AIR = 0` (the entire long is 0, i.e. block bits 27-46 == 0 means air)
- `Mapper.isAir(id)` checks whether block state bits are zero

**Critical:** The block state ID in bits 27–46 is **Voxy's own internal mapping**, NOT the
Minecraft registry block state ID. The mapping is session-unique and persisted per-world in
the storage backend's ID mapping table (string `BlockState.toString()` ↔ integer ID).

---

## 4. VoxelizedSection — Ingestion-Time Pyramid

`VoxelizedSection.java` is used *only during ingestion* (not the stored form). It holds the
full 5-level LOD pyramid derived from a single vanilla 16³ chunk section:

| Level | Size | Offset into flat array |
|---|---|---|
| 0 (full res) | 16³ = 4096 | 0 |
| 1 | 8³ = 512 | 4096 (`1<<12`) |
| 2 | 4³ = 64 | 4608 (`(1<<12)\|(1<<9)`) |
| 3 | 2³ = 8 | 4672 (`(1<<12)\|(1<<9)\|(1<<6)`) |
| 4 | 1³ = 1 | 4680 (`(1<<12)\|(1<<9)\|(1<<6)\|(1<<3)`) |
| **Total** | **4681 longs** | |

Built by `WorldConversionFactory.convert()` (fills level 0) then `mipSection()` (fills
levels 1–4). This structure is then passed to `WorldUpdater.insertUpdate()` which assembles
it into the persistent `WorldSection` (32³).

---

## 5. LOD Downsampling Algorithm (`Mipper.java`)

Voxy's mip algorithm for each 2×2×2 block of voxels is **opacity-biased selection**, NOT
majority vote or averaging:

```
For the 8 children (I000 … I111):
  1. Filter out air voxels.
  2. Among non-air, score = (blockOpacity << 4) | cornerPriority
     where cornerPriority: I111=7 > I110=6 > I011=5 > I010=4
                                    > I101=3 > I100=2 > I001=1 > I000=0
  3. Return the voxel with the highest score (max opacity, tie-break by corner).
  4. If ALL children are air:
       skyLight  = ceil(mean of 8 sky-lights)
       blockLight = floor(mean of 8 block-lights)
       return I111 with averaged light (still air).
```

**Implications for training data construction:**
- Do **not** use majority vote or probability pooling to match Voxy's LOD.
- Use per-block opacity from MC's registry + the I111 corner tie-break rule.
- The result is that opaque blocks (stone, dirt) always win over transparent ones (water, glass).

---

## 6. On-Disk Serialization Format

From `SaveLoadSystem.java`. Each section is serialized as:

| Field | Size | Notes |
|---|---|---|
| `key` | 8 bytes | Section's 64-bit position key (§2 above) |
| `metadata` | 8 bytes | Low byte = `nonEmptyChildren` bitmask (which of the 8 octants have data) |
| `lutLen` | 4 bytes | Number of unique `long` values in this section (≤ 32,768) |
| LUT entries | `lutLen × 8 bytes` | The unique 64-bit voxel values (full long encoding, §3) |
| Block indices | `32³ × 2 = 65,536 bytes` | 16-bit indices into LUT, in **z-curve (Morton) order** |
| `hash` | 8 bytes | Integrity check |

**Maximum section size:** `32×32×32×8` (all unique) `+ 8+8+4+8` header/footer.

**Spatial ordering: Morton (z-curve) code** — NOT raster/linear order.
`lin2z(idx)` interleaves the 3 axis bits; `z2lin(morton)` reverses it.
When writing/reading a Python parser, apply `lin2z` / `z2lin` accordingly.

---

## 7. Default Storage Backend

From `StorageConfigUtil.createDefaultSerializer()`:

```java
var baseDB = new RocksDBStorageBackend.Config();
var compressor = new ZSTDCompressor.Config();   // compressionLevel = 1
var compression = new CompressionStorageAdaptor.Config();
// compression wraps baseDB with ZSTD
// then SectionSerializationStorage wraps compression
```

**Default: RocksDB + ZSTD (level 1) compression.** Alternative backends available:
LMDB, Redis, in-memory — but RocksDB is what ships by default and is used unless the
user edits `config.json`.

The ID mapping (block name ↔ Voxy integer ID) is stored as a separate RocksDB key via
`storage.putIdMapping()` / `storage.getIdMappingsData()`.

---

## 8. Block ID Mapping

The block state ID in bits 27–46 is **NOT** the Minecraft block registry ID. It is:
- Assigned lazily at first encounter (`registerNewBlockState()` on the Mapper)
- Persisted in the same RocksDB store as the section data (separate key prefix)
- Sequential starting from 1 (0 = air is implicit)
- Keyed by `BlockState.toString()` (the full property string, e.g.
  `minecraft:grass_block[snowy=false]`)
- Stable intra-world but **different between worlds** (no canonical mapping)

**For our training pipeline:** when extracting from a Voxy world, we must read the
ID mapping table first, then decode voxels. We cannot assume any mapping from our
`block_vocab.json`.

---

## 9. Python Decoding Recipe

```python
import struct, numpy as np

# --- Section key round-trip ---
def make_key(lvl, x, y, z):
    return (
        ((lvl & 0xF) << 60) |
        ((y & 0xFF) << 52) |
        ((z & 0xFFFFFF) << 28) |
        ((x & 0xFFFFFF) << 4)
    )

def decode_key(key):
    lvl = (key >> 60) & 0xF
    y   = np.int8((key >> 52) & 0xFF)
    z   = np.int32((key << 12) >> 40)   # sign-extend 24-bit
    x   = np.int32((key << 36) >> 40)
    return int(lvl), int(x), int(y), int(z)


# --- Voxel long decoding ---
BLOCK_ID_SHIFT  = 27
BLOCK_ID_MASK   = (1 << 20) - 1        # 20 bits
BIOME_ID_SHIFT  = 47
BIOME_ID_MASK   = (1 << 9) - 1         # 9 bits
LIGHT_SHIFT     = 56
LIGHT_MASK      = 0xFF                  # 8 bits (sky<<4 | block)

def decode_voxel(v: int):
    block_id = (v >> BLOCK_ID_SHIFT) & BLOCK_ID_MASK
    biome_id = (v >> BIOME_ID_SHIFT) & BIOME_ID_MASK
    light    = (v >> LIGHT_SHIFT) & LIGHT_MASK
    sky_light   = light & 0xF
    block_light = (light >> 4) & 0xF
    return block_id, biome_id, sky_light, block_light


# --- Morton code (z-curve) helpers ---
def _split3(a):
    a &= 0x1FFFFF
    a = (a | (a << 32)) & 0x1F00000000FFFF
    a = (a | (a << 16)) & 0x1F0000FF0000FF
    a = (a | (a <<  8)) & 0x100F00F00F00F00F
    a = (a | (a <<  4)) & 0x10C30C30C30C30C3
    a = (a | (a <<  2)) & 0x1249249249249249
    return a

def lin2z(idx):
    x, y, z = idx & 31, (idx >> 5) & 31, (idx >> 10) & 31
    return _split3(x) | (_split3(y) << 1) | (_split3(z) << 2)


# --- Parse a serialized section (bytes) ---
def parse_section(data: bytes):
    off = 0
    key,      = struct.unpack_from('>q', data, off); off += 8
    metadata, = struct.unpack_from('>q', data, off); off += 8
    lut_len,  = struct.unpack_from('>I', data, off); off += 4
    lut = np.frombuffer(data, dtype='>i8', count=lut_len, offset=off); off += lut_len * 8
    # indices are Morton-ordered
    indices_morton = np.frombuffer(data, dtype='>u2', count=32**3, offset=off); off += 32**3 * 2
    # rearrange to linear order
    linear_order = np.argsort([lin2z(i) for i in range(32**3)])
    indices_linear = indices_morton[linear_order]
    voxels = lut[indices_linear].reshape(32, 32, 32)   # [x, z, y] → reorder as needed
    return decode_key(key), voxels
```

---

## 10. Corrections to Prior PROJECT-OUTLINE.md Assumptions

| Prior claim | Reality |
|---|---|
| "palette + 16-bit indices" | ✅ Correct structurally, but palette entries are **full 64-bit longs** containing block+biome+light |
| "32³ section format" | ✅ Correct |
| "RocksDB with world_sections column family" | ✅ RocksDB is correct default, but no "column family" — all in one RocksDB store with key prefixes |
| Linear voxel ordering assumed | ❌ Spatial order is **Morton / z-curve** on disk |
| No mention of biome per-voxel | ❌ Each voxel carries 9-bit biome ID (bits 47–55) |
| No mention of light per-voxel | ❌ Each voxel carries 8-bit light (bits 56–63: sky<<4\|block) |
| "Block ID = Minecraft registry ID" (implied) | ❌ Voxy uses its own **internal mapped IDs**, world-specific |
| LOD downsampling = majority vote (implied) | ❌ Opacity-biased corner selection (§5) |

---

*Keep this file updated as integration work proceeds. Cross-reference with `docs/AC.md`.*
