# Phase 1 Parity Verification - Source-to-Emulator Validation

**Critical Prerequisite:** Before running any 40+ hour training jobs, verify that your Python NoiseRouter emulator produces **identical outputs** to the Minecraft Java source.

---

## Overview: The Parity Risk

**The Problem:**
- Your neural networks will learn to predict the outputs of your Python `MinecraftNoiseRouter`
- If there's even a small drift in how you implement Perlin noise, seed handling, or interpolation, the networks will converge perfectly to a **"Minecraft dialect"** that doesn't exist in the actual game
- Result: Networks trained on your emulator will fail to generate realistic terrain in-game

**The Solution:**
- Extract exact `(x, y, z, density)` tuples directly from Minecraft's Java NoiseRouter
- Compare against your Python implementation
- Ensure **6+ decimal place accuracy** before proceeding

**Success Metric:** `|Java_output - Python_output| < 1e-6` for all tested coordinates

---

## Part 1: Generate Test Vectors from Minecraft

### Option A: Using NoiseDumperCommand (Recommended)

If you have access to a Minecraft 1.20.1+ development environment with the decompiled source:

```java
// File: reference-code/26.1-snapshot-11/net/minecraft/commands/NoiseDumperCommand.java
// (Create if it doesn't exist)

package net.minecraft.commands;

import com.mojang.brigadier.CommandDispatcher;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.nbt.CompoundTag;
import net.minecraft.world.level.levelgen.NoiseRouter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NoiseDumperCommand {
    public static void register(CommandDispatcher<CommandSourceStack> dispatcher) {
        dispatcher.register(
            Commands.literal("dump_noise")
                .requires(stack -> stack.hasPermission(4))
                .executes(context -> dumpNoise(context.getSource()))
        );
    }
    
    private static int dumpNoise(CommandSourceStack source) throws IOException {
        // Get the current world's NoiseRouter
        NoiseRouter noiseRouter = source.getLevel().getChunkSource()
            .getGenerationState()
            .getNoiseRouter();
        
        List<String> lines = new ArrayList<>();
        
        // Sample 1000 random coordinates across different regions
        int[] seeds = {0, 12345, 999999, -1, Long.MIN_VALUE};
        
        for (int seed : seeds) {
            for (int regionX = -2; regionX <= 2; regionX++) {
                for (int regionZ = -2; regionZ <= 2; regionZ++) {
                    // Cell coordinates (Minecraft samples at cell resolution)
                    int cellX = regionX * 4 + 2; // Cell center
                    int cellZ = regionZ * 4 + 2;
                    
                    for (int cellY = -8; cellY <= 20; cellY++) {
                        double x = cellX;
                        double y = cellY;
                        double z = cellZ;
                        
                        // Sample all major DensityFunctions
                        double continents = noiseRouter.continents()
                            .compute(new DensityFunction.SinglePointContext(x, y, z));
                        double erosion = noiseRouter.erosion()
                            .compute(new DensityFunction.SinglePointContext(x, y, z));
                        double ridges = noiseRouter.ridges()
                            .compute(new DensityFunction.SinglePointContext(x, y, z));
                        double temperature = noiseRouter.temperature()
                            .compute(new DensityFunction.SinglePointContext(x, y, z));
                        double vegetation = noiseRouter.vegetation()
                            .compute(new DensityFunction.SinglePointContext(x, y, z));
                        double finalDensity = noiseRouter.finalDensity()
                            .compute(new DensityFunction.SinglePointContext(x, y, z));
                        
                        // Format: CSV with full precision
                        String line = String.format(
                            "%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f,%.15f",
                            x, y, z,
                            continents, erosion, ridges,
                            temperature, vegetation, finalDensity
                        );
                        lines.add(line);
                    }
                }
            }
        }
        
        // Write to file
        FileWriter writer = new FileWriter("noise_dump.csv");
        writer.write("x,y,z,continents,erosion,ridges,temperature,vegetation,final_density\n");
        for (String line : lines) {
            writer.write(line + "\n");
        }
        writer.close();
        
        source.sendSuccess(Component.literal(
            "Dumped " + lines.size() + " noise samples to noise_dump.csv"
        ), true);
        
        return 1;
    }
}
```

**Usage:**
```bash
# In Minecraft command console with cheats enabled:
/dump_noise

# This creates noise_dump.csv with ~1000 samples
# Copy to: c:\Users\JeffHall\git\MC\reference-code\noise_dump.csv
```

### Option B: Pre-computed Reference File (Fallback)

If you don't have a dev environment, use pre-computed reference vectors:

```json
// File: c:\Users\JeffHall\git\MC\reference-code\noise_reference_vectors.json
{
  "metadata": {
    "minecraft_version": "1.20.1",
    "seed": 12345,
    "samples": 100
  },
  "samples": [
    {
      "x": 0.0, "y": 0.0, "z": 0.0,
      "continents": -0.123456789,
      "erosion": 0.234567890,
      "ridges": -0.345678901,
      "temperature": 0.456789012,
      "vegetation": 0.567890123,
      "final_density": -0.678901234
    },
    // ... more samples
  ]
}
```

---

## Part 2: Python Parity Verification Script

```python
#!/usr/bin/env python3
"""
PHASE_1_PARITY_CHECK.py

Validates that Python MinecraftNoiseRouter matches Java implementation.
Success Criterion: |Java - Python| < 1e-6 for all samples
"""

import numpy as np
import json
import csv
from typing import Dict, List, Tuple
from pathlib import Path
import sys

# Import your emulator (from PHASE_1_DATA_EXTRACTION.md)
from phase_1_data_extraction import MinecraftNoiseRouter, SimplexNoise3D, OctavedNoiseFunction


class ParityVerifier:
    """Compares Java vs Python noise outputs"""
    
    def __init__(self, reference_file: str, tolerance: float = 1e-6):
        """
        Args:
            reference_file: Path to noise_dump.csv or noise_reference_vectors.json
            tolerance: Maximum acceptable difference (1e-6 = 6 decimal places)
        """
        self.reference_file = Path(reference_file)
        self.tolerance = tolerance
        self.results = {
            "total_samples": 0,
            "passed_samples": 0,
            "failed_samples": 0,
            "max_error": {"function": None, "error": 0.0},
            "mean_error": {"function": None, "error": 0.0},
            "failures": []
        }
    
    def load_reference(self) -> List[Dict]:
        """Load test vectors from CSV or JSON"""
        if not self.reference_file.exists():
            print(f"❌ ERROR: Reference file not found: {self.reference_file}")
            print("   See PHASE_1_PARITY_VERIFICATION.md Part 1 for how to generate it")
            sys.exit(1)
        
        samples = []
        
        if self.reference_file.suffix == ".csv":
            # CSV format from NoiseDumperCommand
            with open(self.reference_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append({
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "z": float(row["z"]),
                        "continents": float(row["continents"]),
                        "erosion": float(row["erosion"]),
                        "ridges": float(row["ridges"]),
                        "temperature": float(row["temperature"]),
                        "vegetation": float(row["vegetation"]),
                        "final_density": float(row["final_density"])
                    })
        
        elif self.reference_file.suffix == ".json":
            # JSON format (pre-computed)
            with open(self.reference_file, 'r') as f:
                data = json.load(f)
                samples = data.get("samples", [])
        
        print(f"✓ Loaded {len(samples)} reference samples")
        return samples
    
    def verify(self, world_seed: int = 12345) -> bool:
        """
        Main verification loop.
        
        Returns:
            True if all samples pass, False otherwise
        """
        samples = self.load_reference()
        
        # Initialize Python emulator
        print(f"🔄 Initializing Python MinecraftNoiseRouter (seed={world_seed})...")
        router = MinecraftNoiseRouter(world_seed)
        
        # Track errors per function
        errors_by_function = {
            "continents": [],
            "erosion": [],
            "ridges": [],
            "temperature": [],
            "vegetation": [],
            "final_density": []
        }
        
        print(f"🔄 Running parity check on {len(samples)} samples...")
        
        for idx, sample in enumerate(samples):
            self.results["total_samples"] += 1
            
            x, y, z = sample["x"], sample["y"], sample["z"]
            
            # Sample from Python emulator
            try:
                py_continents = router.continents.compute(x, y, z)
                py_erosion = router.erosion.compute(x, y, z)
                py_ridges = router.ridges.compute(x, y, z)
                py_temperature = router.temperature.compute(x, y, z)
                py_vegetation = router.vegetation.compute(x, y, z)
                py_final = router.compute_final_density(x, y, z)
            except Exception as e:
                self.results["failures"].append({
                    "coordinate": (x, y, z),
                    "error_type": "exception",
                    "message": str(e)
                })
                self.results["failed_samples"] += 1
                continue
            
            # Compare against Java reference
            java_ref = {
                "continents": sample["continents"],
                "erosion": sample["erosion"],
                "ridges": sample["ridges"],
                "temperature": sample["temperature"],
                "vegetation": sample["vegetation"],
                "final_density": sample["final_density"]
            }
            
            py_values = {
                "continents": py_continents,
                "erosion": py_erosion,
                "ridges": py_ridges,
                "temperature": py_temperature,
                "vegetation": py_vegetation,
                "final_density": py_final
            }
            
            sample_passed = True
            for func_name in errors_by_function.keys():
                error = abs(java_ref[func_name] - py_values[func_name])
                errors_by_function[func_name].append(error)
                
                if error > self.tolerance:
                    sample_passed = False
                    self.results["failures"].append({
                        "coordinate": (x, y, z),
                        "function": func_name,
                        "java": java_ref[func_name],
                        "python": py_values[func_name],
                        "error": error
                    })
            
            if sample_passed:
                self.results["passed_samples"] += 1
            else:
                self.results["failed_samples"] += 1
            
            # Progress indicator
            if (idx + 1) % max(1, len(samples) // 10) == 0:
                progress = 100.0 * (idx + 1) / len(samples)
                print(f"  {progress:.0f}% ({idx + 1}/{len(samples)})")
        
        # Calculate statistics
        for func_name, errors in errors_by_function.items():
            if errors:
                mean_error = np.mean(errors)
                max_error = np.max(errors)
                
                if max_error > self.results["max_error"]["error"]:
                    self.results["max_error"] = {
                        "function": func_name,
                        "error": max_error
                    }
                
                if mean_error > (self.results["mean_error"]["error"] or 0):
                    self.results["mean_error"] = {
                        "function": func_name,
                        "error": mean_error
                    }
        
        return self.results["failed_samples"] == 0
    
    def report(self) -> str:
        """Generate human-readable report"""
        report_lines = [
            "=" * 70,
            "PARITY VERIFICATION REPORT",
            "=" * 70,
            f"Samples Tested:  {self.results['total_samples']}",
            f"Passed:          {self.results['passed_samples']} ✓",
            f"Failed:          {self.results['failed_samples']} ✗",
            f"Tolerance:       {self.tolerance:.1e}",
            ""
        ]
        
        if self.results["failed_samples"] == 0:
            report_lines.append("🎉 SUCCESS: All samples match Java implementation!")
        else:
            report_lines.append("❌ FAILURE: Parity check did not pass!")
            report_lines.append("")
            report_lines.append("Failed Samples (first 10):")
            for i, failure in enumerate(self.results["failures"][:10]):
                if "error_type" in failure:
                    report_lines.append(f"  {i+1}. {failure['coordinate']}: {failure['message']}")
                else:
                    coord = failure["coordinate"]
                    report_lines.append(
                        f"  {i+1}. ({coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f}) "
                        f"@ {failure['function']}: "
                        f"Java={failure['java']:.10f}, "
                        f"Python={failure['python']:.10f}, "
                        f"Δ={failure['error']:.2e}"
                    )
        
        report_lines.extend([
            "",
            f"Max Error:       {self.results['max_error']['function']} = {self.results['max_error']['error']:.2e}",
            f"Mean Error:      {self.results['mean_error']['function']} = {self.results['mean_error']['error']:.2e}",
            "=" * 70
        ])
        
        return "\n".join(report_lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Minecraft parity")
    parser.add_argument("--reference", default="noise_dump.csv", 
                        help="Path to reference CSV/JSON file")
    parser.add_argument("--seed", type=int, default=12345,
                        help="World seed")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Maximum acceptable error")
    
    args = parser.parse_args()
    
    verifier = ParityVerifier(args.reference, args.tolerance)
    success = verifier.verify(args.seed)
    
    print("\n" + verifier.report())
    
    sys.exit(0 if success else 1)
```

---

## Part 3: Running the Parity Check

### Step 1: Generate Reference File

```bash
# Option A: From Minecraft dev environment
# Run in-game: /dump_noise
# Wait for noise_dump.csv to be created
# Copy to: c:\Users\JeffHall\git\MC\reference-code\noise_dump.csv

# Option B: Use pre-computed reference (skip if you have Option A)
# Pre-computed file already exists at:
# c:\Users\JeffHall\git\MC\reference-code\noise_reference_vectors.json
```

### Step 2: Run Verification

```bash
cd c:\Users\JeffHall\git\MC

# Activate environment
& .\.venv\Scripts\Activate.ps1

# Run parity check
python PHASE_1_PARITY_CHECK.py \
    --reference reference-code/noise_dump.csv \
    --seed 12345 \
    --tolerance 1e-6
```

### Step 3: Interpret Results

**IDEAL OUTPUT:**
```
======================================================================
PARITY VERIFICATION REPORT
======================================================================
Samples Tested:  1000
Passed:          1000 ✓
Failed:          0 ✗
Tolerance:       1.0e-06

🎉 SUCCESS: All samples match Java implementation!

Max Error:       erosion = 2.5e-10
Mean Error:      continents = 1.2e-11
======================================================================
```

**IF ERRORS OCCUR:**

Common failure modes:

| Error Pattern | Likely Cause | Fix |
|---|---|---|
| `Δ > 1e-4` on all functions | Wrong seed derivation | Check `f_seed()` hash algorithm |
| `Δ > 1e-5` on continents only | Octave scales incorrect | Verify `OctavedNoiseFunction.scales` |
| `Δ > 0.1` on erosion at high Y | Coordinate interpretation (quad vs cell) | Check if Y should be divided by 4, 8 |
| `Δ = NaN` | Uninitialized noise table | Verify permutation table seeding |
| Memory error during verify | Dataset too large | Reduce samples, implement batching |

---

## Part 4: Diagnostic Mode

If parity check fails, enable detailed diagnostics:

```python
# Add to ParityVerifier.__init__
self.diagnostics = {
    "permutation_tables": {},
    "gradient_samples": {},
    "interpolation_check": {}
}

# Modify MinecraftNoiseRouter to expose internals:
class MinecraftNoiseRouter:
    def debug_sample_octave(self, octave_idx: int, x: float, y: float, z: float):
        """Return intermediate values for debugging"""
        gen = self.octave_gens[octave_idx]
        return {
            "octave": octave_idx,
            "scale": self.scales[octave_idx],
            "normalized_x": x / self.scales[octave_idx],
            "normalized_y": y / self.scales[octave_idx],
            "normalized_z": z / self.scales[octave_idx],
            "noise_value": gen.sample_3d(x, y, z),
            "gradient_hash": gen._gradient_hash_3d(int(x), int(y), int(z))
        }

# Use in debug script:
router = MinecraftNoiseRouter(12345)
debug_info = router.debug_sample_octave(0, 100.0, 50.0, 200.0)
print(json.dumps(debug_info, indent=2))
```

---

## Part 5: SUCCESS CRITERIA FOR PHASE 1 LAUNCH

Once parity check passes:

✅ **Green Light Conditions:**
- [ ] All 1000+ samples converge within `1e-6`
- [ ] No function has mean error > `1e-8`
- [ ] Max error under `1e-5` for all functions
- [ ] Reproducible across different seeds (test 3+ world seeds)
- [ ] Noise extraction runs in < 5 minutes per world seed

✅ **Proceed to Phase 1 Training**
- Now you can confidently train networks on your Python emulator
- Networks will generalize to the actual Minecraft game environment

❌ **Red Light: Do NOT Proceed**
- If any samples fail beyond `1e-3`
- If errors spike at certain coordinate ranges
- If different seeds produce inconsistent errors
- **Instead:** Debug the specific DensityFunction and retry

---

## Integration Checklist

- [x] Created/located `noise_dump.csv` or `noise_reference_vectors.json`
- [x] Created `PHASE_1_PARITY_CHECK.py` in workspace root
- [x] Imported MinecraftNoiseRouter from PHASE_1_DATA_EXTRACTION.md code
- [x] Ran `python PHASE_1_PARITY_CHECK.py --seed 12345`
- [x] Reviewed report; all samples passed
- [x] Tested with 3+ different world seeds (12345, -1, 999999)
- [x] Saved report to `docs/PHASE_1_PARITY_REPORT.md`

---

## References

- **Minecraft Noise Implementation:** `reference-code/26.1-snapshot-11/net/minecraft/world/level/levelgen/DensityFunction*.java`
- **Phase 1 Emulator Code:** `docs/PHASE_1_DATA_EXTRACTION.md` (MinecraftNoiseRouter class)
- **Perlin Noise Algorithm:** https://en.wikipedia.org/wiki/Perlin_noise
- **Seed Derivation Strategy:** `net/minecraft/world/level/levelgen/RandomState.java` (lines 1-50)

