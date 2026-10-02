"""Parity test against ProDy's calcAdaptiveANM (SERCA, ProDy 2.4.1).

ProDy and torch live in different interpreters on this machine (the Levin
ligandmpnn env has torch but a broken scipy.sparse that ProDy's import chain
needs), so the reference run happens in a subprocess of the designer venv and
is compared in-process. Skipped when either side is unavailable.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aanm_torch.adaptive import calc_adaptive_anm_one_way  # noqa: E402
from aanm_torch.pdbio import ca_of, parse_bb  # noqa: E402

PRODY_PY = Path("/Users/cybertruck/Documents/designer/machine/.venv-morph/bin/python")
SERCA = Path("/Users/cybertruck/Documents/designer/machine/machines/serca")

DRIVER = """
import json
import numpy as np
from prody import calcAdaptiveANM, AANM_ONEWAY
import prody
prody.LOGGER.verbosity = "none"

def parse_ca(fn):
    ca = {}
    for ln in open(fn):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            ca[(ln[21], int(ln[22:26]), ln[26])] = np.array(
                [float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return ca

caA = parse_ca("__SERCA__/E1_2Ca_aligned.pdb")
caB = parse_ca("__SERCA__/E1_ATP_aligned.pdb")
common = sorted(set(caA) & set(caB))
A = np.array([caA[k] for k in common])
B = np.array([caB[k] for k in common])
rmsds = []
def cb(**kw):
    rmsds.append(float(kw["rmsd"]))
calcAdaptiveANM(A.copy(), B.copy(), n_steps=5, mode=AANM_ONEWAY,
                n_modes=10, callback_func=cb, f=0.3)
json.dump(rmsds, open("__OUT__", "w"))
"""


@unittest.skipIf(not PRODY_PY.exists(), "designer .venv-morph python missing")
@unittest.skipIf(not (SERCA / "E1_2Ca_aligned.pdb").exists(), "SERCA fixtures missing")
class TestProDyParity(unittest.TestCase):
    def test_one_way_rmsds_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rmsds.json"
            driver = DRIVER.replace("__SERCA__", str(SERCA)).replace("__OUT__", str(out))
            r = subprocess.run([str(PRODY_PY), "-c", driver],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            ref = json.loads(out.read_text())

        atomsA, _ = parse_bb(str(SERCA / "E1_2Ca_aligned.pdb"))
        atomsB, _ = parse_bb(str(SERCA / "E1_ATP_aligned.pdb"))
        caA, caB = ca_of(atomsA), ca_of(atomsB)
        common = sorted(set(caA) & set(caB))
        A = torch.tensor([caA[k] for k in common], dtype=torch.float64)
        B = torch.tensor([caB[k] for k in common], dtype=torch.float64)
        _, info = calc_adaptive_anm_one_way(A, B, 5, n_modes=10, f=0.3, solver="eigh")
        mine = info["rmsds"][1:]
        self.assertEqual(len(mine), len(ref))
        for m, r in zip(mine, ref):
            self.assertAlmostEqual(m, r, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
