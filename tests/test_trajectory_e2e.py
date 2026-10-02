import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth import chain_atoms, helix_ca, helix_p, nuc_chain, rigid, write_pdb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def make_machine(tmp, with_sidechains=False):
    """Two chains x two states: P/Q helices; state 2 bends both chains and
    rigidly displaces Q relative to P (assembly network should carry that)."""
    caP1 = helix_ca(45, center=(0, 0, 0))
    caQ1 = helix_ca(45, center=(35, 0, 0))
    caP2 = helix_ca(45, radius=10.0, phase=0.4, center=(0, 0, 0))
    caQ2 = rigid(helix_ca(45, radius=10.0, phase=0.9, center=(35, 0, 0)), seed=7)
    s1 = {}
    s1.update(chain_atoms(caP1, "P", with_sidechains=with_sidechains))
    s1.update(chain_atoms(caQ1, "Q", with_sidechains=with_sidechains))
    s2 = {}
    s2.update(chain_atoms(caP2, "P", with_sidechains=with_sidechains))
    s2.update(chain_atoms(caQ2, "Q", with_sidechains=with_sidechains))
    d = Path(tmp)
    write_pdb(d / "s1.pdb", s1)
    write_pdb(d / "s2.pdb", s2)
    return d


def run_cli(tmp, extra_cfg, extra_args=()):
    cfg = {
        "name": "synth",
        "dir": tmp,
        "states": [{"file": "s1.pdb", "name": "S1"},
                   {"file": "s2.pdb", "name": "S2"}],
        "aanm": {"n_steps": 4, "f": 0.3, "n_modes": 6},
        "dwell": 1,
        "morph": 4,
        "loop": False,
        "batch": False,
        "out": "traj.pdb",
    }
    cfg.update(extra_cfg)
    cfg_path = Path(tmp) / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    cmd = [sys.executable, str(ROOT / "scripts" / "make_aanm_trajectory.py"),
           "--config", str(cfg_path), "--device", "cpu", *extra_args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout, (Path(tmp) / "traj.pdb")


def read_models(fn):
    models, cur = [], None
    for ln in open(fn):
        if ln.startswith("MODEL"):
            cur = []
        elif ln.startswith("ENDMDL"):
            models.append(cur)
        elif ln.startswith("ATOM"):
            cur.append((ln[21], int(ln[22:26]), ln[12:16].strip()))
    return models


class TestTrajectoryE2E(unittest.TestCase):
    def test_anchor_backbone(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_machine(tmp)
            out, path = run_cli(tmp, {})
            models = read_models(path)
            self.assertEqual(len(models), 1 + 4 + 1)  # dwell + morph + dwell
            self.assertGreater(len(models[0]), 300)   # 2 chains x 43 res x 4 atoms
            # morph frame: every chain got the anchor backbone
            chains = {a[0] for a in models[2]}
            self.assertEqual(chains, {"P", "Q"})
            self.assertIn("max clashes", out)

    def test_assembly_with_sidechains(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_machine(tmp, with_sidechains=True)
            out, path = run_cli(tmp, {"sidechains": True,
                                      "aanm": {"n_steps": 4, "f": 0.3, "n_modes": 6,
                                               "network": "assembly"}})
            models = read_models(path)
            m = models[2]
            atoms = {(a[0], a[1]) for a in m}
            names = {a[2] for a in m}
            self.assertIn("CB", names)                # sidechains present
            self.assertEqual({a[0] for a in m}, {"P", "Q"})
            per_chain_res = {c: sum(1 for x in atoms if x[0] == c) for c in ("P", "Q")}
            self.assertEqual(per_chain_res["P"], 43)
            self.assertEqual(per_chain_res["Q"], 43)

    def test_alternating_mode_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_machine(tmp)
            out, path = run_cli(tmp, {"aanm": {"n_steps": 4, "f": 0.3, "n_modes": 6,
                                               "mode": "alternating"}})
            models = read_models(path)
            self.assertEqual(len(models), 6)

    def test_nucleic_only_machine(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_pdb(d / "r1.pdb", nuc_chain(helix_p(45), "A"))
            write_pdb(d / "r2.pdb", nuc_chain(rigid(helix_p(45, radius=16.0, phase=1.0),
                                                    seed=3), "A"))
            cfg = {
                "name": "rna", "dir": tmp,
                "states": [{"file": "r1.pdb", "name": "R1"},
                           {"file": "r2.pdb", "name": "R2"}],
                "aanm": {"n_steps": 4, "f": 0.3, "n_modes": 6},
                "dwell": 1, "morph": 4, "loop": False, "batch": False,
                "out": "rna.pdb",
            }
            p = d / "cfg.json"
            p.write_text(json.dumps(cfg))
            cmd = [sys.executable, str(ROOT / "scripts" / "make_aanm_trajectory.py"),
                   "--config", str(p), "--device", "cpu"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            models = read_models(d / "rna.pdb")
            self.assertEqual(len(models), 6)
            names = {a[2] for a in models[2]}
            self.assertIn("P", names)
            self.assertIn("C4'", names)
            # per-residue node count matches the 43 reconstructed nucleotides
            self.assertEqual(sum(1 for a in models[2] if a[2] == "P"), 43)
            self.assertIn("(auto)", r.stdout)  # cutoff auto-suggested from P spacing

    def test_empty_frames_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 10-residue chains: reconstruction works but the per-chain site
            # fit needs >= 30 CAs, so morph frames end up empty -> guard fires
            s1 = {}
            s1.update(chain_atoms(helix_ca(10, center=(0, 0, 0)), "P"))
            s1.update(chain_atoms(helix_ca(10, center=(40, 0, 0)), "Q"))
            s2 = {}
            s2.update(chain_atoms(helix_ca(10, radius=9.0, center=(0, 0, 0)), "P"))
            s2.update(chain_atoms(helix_ca(10, radius=9.0, center=(40, 0, 0)), "Q"))
            d = Path(tmp)
            write_pdb(d / "s1.pdb", s1)
            write_pdb(d / "s2.pdb", s2)
            cfg = {
                "name": "bad", "dir": tmp,
                "states": [{"file": "s1.pdb", "name": "S1"},
                           {"file": "s2.pdb", "name": "S2"}],
                "aanm": {"n_steps": 2, "n_modes": 3},
                "dwell": 1, "morph": 2, "loop": False, "batch": False,
                "out": "traj_bad.pdb",
            }
            p = d / "cfg_bad.json"
            p.write_text(json.dumps(cfg))
            cmd = [sys.executable, str(ROOT / "scripts" / "make_aanm_trajectory.py"),
                   "--config", str(p), "--device", "cpu"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("empty", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()
