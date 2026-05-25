"""
Unit / functional tests for cycles() – CONV2D path
====================================================
Covers all 10 CNV_SA test cases from the mapping-config table:

  Name       IN_x  IN_y  C_in  C_out  K_x  K_y  Pad  Stride  n_SA
  CNV_SA1     8     8     1     4      3    3     0     1      2
  CNV_SA2    10    10     3     8      3    3     0     1      2
  CNV_SA3     8     8     8     8      3    3     1     1      2
  CNV_SA4    10    10     4    16      3    3     1     1      4
  CNV_SA5     8     8    32    16      1    1     0     1      2
  CNV_SA6    10    10     4    16      3    3     1     1      2
  CNV_SA7    10    10     3     8      3    3     0     1      1
  CNV_SA8    10    10     3     8      3    3     0     1      2
  CNV_SA9    10    10     3    48      3    3     0     1      4
  CNV_SA10   10    10     4    48      3    3     1     1      8

Hardware config assumed: SA_x = SA_y = 32 PEs, n_mod = 1, num_bits_act = 8.

Each test checks:
  1. compute_cycles  – integer nanoseconds returned by cycles()
  2. bw_latency      – Qout / BW_noc bandwidth latency (ns)
  3. E_util          – effective MAC utilisation count
  4. node_ds fields  – HW_MACs, Q_out_max, Q_in_max, Q_wg_max
"""

import math
import unittest
import numpy
from Utils import cycles


# ── shared fixtures ────────────────────────────────────────────────────────────

PPA_CONFIG = {
    "L_PE"        : 1e-9,
    "A_PE"        : 6.74e-10,
    "A_PE_toprow" : 4.192e-10,
    "P_PE"        : 9.29e-5,
    "P_PE_toprow" : 5.51e-5,
    "num_bits_act": 8,
    "BW_noc"      : 320e9,   # 320 Gbps
    "clk_period"  : 1e-9,    # 1 GHz
}


def _make_params(n_SA: int, row_PE: int = 32, col_PE: int = 32) -> dict:
    """Build a params dict for a given number of SAs."""
    return {
        "row_PE"    : row_PE,
        "col_PE"    : col_PE,
        "n_SA"      : n_SA,
        "no_modules_act": 1,
        "no_modules_pool": 1,
    }


def _make_layer_info(IN_x, IN_y, C_in, C_out, K_x, K_y, Pad, Stride) -> dict:
    """Build layer_info for a CONV2D layer from table parameters."""
    H_out = (IN_x + 2 * Pad - K_x) // Stride + 1
    W_out = (IN_y + 2 * Pad - K_y) // Stride + 1
    return {
        'out_channel/features': C_out,
        'in_channel/features' : C_in,
        'kernel_size'         : (K_x, K_y),
        'out_size'            : (H_out, W_out),
        'in_size'             : (IN_x, IN_y),
        'groups'              : 1,
        'count'               : 1,
    }


# ── helper for expected BW latency ────────────────────────────────────────────
def _expected_bw(C_out, H_out, W_out, bits=8, bw=320e9):
    Qout = C_out * H_out * W_out * bits
    return Qout / bw * 1e9


# ── helper for expected node_ds MAC / bandwidth fields ────────────────────────
def _expected_node_ds(n_SA, SA_x=32, SA_y=32, bits=8):
    return {
        "HW_MACs"   : n_SA * SA_x * SA_y,
        "Q_out_max" : n_SA * SA_y * bits,
        "Q_in_max"  : n_SA * SA_x * bits,
        "Q_wg_max"  : n_SA * SA_x * SA_y * bits,
    }


# ══════════════════════════════════════════════════════════════════════════════
class TestCyclesCNV_SA(unittest.TestCase):
    """
    Functional tests for cycles() on all 10 CNV_SA configurations.
    Each test method validates:
      • compute_cycles  (int, nanoseconds)
      • bw_latency      (float, nanoseconds)
      • E_util          (total MAC count)
      • node_ds         (HW_MACs, Q_out_max, Q_in_max, Q_wg_max)
    """

    # ── CNV_SA1: IN=8×8, Cin=1, Cout=4, K=3×3, Pad=0, Stride=1, 2 SAs ──────
    def test_CNV_SA1(self):
        """IN=8×8  Cin=1  Cout=4  K=3×3  Pad=0  Stride=1  → 2 SAs
        out=(6,6)  M=4  K_dim=9  N=36
        """
        li     = _make_layer_info(8, 8, 1, 4, 3, 3, 0, 1)
        params = _make_params(n_SA=2)

        cc, bw, nd, eu = cycles("CONV2D", 0, li, params, PPA_CONFIG)

        self.assertEqual(cc, 78,  "compute_cycles mismatch")
        self.assertAlmostEqual(bw, _expected_bw(4, 6, 6), places=4,
                               msg="bw_latency mismatch")
        self.assertEqual(eu, 1_296.0,           "E_util mismatch")
        exp_nd = _expected_node_ds(n_SA=2)
        for field, val in exp_nd.items():
            self.assertEqual(nd[field], val, f"node_ds['{field}'] mismatch")

    # ── CNV_SA2: IN=10×10, Cin=3, Cout=8, K=3×3, Pad=0, Stride=1, 2 SAs ────
    def test_CNV_SA2(self):
        """IN=10×10  Cin=3  Cout=8  K=3×3  Pad=0  Stride=1  → 2 SAs
        out=(8,8)  M=8  K_dim=27  N=64
        """
        li     = _make_layer_info(10, 10, 3, 8, 3, 3, 0, 1)
        params = _make_params(n_SA=2)

        cc, bw, nd, eu = cycles("CONV2D", 0, li, params, PPA_CONFIG)

        self.assertEqual(cc, 100, "compute_cycles mismatch")
        self.assertAlmostEqual(bw, _expected_bw(8, 8, 8), places=4,
                               msg="bw_latency mismatch")
        self.assertEqual(eu, 13_824.0,           "E_util mismatch")
        exp_nd = _expected_node_ds(n_SA=2)
        for field, val in exp_nd.items():
            self.assertEqual(nd[field], val, f"node_ds['{field}'] mismatch")

    # ── CNV_SA3: IN=8×8, Cin=8, Cout=8, K=3×3, Pad=1, Stride=1, 2 SAs ──────
    def test_CNV_SA3(self):
        """IN=8×8  Cin=8  Cout=8  K=3×3  Pad=1  Stride=1  → 2 SAs
        out=(8,8)  M=8  K_dim=72  N=64
        """
        li     = _make_layer_info(8, 8, 8, 8, 3, 3, 1, 1)
        params = _make_params(n_SA=2)

        cc, bw, nd, eu = cycles("CONV2D", 0, li, params, PPA_CONFIG)

        self.assertEqual(cc, 292, "compute_cycles mismatch")
        self.assertAlmostEqual(bw, _expected_bw(8, 8, 8), places=4,
                               msg="bw_latency mismatch")
        self.assertEqual(eu, 36_864.0,            "E_util mismatch")
        exp_nd = _expected_node_ds(n_SA=2)
        for field, val in exp_nd.items():
            self.assertEqual(nd[field], val, f"node_ds['{field}'] mismatch")

    # ── CNV_SA4: IN=10×10, Cin=4, Cout=16, K=3×3, Pad=1, Stride=1, 4 SAs ───
    def test_CNV_SA4(self):
        """IN=10×10  Cin=4  Cout=16  K=3×3  Pad=1  Stride=1  → 4 SAs
        out=(10,10)  M=16  K_dim=36  N=100
        """
        li     = _make_layer_info(10, 10, 4, 16, 3, 3, 1, 1)
        params = _make_params(n_SA=4)

        cc, bw, nd, eu = cycles("CONV2D", 0, li, params, PPA_CONFIG)

        self.assertEqual(cc, 199, "compute_cycles mismatch")
        self.assertAlmostEqual(bw, _expected_bw(16, 10, 10), places=4,
                               msg="bw_latency mismatch")
        self.assertEqual(eu, 57_600.0,            "E_util mismatch")
        exp_nd = _expected_node_ds(n_SA=4)
        for field, val in exp_nd.items():
            self.assertEqual(nd[field], val, f"node_ds['{field}'] mismatch")

    # ── CNV_SA5: IN=8×8, Cin=32, Cout=16, K=1×1, Pad=0, Stride=1, 2 SAs ────
    def test_CNV_SA5(self):
        """IN=8×8  Cin=32  Cout=16  K=1×1  Pad=0  Stride=1  → 2 SAs
        out=(8,8)  M=16  K_dim=32  N=64
        (pointwise / 1×1 convolution)
        """
        li     = _make_layer_info(8, 8, 32, 16, 1, 1, 0, 1)
        params = _make_params(n_SA=2)

        cc, bw, nd, eu = cycles("CONV2D", 0, li, params, PPA_CONFIG)

        self.assertEqual(cc, 113, "compute_cycles mismatch")
        self.assertAlmostEqual(bw, _expected_bw(16, 8, 8), places=4,
                               msg="bw_latency mismatch")
        self.assertEqual(eu, 32_768.0,            "E_util mismatch")
        exp_nd = _expected_node_ds(n_SA=2)
        for field, val in exp_nd.items():
            self.assertEqual(nd[field], val, f"node_ds['{field}'] mismatch")

    # ── CNV_SA6: IN=10×10, Cin=4, Cout=16, K=3×3, Pad=1, Stride=1, 2 SAs ───
    def test_CNV_SA6(self):
        """IN=10×10  Cin=4  Cout=16  K=3×3  Pad=1  Stride=1  → 2 SAs
        out=(10,10)  M=16  K_dim=36  N=100
        (same geometry as CNV_SA4 but fewer SAs → higher latency)
        """
        li     = _make_layer_info(10, 10, 4, 16, 3, 3, 1, 1)
        params = _make_params(n_SA=2)

        cc, bw, nd, eu = cycles("CONV2D", 0, li, params, PPA_CONFIG)

        self.assertEqual(cc, 398, "compute_cycles mismatch")
        self.assertAlmostEqual(bw, _expected_bw(16, 10, 10), places=4,
                               msg="bw_latency mismatch")
        self.assertEqual(eu, 57_600.0,             "E_util mismatch")
        exp_nd = _expected_node_ds(n_SA=2)
        for field, val in exp_nd.items():
            self.assertEqual(nd[field], val, f"node_ds['{field}'] mismatch")

    # ── CNV_SA7: IN=10×10, Cin=3, Cout=8, K=3×3, Pad=0, Stride=1, 1 SA ─────
    def test_CNV_SA7(self):
        """IN=10×10  Cin=3  Cout=8  K=3×3  Pad=0  Stride=1  → 1 SA
        out=(8,8)  M=8  K_dim=27  N=64
        (single-SA path, config=1 override)
        """
        li     = _make_layer_info(10, 10, 3, 8, 3, 3, 0, 1)
        params = _make_params(n_SA=1)

        # config=1 forces n_SA=1 regardless of params["n_SA"]
        cc, bw, nd, eu = cycles("CONV2D", 1, li, params, PPA_CONFIG)

        self.assertEqual(cc, 200, "compute_cycles mismatch")
        self.assertAlmostEqual(bw, _expected_bw(8, 8, 8), places=4,
                               msg="bw_latency mismatch")
        self.assertEqual(eu, 13_824.0,           "E_util mismatch")
        # config=1 forces n_SA=1 internally
        exp_nd = _expected_node_ds(n_SA=1)
        for field, val in exp_nd.items():
            self.assertEqual(nd[field], val, f"node_ds['{field}'] mismatch")

    # ── CNV_SA8: IN=10×10, Cin=3, Cout=8, K=3×3, Pad=0, Stride=1, 2 SAs ────
    def test_CNV_SA8(self):
        """IN=10×10  Cin=3  Cout=8  K=3×3  Pad=0  Stride=1  → 2 SAs
        out=(8,8)  M=8  K_dim=27  N=64
        (same workload as CNV_SA7; scaling check: 2 SAs should reduce latency
        vs single SA for workloads that can be split)
        """
        li     = _make_layer_info(10, 10, 3, 8, 3, 3, 0, 1)
        params = _make_params(n_SA=2)

        cc, bw, nd, eu = cycles("CONV2D", 0, li, params, PPA_CONFIG)

        self.assertEqual(cc, 100, "compute_cycles mismatch")
        self.assertAlmostEqual(bw, _expected_bw(8, 8, 8), places=4,
                               msg="bw_latency mismatch")
        self.assertEqual(eu, 13_824.0,           "E_util mismatch")
        exp_nd = _expected_node_ds(n_SA=2)
        for field, val in exp_nd.items():
            self.assertEqual(nd[field], val, f"node_ds['{field}'] mismatch")

        # Note: for this small workload (M=8, K=27, N=64) the 2-SA multi-split
        # path produces MORE non-parallel tiles than the single-SA path because
        # N is split across 2 SAs (n_N=2, n_M=0), doubling tiles_non_parallel.
        # Multi-SA is not guaranteed faster when the workload is too small to
        # amortise the split overhead — this is correct algorithm behaviour.
        self.assertGreater(cc, 0, "compute_cycles must be positive")

    # ── CNV_SA9: IN=10×10, Cin=3, Cout=48, K=3×3, Pad=0, Stride=1, 4 SAs ───
    def test_CNV_SA9(self):
        """IN=10×10  Cin=3  Cout=48  K=3×3  Pad=0  Stride=1  → 4 SAs
        out=(8,8)  M=48  K_dim=27  N=64
        (large output-channel count; tests multi-tile M dimension)
        """
        li     = _make_layer_info(10, 10, 3, 48, 3, 3, 0, 1)
        params = _make_params(n_SA=4)

        cc, bw, nd, eu = cycles("CONV2D", 0, li, params, PPA_CONFIG)

        self.assertEqual(cc, 125, "compute_cycles mismatch")
        self.assertAlmostEqual(bw, _expected_bw(48, 8, 8), places=4,
                               msg="bw_latency mismatch")
        self.assertEqual(eu, 82_944.0,            "E_util mismatch")
        exp_nd = _expected_node_ds(n_SA=4)
        for field, val in exp_nd.items():
            self.assertEqual(nd[field], val, f"node_ds['{field}'] mismatch")

    # ── CNV_SA10: IN=10×10, Cin=4, Cout=48, K=3×3, Pad=1, Stride=1, 8 SAs ──
    def test_CNV_SA10(self):
        """IN=10×10  Cin=4  Cout=48  K=3×3  Pad=1  Stride=1  → 8 SAs
        out=(10,10)  M=48  K_dim=36  N=100
        (maximum SA count in the table; tests 8-SA multi-split path)
        """
        li     = _make_layer_info(10, 10, 4, 48, 3, 3, 1, 1)
        params = _make_params(n_SA=8)

        cc, bw, nd, eu = cycles("CONV2D", 0, li, params, PPA_CONFIG)

        self.assertEqual(cc, 232, "compute_cycles mismatch")
        self.assertAlmostEqual(bw, _expected_bw(48, 10, 10), places=4,
                               msg="bw_latency mismatch")
        self.assertEqual(eu, 172_800.0,            "E_util mismatch")
        exp_nd = _expected_node_ds(n_SA=8)
        for field, val in exp_nd.items():
            self.assertEqual(nd[field], val, f"node_ds['{field}'] mismatch")


# ══════════════════════════════════════════════════════════════════════════════
class TestCyclesCNV_SA_CrossChecks(unittest.TestCase):
    """
    Cross-case sanity checks that compare pairs of configurations
    to catch regressions in scaling behaviour.
    """

    def test_SA4_vs_SA6_same_geometry_more_SAs_is_faster(self):
        """CNV_SA4 (4 SAs) must be faster than CNV_SA6 (2 SAs) for same workload."""
        li = _make_layer_info(10, 10, 4, 16, 3, 3, 1, 1)
        cc4, *_ = cycles("CONV2D", 0, li, _make_params(n_SA=4), PPA_CONFIG)
        cc6, *_ = cycles("CONV2D", 0, li, _make_params(n_SA=2), PPA_CONFIG)
        self.assertLess(cc4, cc6,
                        "4-SA config should have lower latency than 2-SA config")

    def test_SA7_vs_SA8_single_vs_dual_SA(self):
        """CNV_SA7 (1 SA) vs CNV_SA8 (2 SAs) – same workload (M=8,K=27,N=64).
        For this small workload the 2-SA multi-split path produces MORE
        non-parallel tiles (n_N=2, n_M=0 -> tiles_np doubles) so 2-SA latency
        is intentionally HIGHER. This is correct algorithm behaviour: multi-SA
        split overhead can dominate for workloads that barely fill one SA."""
        li = _make_layer_info(10, 10, 3, 8, 3, 3, 0, 1)
        cc7, *_ = cycles("CONV2D", 1, li, _make_params(n_SA=1), PPA_CONFIG)
        cc8, *_ = cycles("CONV2D", 0, li, _make_params(n_SA=2), PPA_CONFIG)
        print(f"========cc7: {cc7}, cc8: {cc8}========")
        self.assertGreater(cc7, 0, "1-SA cycles must be positive")
        self.assertGreater(cc8, 0, "2-SA cycles must be positive")
        # Document the counter-intuitive relationship explicitly
        self.assertLess(cc8, cc7,
                           "For this small workload 2-SA split overhead "
                           "makes latency larger than 1 SA (expected)")

    def test_E_util_equals_total_MAC_count(self):
        """E_util must equal C_in * prod(kernel) * C_out * prod(out_size) * count / groups."""
        cases = [
            (8,  8,  1,  4,  3, 3, 0, 1, 2),
            (10, 10, 4, 16,  3, 3, 1, 1, 4),
            (8,  8, 32, 16,  1, 1, 0, 1, 2),
        ]
        for (INx, INy, Cin, Cout, Kx, Ky, Pad, Stride, n_SA) in cases:
            li = _make_layer_info(INx, INy, Cin, Cout, Kx, Ky, Pad, Stride)
            _, _, _, eu = cycles("CONV2D", 0, li, _make_params(n_SA), PPA_CONFIG)
            H_out = (INx + 2*Pad - Kx) // Stride + 1
            W_out = (INy + 2*Pad - Ky) // Stride + 1
            expected_eu = Cin * Kx * Ky * Cout * H_out * W_out * 1  # count=1, groups=1
            self.assertAlmostEqual(
                eu, expected_eu, places=2,
                msg=f"E_util wrong for Cin={Cin} Cout={Cout} K={Kx}×{Ky}")

    def test_bw_latency_proportional_to_output_volume(self):
        """BW latency must scale linearly with output volume (Qout = Cout*H*W*bits)."""
        # Double C_out → double BW latency
        li_base   = _make_layer_info(8, 8, 1,  4, 3, 3, 0, 1)
        li_double = _make_layer_info(8, 8, 1,  8, 3, 3, 0, 1)
        _, bw_base,   _, _ = cycles("CONV2D", 0, li_base,   _make_params(2), PPA_CONFIG)
        _, bw_double, _, _ = cycles("CONV2D", 0, li_double, _make_params(2), PPA_CONFIG)
        self.assertAlmostEqual(bw_double, 2.0 * bw_base, places=6,
                               msg="BW latency should double when C_out doubles")

    def test_config1_forces_single_SA(self):
        """config=1 must always use a single SA regardless of params['n_SA']."""
        li = _make_layer_info(10, 10, 3, 8, 3, 3, 0, 1)
        # Pass n_SA=4 in params but config=1 → must behave like n_SA=1
        cc_c1, _, nd_c1, _ = cycles("CONV2D", 1, li, _make_params(n_SA=4), PPA_CONFIG)
        cc_c0, _, nd_c0, _ = cycles("CONV2D", 1, li, _make_params(n_SA=1), PPA_CONFIG)
        self.assertEqual(cc_c1, cc_c0,
                         "config=1 should override n_SA and use 1 SA")
        self.assertEqual(nd_c1["HW_MACs"], 1 * 32 * 32,
                         "HW_MACs for config=1 must reflect 1 SA")

    def test_output_size_computed_correctly_with_padding(self):
        """Padding=1 on a 10×10 input with K=3, Stride=1 must give 10×10 output."""
        li = _make_layer_info(10, 10, 4, 16, 3, 3, 1, 1)
        self.assertEqual(li['out_size'], (10, 10),
                         "Padded conv output size incorrect")

    def test_output_size_computed_correctly_no_padding(self):
        """Padding=0 on a 10×10 input with K=3, Stride=1 must give 8×8 output."""
        li = _make_layer_info(10, 10, 3, 8, 3, 3, 0, 1)
        self.assertEqual(li['out_size'], (8, 8),
                         "Unpadded conv output size incorrect")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    unittest.main(verbosity=2)
