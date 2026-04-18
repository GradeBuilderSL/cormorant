"""Tests for ap_fixed<16,8> weight encoding (TestTensorEncoding)."""

import os
import sys
import unittest

from helpers import _model, _models_exist


class TestTensorEncoding(unittest.TestCase):
    """ap_fixed<16,8> weight encoding."""

    def test_zero(self):
        import numpy as np
        from src.tensor import _float_to_apfixed_hex
        self.assertEqual(_float_to_apfixed_hex(np.array([0.0])), ["0x0000"])

    def test_one(self):
        import numpy as np
        from src.tensor import _float_to_apfixed_hex
        # 1.0 * 256 = 256 = 0x0100
        self.assertEqual(_float_to_apfixed_hex(np.array([1.0])), ["0x0100"])

    def test_minus_one(self):
        import numpy as np
        from src.tensor import _float_to_apfixed_hex
        # -1.0 * 256 = -256 = 0xFF00 as uint16
        self.assertEqual(_float_to_apfixed_hex(np.array([-1.0])), ["0xFF00"])

    def test_half(self):
        import numpy as np
        from src.tensor import _float_to_apfixed_hex
        # 0.5 * 256 = 128 = 0x0080
        self.assertEqual(_float_to_apfixed_hex(np.array([0.5])), ["0x0080"])

    def test_saturation_positive(self):
        import numpy as np
        from src.tensor import _float_to_apfixed_hex, _AP_FIXED_MAX
        hexvals  = _float_to_apfixed_hex(np.array([999.0]))
        expected = _float_to_apfixed_hex(np.array([_AP_FIXED_MAX]))
        self.assertEqual(hexvals, expected)

    def test_saturation_negative(self):
        import numpy as np
        from src.tensor import _float_to_apfixed_hex, _AP_FIXED_MIN
        hexvals  = _float_to_apfixed_hex(np.array([-999.0]))
        expected = _float_to_apfixed_hex(np.array([_AP_FIXED_MIN]))
        self.assertEqual(hexvals, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
