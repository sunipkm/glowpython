import glowpython
from glow834.fortran import cglow
from numpy import ndarray
import unittest


class TestExtension(unittest.TestCase):
    def setUp(self):
        self.jmax = 102

    def test_cglow_init(self):
        self.assertEqual(cglow.zz, None)
        glowpython.init_cglow(self.jmax)
        self.assertTrue(isinstance(cglow.zz, ndarray))
        self.assertTrue((cglow.zz == 0).all())
        self.assertRaises(RuntimeError, glowpython.init_cglow, self.jmax)

    def test_cglow_release(self):
        self.assertTrue(cglow.zz is not None)
        glowpython.fortran.cglow_release()
        self.assertTrue(cglow.zz is None)
        glowpython.reset_cglow(12)
        self.assertEqual(cglow.zz.size, 12)
        glowpython.reset_cglow(self.jmax)
        self.assertEqual(cglow.zz.size, self.jmax)

    def test_dataset(self):
        glowpython.reset_cglow(self.jmax)
        ds = glowpython.get_dataset()
        self.assertTrue((ds.E == cglow.ener).all())

    def test_data_dir(self):
        self.assertTrue(cglow.data_dir.tolist().endswith(b' '))
        # print(glow834.fortran.snoem(1, 1, 100))
