import unittest

from p64.editor.utils.math import _add_vec3, _lerp_vec3, _normalize_vec3, _scale_vec3, _sub_vec3, _vec3_length
from p64.engine.math import Quaternion, Vec3, basis_from_rotation, clamp, cross, dot, forward_from_yaw, length, lerp, lerp_vec3, normalize


class EngineMathTests(unittest.TestCase):
    def assertVec3AlmostEqual(self, actual: Vec3, expected: Vec3, places: int = 6) -> None:
        self.assertAlmostEqual(actual.x, expected.x, places=places)
        self.assertAlmostEqual(actual.y, expected.y, places=places)
        self.assertAlmostEqual(actual.z, expected.z, places=places)

    def test_vec3_add_subtract_and_negate_return_new_vectors(self):
        a = Vec3(1, 2, 3)
        b = Vec3(4, 5, 6)

        self.assertEqual(a + b, Vec3(5, 7, 9))
        self.assertEqual(b - a, Vec3(3, 3, 3))
        self.assertEqual(-a, Vec3(-1, -2, -3))
        self.assertEqual(a, Vec3(1, 2, 3))
        self.assertEqual(b, Vec3(4, 5, 6))

    def test_vec3_scalar_multiply_and_divide(self):
        vector = Vec3(1, 2, 3)

        self.assertEqual(vector * 2, Vec3(2, 4, 6))
        self.assertEqual(2 * vector, Vec3(2, 4, 6))
        self.assertEqual(vector / 2, Vec3(0.5, 1.0, 1.5))
        with self.assertRaises(ZeroDivisionError):
            _ = vector / 0
        with self.assertRaises(TypeError):
            _ = vector * Vec3(1, 1, 1)

    def test_vec3_length_normalized_dot_cross_and_lerp(self):
        vector = Vec3(3, 4, 0)

        self.assertEqual(vector.length_squared(), 25)
        self.assertEqual(vector.length(), 5)
        self.assertVec3AlmostEqual(vector.normalized(), Vec3(0.6, 0.8, 0.0))
        self.assertEqual(Vec3().normalized(), Vec3())
        self.assertEqual(vector.dot(Vec3(2, 0, 1)), 6)
        self.assertEqual(Vec3(1, 0, 0).cross(Vec3(0, 1, 0)), Vec3(0, 0, 1))
        self.assertEqual(Vec3(0, 0, 0).lerp(Vec3(10, 20, 30), 0.25), Vec3(2.5, 5.0, 7.5))

    def test_vec3_factories_and_copy(self):
        original = Vec3(1, 2, 3)
        copied = original.copy()

        self.assertEqual(Vec3.zero(), Vec3())
        self.assertEqual(Vec3.one(), Vec3(1.0, 1.0, 1.0))
        self.assertEqual(Vec3.up(), Vec3(0.0, 1.0, 0.0))
        self.assertEqual(Vec3.forward(), Vec3(0.0, 0.0, -1.0))
        self.assertEqual(Vec3.right(), Vec3(1.0, 0.0, 0.0))
        self.assertEqual(copied, original)
        self.assertIsNot(copied, original)

    def test_module_helpers(self):
        self.assertEqual(clamp(12.0, 0.0, 10.0), 10.0)
        self.assertEqual(clamp(-1.0, 0.0, 10.0), 0.0)
        self.assertEqual(lerp(10.0, 20.0, 0.25), 12.5)
        self.assertEqual(dot(Vec3(1, 2, 3), Vec3(4, 5, 6)), 32)
        self.assertEqual(cross(Vec3(1, 0, 0), Vec3(0, 1, 0)), Vec3(0, 0, 1))
        self.assertEqual(length(Vec3(0, 3, 4)), 5)
        self.assertVec3AlmostEqual(normalize(Vec3(0, 3, 4)), Vec3(0, 0.6, 0.8))
        self.assertEqual(lerp_vec3(Vec3(), Vec3(10, 20, 30), 0.5), Vec3(5, 10, 15))

    def test_direction_helpers(self):
        self.assertVec3AlmostEqual(forward_from_yaw(0), Vec3(0.0, 0.0, -1.0))
        self.assertVec3AlmostEqual(forward_from_yaw(90), Vec3(1.0, 0.0, 0.0))

        forward, right, up = basis_from_rotation(Vec3())
        self.assertVec3AlmostEqual(forward, Vec3(0.0, 0.0, -1.0))
        self.assertVec3AlmostEqual(right, Vec3(1.0, 0.0, 0.0))
        self.assertVec3AlmostEqual(up, Vec3(0.0, 1.0, 0.0))

    def test_editor_vector_helpers_delegate_to_engine_math(self):
        self.assertEqual(_add_vec3(Vec3(1, 2, 3), Vec3(4, 5, 6)), Vec3(5, 7, 9))
        self.assertEqual(_sub_vec3(Vec3(4, 5, 6), Vec3(1, 2, 3)), Vec3(3, 3, 3))
        self.assertEqual(_scale_vec3(Vec3(1, 2, 3), 2), Vec3(2, 4, 6))
        self.assertEqual(_vec3_length(Vec3(0, 3, 4)), 5)
        self.assertVec3AlmostEqual(_normalize_vec3(Vec3(0, 3, 4)), Vec3(0, 0.6, 0.8))
        self.assertEqual(_lerp_vec3(Vec3(), Vec3(10, 20, 30), 0.5), Vec3(5, 10, 15))

    def test_quaternion_euler_inverse_and_vector_rotation(self):
        quaternion = Quaternion.from_euler(Vec3(20.0, 35.0, -15.0))
        euler = quaternion.to_euler()
        restored = Quaternion.from_euler(euler)
        identity = quaternion * quaternion.inverse()
        forward = quaternion * Vec3.forward()

        self.assertAlmostEqual(abs(quaternion.x * restored.x + quaternion.y * restored.y + quaternion.z * restored.z + quaternion.w * restored.w), 1.0, places=6)
        self.assertAlmostEqual(identity.x, 0.0, places=6)
        self.assertAlmostEqual(identity.y, 0.0, places=6)
        self.assertAlmostEqual(identity.z, 0.0, places=6)
        self.assertAlmostEqual(identity.w, 1.0, places=6)
        self.assertAlmostEqual(forward.length(), 1.0, places=6)

    def test_quaternion_angle_axis_look_rotation_and_slerp(self):
        yaw = Quaternion.angle_axis(90.0, Vec3.up())
        halfway = Quaternion.slerp(Quaternion.identity(), yaw, 0.5)
        looked = Quaternion.look_rotation(Vec3.forward())

        self.assertVec3AlmostEqual(yaw * Vec3.forward(), Vec3(-1.0, 0.0, 0.0))
        self.assertVec3AlmostEqual(looked * Vec3.forward(), Vec3.forward())
        rotated = halfway * Vec3.forward()
        self.assertAlmostEqual(rotated.x, -0.707106, places=5)
        self.assertAlmostEqual(rotated.z, -0.707106, places=5)


if __name__ == "__main__":
    unittest.main()
