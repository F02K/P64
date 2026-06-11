import struct
import unittest

from p64.engine.math import Vec3
from p64.renderer.scene_renderer import RenderCamera, _mat4_bytes, _perspective_matrix, _project_point, _view_matrix, camera_basis


class RendererMathTests(unittest.TestCase):
    def test_mat4_bytes_transposes_row_major_for_opengl(self):
        row_major_translation = [
            1.0, 0.0, 0.0, 2.0,
            0.0, 1.0, 0.0, 3.0,
            0.0, 0.0, 1.0, 4.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        values = struct.unpack("16f", _mat4_bytes(row_major_translation))
        self.assertEqual(values[12:15], (2.0, 3.0, 4.0))

    def test_camera_negative_pitch_looks_down(self):
        forward, _right, _up = camera_basis(Vec3(-18.0, 0.0, 0.0))
        self.assertLess(forward.y, 0.0)
        self.assertLess(forward.z, 0.0)

    def test_project_point_maps_camera_target_to_screen_center(self):
        camera = RenderCamera(position=Vec3(0, 0, 5), rotation=Vec3())
        view = _view_matrix(camera)
        projection = _perspective_matrix(60, 1.0, 0.1, 100)
        projected = _project_point((0, 0, 0), view, projection, 800, 800)
        self.assertIsNotNone(projected)
        self.assertAlmostEqual(projected[0], 400, delta=1)
        self.assertAlmostEqual(projected[1], 400, delta=1)


if __name__ == "__main__":
    unittest.main()
