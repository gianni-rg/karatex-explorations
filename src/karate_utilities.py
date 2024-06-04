import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
import random
import numpy as np

from dtaidistance import dtw
from typing import Optional

random.seed(1234)

def load_json(filepath):
    json_file = {}
    with open(filepath, "r") as f:
        json_file = json.load(f)
    return json_file


def get_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)

    return (b, g, r)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Camera:
    def __init__(
        self,
        K: Optional[np.array] = None,
        Rot: Optional[np.array] = None,
        tvec: Optional[np.array] = None,
        rvec: Optional[np.array] = None,
        pos: Optional[np.array] = None,
        dist_coeffs: Optional[np.array] = None,
    ):
        self.K = K

        self.K_inv = np.linalg.inv(K) if K is not None else None
        self.Rot = Rot
        self.R_inv = np.linalg.inv(Rot) if Rot is not None else None
        self.pos = pos
        self.tvec = tvec
        self.rvec = rvec
        self.dist_coeffs = dist_coeffs

    def to_json_serializeable(self) -> str:
        return {
            "K": self.K.tolist(),
            "K_inv": self.K_inv.tolist(),
            "Rot": self.Rot.tolist(),
            "R_inv": self.R_inv.tolist(),
            "pos": self.pos.tolist(),
            "tvec": self.tvec.tolist(),
            "rvec": self.rvec.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
        }

    def from_json(self, json_camera: str) -> None:
        self.pos = np.array(json_camera["pos"], dtype=np.float64).reshape(
            3,
        )
        self.dist_coeffs = np.array(json_camera["dist_coeffs"], dtype=np.float64)
        self.K = np.array(json_camera["K"], dtype=np.float64).reshape(3, 3)
        self.K_inv = np.array(json_camera["K_inv"], dtype=np.float64).reshape(3, 3)
        self.Rot = np.array(json_camera["Rot"], dtype=np.float64).reshape(3, 3)
        self.R_inv = np.array(json_camera["R_inv"], dtype=np.float64).reshape(3, 3)
        self.tvec = np.array(json_camera["tvec"], dtype=np.float64).reshape(
            3,
        )
        self.rvec = np.array(json_camera["rvec"], dtype=np.float64).reshape(
            3,
        )


def factory_camera_from_k4a(
    resolution: list, principal_point: list, focal_length: list, dist_params: list
) -> Camera:
    K = np.array(
        [
            [focal_length[0], 0, principal_point[0]],
            [0, focal_length[1], principal_point[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64)
    r_vec = np.array([0, 0, 0], dtype=np.float64)
    t_vec = np.array([0, 0, 0], dtype=np.float64)
    pos = np.array([0, 0, 0], dtype=np.float64)
    dist_coeff = np.array(dist_params, dtype=np.float64)
    camera = Camera(K=K, Rot=R, tvec=t_vec, rvec=r_vec, pos=pos, dist_coeffs=dist_coeff)
    return camera


## Possible fix: https://math.stackexchange.com/questions/2738535/intersection-between-two-lines-3d
def RayRayIntersectionExDualUpdated(
    origin_ray1, dir_ray1, origin_ray2, dir_ray2, eps=1e-4
):
    # Vector3 line2Point1, Vector3 line2Point2, out Vector3 resultSegmentPoint1, out Vector3 resultSegmentPoint2)
    # Algorithm is ported from the C algorithm of
    # Paul Bourke at http://local.wasp.uwa.edu.au/~pbourke/geometry/lineline3d/
    resultSegmentPoint1 = np.zeros(3)
    resultSegmentPoint2 = np.zeros(3)
    k_dir = 10
    p1 = origin_ray1
    p2 = origin_ray1 + k_dir * dir_ray1
    p3 = origin_ray2
    p4 = origin_ray2 + k_dir * dir_ray2

    p13 = p1 - p3
    p43 = p4 - p3

    if np.linalg.norm(p43) < eps:
        return None, None, None

    p21 = p2 - p1
    if np.linalg.norm(p21) < eps:
        return None, None, None

    d1343 = p13[0] * p43[0] + p13[1] * p43[1] + p13[2] * p43[2]
    d4321 = p43[0] * p21[0] + p43[1] * p21[1] + p43[2] * p21[2]
    d1321 = p13[0] * p21[0] + p13[1] * p21[1] + p13[2] * p21[2]
    d4343 = p43[0] * p43[0] + p43[1] * p43[1] + p43[2] * p43[2]
    d2121 = p21[0] * p21[0] + p21[1] * p21[1] + p21[2] * p21[2]

    denom = d2121 * d4343 - d4321 * d4321
    if np.abs(denom) < eps:
        return None, None, None

    numer = d1343 * d4321 - d1321 * d4343

    mua = numer / denom
    mub = (d1343 + d4321 * (mua)) / d4343

    resultSegmentPoint1[0] = p1[0] + mua * p21[0]
    resultSegmentPoint1[1] = p1[1] + mua * p21[1]
    resultSegmentPoint1[2] = p1[2] + mua * p21[2]
    resultSegmentPoint2[0] = p3[0] + mub * p43[0]
    resultSegmentPoint2[1] = p3[1] + mub * p43[1]
    resultSegmentPoint2[2] = p3[2] + mub * p43[2]

    cross = np.cross(dir_ray1, dir_ray2)
    norm_cross = np.linalg.norm(cross)
    v1_norm = np.linalg.norm(dir_ray1)
    v2_norm = np.linalg.norm(dir_ray2)
    sin_theta = norm_cross / (v1_norm * v2_norm)

    return resultSegmentPoint1, resultSegmentPoint2, sin_theta


def GetPointCameraRayFromPixel(pixel, k_inv, rot_inv, tvec, dist_coeffs=None, cam=None):
    # fix for pixel distortion
    # https://groups.google.com/g/vsfm/c/IcbdIVv_Uek/m/Us32SBUNK9oJ?pli=1
    # https://newbedev.com/opencv-distort-back
    # https://programming.vip/docs/detailed-explanation-of-camera-model-and-de-distortion-method.html

    # this part checks the reprojection of the vector in
    # finding ray going from camera center to pixel coord
    hom_pt = np.array([pixel[0], pixel[1], 1], dtype="double")
    # hom_pt = np.array([721,391,1],dtype='double')

    # https://answers.opencv.org/question/148670/re-distorting-a-set-of-points-after-camera-calibration/ # <- this
    # TODO: apply distortion coefficients

    if (cam.dist_coeffs is not None) and (np.sum(cam.dist_coeffs) != 0) and False:
        # print("APPLYING DIST Coefficients")
        undistorted_pixel_coords = cv2.undistortPoints(pixel, cam.K, cam.dist_coeffs)
        undistorted_pixel_coords = undistorted_pixel_coords.reshape(
            2,
        )
        hom_pt = np.array(
            [undistorted_pixel_coords[0], undistorted_pixel_coords[1], 1],
            dtype="double",
        )

        # GeometryUtilities.distort_point(dir_cam_space[0],dir_cam_space[1],cam,hom_pt)
        # print(f"{x} {y} {dir_cam_space}")
        # dir_cam_space[0] = x
        # dir_cam_space[1] = y

    dir_cam_space = k_inv @ hom_pt  # this bit transform points in camera space
    dir_in_world = rot_inv @ dir_cam_space  # + bb
    return dir_in_world


def ProjectPointsCV(xyz, K, rvec, tvec, dist_coeffs):
    p_uvs, _ = cv2.projectPoints(xyz, rvec, tvec, K, dist_coeffs)
    return p_uvs


class PoseSimilarityScorer(object):

    def __init__(self, skeleton_keypoints=23, coordinates=3):
        # TODO: change default to the number of keypoints in the skeleton
        # TODO: change default to the coordinates 2D or 3D
        self.skeleton_keypoints = skeleton_keypoints
        self.coordinates = coordinates

    def percentage_score(self, score):
        # TODO: to be replaced with a better scoring algorithm, if found in the future
        percentage = 100 - (score * 100)
        return int(percentage)

    def dtwdis(self, input_points, reference_points, num_input_points, num_reference_points):
        input_points = input_points.reshape(self.coordinates * num_input_points)
        input_points = input_points / np.linalg.norm(input_points)
        reference_points = reference_points.reshape(self.coordinates * num_reference_points)
        reference_points = reference_points / np.linalg.norm(reference_points)
        return self.percentage_score(dtw.distance_fast(reference_points, input_points))

    def normalize(self, input_test):
        for keypoint in range(0, self.skeleton_keypoints):
            input_test[:, keypoint] = input_test[:, keypoint] / np.linalg.norm(input_test[:, keypoint])
        return input_test

    def compare(self, input_points, reference_points, input_frames_count, reference_frames_count):
        input_points = self.normalize(input_points)
        scores = []
        for keypoint in range(0, self.skeleton_keypoints):
            scores.append(self.dtwdis(input_points[:, keypoint], reference_points[:, keypoint], input_frames_count, reference_frames_count))
        return np.mean(scores), scores
