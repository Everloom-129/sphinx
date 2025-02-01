# import threading
# from abc import abstractmethod, abstractproperty
# from typing import Dict, Optional, Union

import numpy as np
import cv2
import pyzed.sl as sl
import multiprocessing as mp


"""
NOTE: All cameras are set to record at the specified resolution using the ZED SDK.
"""


class ZEDCamera:
    def __init__(
        self,
        serial_number: str,
        width: int,
        height: int,
        use_depth: bool,
        exposure=-1,
    ):
        self.width = width
        self.height = height
        self.use_depth = use_depth
        self.serial_number = serial_number
        self.exposure = exposure

        # Initialize ZED camera
        self.zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720  # You can adjust this
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE if use_depth else sl.DEPTH_MODE.NONE
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.set_from_serial_number(int(serial_number))

        # Open the camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

        # Set camera settings
        if exposure > 0:
            self.zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, exposure)

        # Prepare runtime parameters
        self.runtime_params = sl.RuntimeParameters()
        if self.use_depth:
            self.runtime_params.confidence_threshold = 50
            self.runtime_params.texture_confidence_threshold = 100

        # Create image objects
        self.image = sl.Mat()
        self.depth = sl.Mat() if use_depth else None

        # Warm up camera
        for _ in range(2):
            self.zed.grab(self.runtime_params)

    def get_intrinsics(self):
        calibration_params = self.zed.get_camera_information().calibration_parameters
        left_cam = calibration_params.left_cam
        return dict(
            matrix=np.array([
                [left_cam.fx, 0, left_cam.cx],
                [0, left_cam.fy, left_cam.cy],
                [0, 0, 1.0],
            ]),
            width=self.width,
            height=self.height,
            depth_scale=1.0,  # ZED depth is in millimeters by default
        )

    def get_frames(self) -> dict[str, np.ndarray]:
        if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to grab frame from ZED camera")

        # Get color image
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        image = self.image.get_data()
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frames = dict(image=image)

        # Get depth if enabled
        if self.use_depth:
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            depth = self.depth.get_data()
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, axis=-1)
            frames["depth"] = depth

        return frames

    def close(self):
        self.zed.close()


class SequentialCameras:
    def __init__(self, camera_args_list: list[dict]):
        self.name_to_camera_args = {}
        self.cameras: dict[str, ZEDCamera] = {}
        for camera_args in camera_args_list:
            name = camera_args.pop("name")
            self.name_to_camera_args[name] = camera_args
            self.cameras[name] = ZEDCamera(**camera_args)

    def get_intrinsics(self, name):
        return self.cameras[name].get_intrinsics()

    def get_frames(self):
        frames = {}
        for name, camera in self.cameras.items():
            frames[name] = camera.get_frames()
        return frames

    def __del__(self):
        # FIXME: well
        for _, camera in self.cameras.items():
            camera.close()


class ParallelCameras:
    def __init__(self, camera_args_list: list[dict]):
        # self.camera_args_list = camera_args_list
        self.name_to_camera_args = {}
        for camera_args in camera_args_list:
            name = camera_args.pop("name")
            self.name_to_camera_args[name] = camera_args

        self.camera_procs = {}
        self.put_queues = {}
        self.get_queues = {}
        self.intrinsics = {}

        for name, camera_args in self.name_to_camera_args.items():
            put_queue = mp.Queue(maxsize=1)
            get_queue = mp.Queue(maxsize=1)
            proc = mp.Process(target=self._camera_proc, args=(camera_args, put_queue, get_queue))
            proc.start()
            self.camera_procs[name] = proc

            self.intrinsics[name] = get_queue.get()
            print(f"cam {name} constructed")

            self.put_queues[name] = put_queue
            self.get_queues[name] = get_queue

    def _camera_proc(self, camera_args, receive_queue: mp.Queue, send_queue: mp.Queue):
        camera = ZEDCamera(**camera_args)
        send_queue.put(camera.get_intrinsics())

        while True:
            msg = receive_queue.get()
            if msg == "terminate":
                break

            assert msg == "get"
            assert send_queue.empty()

            frames = camera.get_frames()
            send_queue.put(frames)

        camera.close()

    def get_intrinsics(self, name):
        return self.intrinsics[name]

    def get_frames(self):
        for _, put_queue in self.put_queues.items():
            assert put_queue.empty()
            put_queue.put("get")

        camera_frames = {}
        for name, get_queue in self.get_queues.items():
            camera_frames[name] = get_queue.get()

        return camera_frames

    def __del__(self):
        # FIXME: well
        for name, put_queue in self.put_queues.items():
            print(f"terminating {name}")
            put_queue.put("terminate")
            self.camera_procs[name].join()


if __name__ == "__main__":
    from envs.franka_env_config import FrankaEnvConfig
    import pyrallis
    from common_utils import FreqGuard, Stopwatch

    cfg = pyrallis.parse(config_class=FrankaEnvConfig)  # type: ignore
    # cfg.show_camera = 1
      
    if cfg.parallel_camera:
        camera = ParallelCameras(cfg.cameras)
    else:
        camera = SequentialCameras(cfg.cameras)

    frames = camera.get_frames()
    print(frames)
