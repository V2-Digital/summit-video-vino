import logging as log
import sys
from pathlib import Path
from time import perf_counter

import cv2
from queue import Queue
import threading

sys.path.append(str(Path(__file__).resolve().parents[1] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[1] / 'common/python/model_zoo'))

from model_api.models import DetectionModel, DetectionWithLandmarks, RESIZE_TYPES, OutputTransform
from model_api.performance_metrics import PerformanceMetrics
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.adapters import create_core, OpenvinoAdapter, OVMSAdapter

import monitors
from images_capture import open_images_capture
from helpers import resolution, log_latency_per_stage
from visualizers import ColorPalette

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

overlay_image = cv2.imread(str(Path(__file__).resolve().parents[0] / 'AI.png'), -1)
overlay_scale_factor = 0.5
def draw_icon(frame, detections, output_transform):
    frame = output_transform.resize(frame)

    for detection in detections:
        class_id = int(detection.id)
        if class_id != 1:
            continue
        xmin, ymin, xmax, ymax = detection.get_coords()
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])

        # Calculate the size of the bounding box
        box_width = xmax - xmin
        box_height = ymax - ymin

        # Scale the overlay image based on the bounding box size and scale_factor
        scaled_overlay_width = int(box_width * overlay_scale_factor)
        scaled_overlay_height = int(overlay_image.shape[0] * (scaled_overlay_width / overlay_image.shape[1]))
        scaled_overlay_image = cv2.resize(overlay_image, (scaled_overlay_width, scaled_overlay_height))

        # Calculate the position to overlay the scaled image
        overlay_x = xmin + (box_width - scaled_overlay_width) // 2
        overlay_y = ymin + (box_height - scaled_overlay_height) // 4

        # Overlay the scaled image
        overlay_with_alpha(frame, scaled_overlay_image, (overlay_x, ymin))

    return frame

def overlay_with_alpha(background, overlay, position):
    x, y = position
    overlay_height, overlay_width = overlay.shape[:2]

    # Calculate the region of the background where the overlay will be placed
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + overlay_width, background.shape[1])
    y2 = min(y + overlay_height, background.shape[0])

    # Calculate the region of the overlay that will be used
    overlay_x1 = max(-x, 0)
    overlay_y1 = max(-y, 0)
    overlay_x2 = overlay_width - max(x + overlay_width - background.shape[1], 0)
    overlay_y2 = overlay_height - max(y + overlay_height - background.shape[0], 0)

    # Extract the alpha mask of the RGBA overlay, convert to RGB
    alpha_mask = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3] / 255.0
    overlay_bgr = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, :3]

    # Take the region of the background where the overlay will be placed
    background_region = background[y1:y2, x1:x2]

    # Apply the alpha mask to each channel of the overlay
    for c in range(0, 3):
        background_region[:, :, c] = (alpha_mask * overlay_bgr[:, :, c] +
                                      (1 - alpha_mask) * background_region[:, :, c])

    # Replace the region in the original background with the modified region
    background[y1:y2, x1:x2] = background_region

class PeopleCounter:
    def __init__(self, args):
        self.args = args
        args = self.args
        self.cap = open_images_capture(args.input, args.loop)

        if args.adapter == 'openvino':
            plugin_config = get_user_config(args.device, args.num_streams, args.num_threads)
            self.model_adapter = OpenvinoAdapter(create_core(), args.model, device=args.device, plugin_config=plugin_config,
                                            max_num_requests=args.num_infer_requests, model_parameters = {'input_layouts': args.layout})
        elif args.adapter == 'ovms':
            self.model_adapter = OVMSAdapter(args.model)

        configuration = {
            'resize_type': args.resize_type,
            'mean_values': args.mean_values,
            'scale_values': args.scale_values,
            'reverse_input_channels': args.reverse_input_channels,
            'path_to_labels': args.labels,
            'confidence_threshold': args.prob_threshold,
            'input_size': args.input_size, # The CTPN specific
            'num_classes': args.num_classes, # The NanoDet and NanoDetPlus specific
        }
        self.model = DetectionModel.create_model(args.architecture_type, self.model_adapter, configuration)
        self.model.log_layers_info()

        self.detector_pipeline = AsyncPipeline(self.model)
        self.palette = ColorPalette(len(self.model.labels) if self.model.labels else 100)

        self.frame_queue = Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.count)
        self.thread.start()

    def get_latest_frame(self):
        try:
            return self.frame_queue.get(timeout=0.1)  # Adjust the timeout as needed
        except queue.Empty:
            return None
    
    def stop(self):
        self.stop_event.set()
        self.thread.join()

    def count(self):

        next_frame_id = 0
        next_frame_id_to_show = 0
        args = self.args
        model = self.model
        detector_pipeline = self.detector_pipeline
        # palette = self.palette
        # metrics = self.metrics
        # render_metrics = self.render_metrics
        presenter = None
        output_transform = None
        cap = self.cap
        model_adapter = self.model_adapter

        configuration = {
            'resize_type': args.resize_type,
            'mean_values': args.mean_values,
            'scale_values': args.scale_values,
            'reverse_input_channels': args.reverse_input_channels,
            'path_to_labels': args.labels,
            'confidence_threshold': args.prob_threshold,
            'input_size': args.input_size, # The CTPN specific
            'num_classes': args.num_classes, # The NanoDet and NanoDetPlus specific
        }
        model = DetectionModel.create_model(args.architecture_type, model_adapter, configuration)
        model.log_layers_info()

        detector_pipeline = AsyncPipeline(model)

        next_frame_id = 0
        next_frame_id_to_show = 0
        # palette = ColorPalette(len(model.labels) if model.labels else 100)
        # metrics = PerformanceMetrics()
        # render_metrics = PerformanceMetrics()
        presenter = None
        output_transform = None

        while not self.stop_event.is_set():
            if detector_pipeline.callback_exceptions:
                raise detector_pipeline.callback_exceptions[0]
            # Process all completed requests
            results = detector_pipeline.get_result(next_frame_id_to_show)
            if results:
                objects, frame_meta = results
                frame = frame_meta['frame']
                start_time = frame_meta['start_time']

                # if len(objects) and args.raw_output_message:
                    # print_raw_results(objects, model.labels, next_frame_id_to_show)

                presenter.drawGraphs(frame)
                # rendering_start_time = perf_counter()
                frame = draw_icon(frame, objects, output_transform)
                # render_metrics.update(rendering_start_time)
                # metrics.update(start_time, frame)
                (flag, encodedImage) = cv2.imencode(".jpg", frame) 

                next_frame_id_to_show += 1

                if not self.frame_queue.full():
                    self.frame_queue.put(bytearray(encodedImage))
                    log.info('Frame #{} processed'.format(next_frame_id_to_show))
                else:
                    log.warning('Frame queue is full')
                continue

            if detector_pipeline.is_ready():
                # Get new image/frame   
                start_time = perf_counter()
                frame = cap.read()
                if frame is None:
                    if next_frame_id == 0:
                        raise ValueError("Can't read an image from the input")
                    break
                if next_frame_id == 0:
                    output_transform = OutputTransform(frame.shape[:2], args.output_resolution)
                    if args.output_resolution:
                        output_resolution = output_transform.new_resolution
                    else:
                        output_resolution = (frame.shape[1], frame.shape[0])
                    presenter = monitors.Presenter(args.utilization_monitors, 55,
                                                (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
                # Submit for inference
                detector_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
                next_frame_id += 1
            else:
                # Wait for empty request
                detector_pipeline.await_any()

        detector_pipeline.await_all()
        if detector_pipeline.callback_exceptions:
            raise detector_pipeline.callback_exceptions[0]