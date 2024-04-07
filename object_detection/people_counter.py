#!/usr/bin/env python3
"""
 Copyright (C) 2018-2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

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

def draw_icon(frame, detections, output_transform):
    frame = output_transform.resize(frame)
    overlay_image = cv2.imread(str(Path(__file__).resolve().parents[0] / 'AI.png'), -1)  # Load the PNG image with alpha channel

    for detection in detections:
        class_id = int(detection.id)
        if(class_id != 1):
            continue
        xmin, ymin, xmax, ymax = detection.get_coords()
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])

        # Calculate the position to overlay the image
        overlay_x = xmin + (xmax - xmin) // 2 - overlay_image.shape[1] // 2
        overlay_y = ymin + (ymax - ymin) // 4 - overlay_image.shape[0] // 2

        # Overlay the image
        overlay_with_alpha(frame, overlay_image, (overlay_x, overlay_y))


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

def draw_detections(frame, detections, palette, labels, output_transform):
    frame = output_transform.resize(frame)
    for detection in detections:
        class_id = int(detection.id)
        color = palette[class_id]
        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        xmin, ymin, xmax, ymax = detection.get_coords()
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                    (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
    return frame


def print_raw_results(detections, labels, frame_id):
    log.debug(' ------------------- Frame # {} ------------------ '.format(frame_id))
    log.debug(' Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ')
    for detection in detections:
        xmin, ymin, xmax, ymax = detection.get_coords()
        class_id = int(detection.id)
        det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
        log.debug('{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} '
                  .format(det_label, detection.score, xmin, ymin, xmax, ymax))


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
        # self.metrics = PerformanceMetrics()
        # self.render_metrics = PerformanceMetrics()

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

        while True:
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

                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
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

# if __name__ == '__main__':
#     sys.exit(main() or 0)
