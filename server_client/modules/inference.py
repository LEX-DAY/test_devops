import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
from pycuda import autoinit
import cv2
import logging

logger = logging.getLogger(__name__)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    
class EngineLoader:
    @staticmethod
    def load_engine(trt_runtime:trt.Runtime, engine_path:str) -> trt.ICudaEngine:
        """Load TensorRT engine from file"""
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

class Allocator:
    @staticmethod
    def allocate_buffers(engine: trt.ICudaEngine, max_batch_size:int) -> list:
        """Allocate buffers for TensorRT engine"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        # Get the number of bindings
        num_bindings = engine.num_io_tensors

        # Iterate over all bindings
        for i in range(num_bindings):
            # Get the tensor name for the current binding
            tensor_name = engine.get_tensor_name(i)

            # Get the shape and dtype of the tensor
            size = trt.volume(engine.get_tensor_shape(tensor_name)) * max_batch_size
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append to bindings list
            bindings.append(int(device_mem))

            # Check if the tensor is an input or output
            if  engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream
        
class Postprocessor:
    @staticmethod
    def postprocessing(outputs: np.ndarray,conf:float,iou:float) -> np.ndarray:
        """Postprocess TensorRT output"""
        boxes = outputs[0].host.reshape(84, -1).T  # Adjust dimensions as needed
        boxes = boxes[:, :5]
        boxes = boxes[boxes[:, 4] > conf, :] 
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        conf = boxes[:, 4]
        areas = w * h  # compute areas of boxes
        ordered = conf.argsort()[::-1]  # get sorted indexes of scores in descending order
        keep = []  # boxes to keep

        while ordered.size > 0:
            i = ordered[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[ordered[1:]])
            yy1 = np.maximum(y[i], y[ordered[1:]])
            xx2 = np.minimum(x[i] + w[i], x[ordered[1:]] + w[ordered[1:]])
            yy2 = np.minimum(y[i] + h[i], y[ordered[1:]] + h[ordered[1:]])
            width1 = np.maximum(0.0, xx2 - xx1 + 1)
            height1 = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = width1 * height1
            union = (areas[i] + areas[ordered[1:]] - intersection)
            iou = intersection / union
            indexes = np.where(iou <= iou)[0]
            ordered = ordered[indexes + 1]

        keep = np.array(keep)
        if len(keep) == 0:
            return ()
        boxes = boxes[keep]

        return boxes

class Model:
    """Class for TensorRT inference on Jetson Nano"""
    def __init__(self, engine_path: str, input_shape: list, conf=0.25, iou=0.45):
        """Class for loading .engine file with YOLOv8 and performing TensorRT inference on Jetson Nano
        Parameters
        ----------
        engine_path : str
            Path of engine file
        input_shape : tuple
            Model's input shape (width, height)
        conf : float
            Confidence YOLO threshold, by default 0.25
        iou : float
            IoU threshold for NMS algo, by default 0.45
        """
        self.engine_path = engine_path
        self.model_name = engine_path.split('/')[-1]
        self.dtype = np.float16
        self.conf = conf
        self.iou = iou
        self.input_shape = tuple(input_shape)
        
        # Initialize logger and runtime
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        
        # Load the TensorRT engine
        self.engine = self.__load_engine(self.runtime, self.engine_path)
        self.max_batch_size = 1
        
        # Allocate buffers and create execution context
        logger.info("Load engine done...")
        self.inputs, self.outputs, self.bindings, self.stream = self.__allocate_buffers(self.engine, self.max_batch_size)
        logger.info("Allocate buffers done...")
        self.context = self.engine.create_execution_context()
        logger.info("Create execution context done...")

    @staticmethod
    def __load_engine(trt_runtime, engine_path):
        return EngineLoader.load_engine(trt_runtime, engine_path)

    def __allocate_buffers(self,engine: trt.ICudaEngine, max_batch_size:int) -> list:
        """Allocate memory for inputs and outputs"""
        return Allocator.allocate_buffers(engine, max_batch_size)

    def __nms(self, outputs: np.ndarray, conf: float, iou: float) -> np.ndarray:
        """Perform postprocess and NMS on output boxes"""
        return Postprocessor.postprocessing(outputs,conf,iou)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Perform object detection with YOLOv8 on given frame"""
        ratio = (x.shape[1] / self.input_shape[0], x.shape[0] / self.input_shape[1])
        ratio = np.array(ratio)
        x = cv2.resize(x, self.input_shape)
        x = x[:, :, ::-1].transpose(2, 0, 1)[None, ...] / 255.0
        x = x.astype(self.dtype)
        np.copyto(self.inputs[0].host, x.ravel())

        # Transfer data to CUDA device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        # Execute inference
        self.context.execute_v2(self.bindings)

        # Retrieve results from CUDA device
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
        self.stream.synchronize()

        boxes = self.__nms(self.outputs,self.conf,self.iou)  # Apply NMS
        if len(boxes) > 0:
            boxes[:, :2] = boxes[:, :2] * ratio  # Scale boxes
            boxes[:, 2:4] = boxes[:, 2:4] * ratio
        return boxes

    def release(self) -> None:
        """Free allocated resources"""
        del self.outputs
        del self.inputs
        del self.stream
        del self.bindings

    def __del__(self):
        """Destructor to free allocated resources"""
        # Free device memory
        for inp in self.inputs:
            inp.device.free()
        for out in self.outputs:
            out.device.free()

        # Destroy the CUDA stream
        del self.stream

        # Release the execution context and engine
        del self.context
        del self.engine

def draw_bboxes(frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Draw red bounding boxes on given frame"""
    if len(boxes) > 0:
        boxes[:, :2] = boxes[:, :2] - boxes[:, 2:4] / 2
        boxes[:, 2:4] = boxes[:, :2] + boxes[:, 2:4]
        for box in boxes:
            frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
            frame = cv2.putText(frame, 'PERSON {:.2f}'.format(box[4]), (int(box[0]), int(box[1])),
                                cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return frame