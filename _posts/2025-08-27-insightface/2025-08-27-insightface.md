---
layout: post
title: "Insightface"
date: 2025-08-27 16:14:00 +0800
categories: [Deploy, Vision]
---

# InsightFace: 2D and 3D Face Analysis Project

code: [https://github.com/deepinsight/insightface](https://github.com/deepinsight/insightface)

## TensorRT

在Jetson AGX Orin上，onnx模型转tensorRT

Jetpack自带tensorRT

添加 `trtexec` 到路径
``` shell
export PATH=/usr/src/tensorrt/bin:$PATH
```

转模型：
``` shell
trtexec --onnx=model.onnx --saveEngine=model.plan
```

显式指定Batch大小
``` shell
trtexec --onnx=det_2.5g.onnx --saveEngine=det_2.5g.plan --explicitBatch --minShapes=input.1:1x3x640x640 --optShapes=input.1:1x3x640x640 --maxShapes=input.1:1x3x640x640
```

检查模型输入输出脚本：
<details markdown="1">
  <summary>inspect.py</summary>

```python
import argparse
import os
import sys
import glob
import tensorrt as trt

parser = argparse.ArgumentParser(description="Inspect TensorRT engine IO tensors")
parser.add_argument(
    "--model",
    type=str,
    default=None,
    help="Path to the .plan file; if not specified, all .plan files in current directory will be used"
)
args = parser.parse_args()

def inspect_engine(path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    print(f"\n=== Inspecting: {path} ===")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = engine.get_tensor_dtype(name)
        mode = engine.get_tensor_mode(name)
        io_type = "Input" if mode == trt.TensorIOMode.INPUT else "Output"
        print(f"{io_type} - {name}: shape={shape}, dtype={dtype}")

if args.model:
    # 单文件模式
    if not os.path.isfile(args.model):
        print(f"[错误] 找不到模型文件: {args.model}")
        sys.exit(1)
    inspect_engine(args.model)
else:
    # 遍历当前路径下所有 .plan 文件
    plan_files = glob.glob("*.plan")
    if not plan_files:
        print("[提示] 当前目录下未找到任何 .plan 文件")
        sys.exit(0)
    for pf in plan_files:
        inspect_engine(pf)
```

</details>

## Conda tensorRT

Jetson 上 tensorRT 只能通过软链接系统包实现, 具体版本自行更换

``` shell
conda install mamba -c conda-forge
mamba env create -n trt python=3.8
cd ~/miniconda3/envs/trt/lib/python3.8/site-packages
ln -s /usr/lib/python3.8/dist-packages/tensorrt tensorrt
ln -s /usr/lib/python3.8/dist-packages/tensorrt-8.5.2.2.dist-info  tensorrt-8.5.2.2.dist-info
ln -s /usr/lib/python3.8/dist-packages/onnx_graphsurgeon onnx_graphsurgeon
ln -s /usr/lib/python3.8/dist-packages/onnx_graphsurgeon-0.3.12.dist-info  onnx_graphsurgeon-0.3.12.dist-info
ln -s /usr/lib/python3.8/dist-packages/uff uff
ln -s /usr/lib/python3.8/dist-packages/uff-0.6.9.dist-info uff-0.6.9.dist-info
mamba activate trt
pip install opencv-python
pip install scikit-image
sed -i 's/bool: np.bool/bool: bool/g' ~/miniconda3/envs/trt/lib/python3.8/site-packages/tensorrt/__init__.py
```

测试
``` shell
python -c "import tensorrt as trt; print(trt.__version__)"
```

## 模型

模型有5个，详见[链接](https://github.com/deepinsight/insightface/tree/master/python-package)：
- `det_2.5g`: 人脸检测
- `1k3d68`: 3D 68点 landmark
- `2d106det`: 2D 106点 landmark
- `genderage`: 性别年龄
- `w600k_r50`: 特征

### 基本框架
<details markdown="1">
  <summary>common.py</summary>

```python
import numpy as np
from numpy.linalg import norm as l2norm
from loguru import logger
import tensorrt as trt
import pycuda.driver as cuda
import os
import cv2
import pycuda.autoinit  # initializes CUDA context

def affine_crop(img, bbox, out_size):
    x1, y1, x2, y2 = bbox[:4]
    w, h = (x2 - x1), (y2 - y1)
    center = ((x2 + x1) / 2.0, (y2 + y1) / 2.0)
    scale = out_size / (max(w, h) * 1.5)
    M = cv2.getRotationMatrix2D(center, 0, scale)
    M[0, 2] += (out_size / 2.0 - center[0])
    M[1, 2] += (out_size / 2.0 - center[1])
    aimg = cv2.warpAffine(img, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return aimg, M

class Face(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        #for k in self.__class__.__dict__.keys():
        #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #        setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                    if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property 
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property 
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender==1 else 'F'

# ----------------------------
# Generic TensorRT runner
# ----------------------------
class TrtRunner:
    def __init__(self, engine_path: str):
        assert os.path.exists(engine_path), f"Missing engine: {engine_path}"
        logger.info(f"Loading TRT engine: {engine_path}")
        trt_logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        runtime = trt.Runtime(trt_logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        logger.info(f"Engine created: {engine_path}")
        # Bindings
        self.input_indices = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
        self.output_indices = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
        logger.debug(f"Bindings total={self.engine.num_bindings}, inputs={self.input_indices}, outputs={self.output_indices}")
        # Allocated buffers
        self.bindings = [None] * self.engine.num_bindings
        self._allocated = False

    def _nptype(self, dt):
        return trt.nptype(dt)

    def allocate(self, input_shape: tuple):
        # Set dynamic shape (if needed) and allocate IO buffers
        inp_idx = self.input_indices[0]
        self.context.set_binding_shape(inp_idx, input_shape)
        logger.debug(f"Allocating buffers for input shape={input_shape}")
        # Input
        in_dtype = self._nptype(self.engine.get_binding_dtype(inp_idx))
        in_size = int(np.prod(input_shape))
        self.in_host = cuda.pagelocked_empty(in_size, in_dtype)
        self.in_dev = cuda.mem_alloc(self.in_host.nbytes)
        self.bindings[inp_idx] = int(self.in_dev)
        logger.debug(f"Input dtype={in_dtype}, size={in_size}, bytes={self.in_host.nbytes}")
        # Outputs
        self.out_hosts, self.out_devs, self.out_shapes = [], [], []
        for oi in self.output_indices:
            oshape = tuple(self.context.get_binding_shape(oi))
            odtype = self._nptype(self.engine.get_binding_dtype(oi))
            osize = int(np.prod(oshape))
            o_host = cuda.pagelocked_empty(osize, odtype)
            o_dev = cuda.mem_alloc(o_host.nbytes)
            self.out_hosts.append(o_host)
            self.out_devs.append(o_dev)
            self.out_shapes.append(oshape)
            self.bindings[oi] = int(o_dev)
            logger.debug(f"Output[{oi}] shape={oshape}, dtype={odtype}, bytes={o_host.nbytes}")
        self._allocated = True

    def infer(self, input_array: np.ndarray):
        assert self._allocated, "Call allocate(input_shape) before infer()"
        assert tuple(input_array.shape) == tuple(self.context.get_binding_shape(self.input_indices[0]))
        logger.debug("Starting TRT inference")
        # HtoD input
        np.copyto(self.in_host, input_array.ravel())
        cuda.memcpy_htod_async(self.in_dev, self.in_host, self.stream)
        # Execute
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        # DtoH outputs
        for o_host, o_dev in zip(self.out_hosts, self.out_devs):
            cuda.memcpy_dtoh_async(o_host, o_dev, self.stream)
        self.stream.synchronize()
        logger.debug("Finished TRT inference")
        # Wrap outputs
        return [np.array(o_host).reshape(shape) for o_host, shape in zip(self.out_hosts, self.out_shapes)]
```

</details>

### 人脸检测
<details markdown="1">
  <summary>retinaface.py</summary>

```python
from common import TrtRunner
from loguru import logger
import numpy as np
import cv2
import time
import pycuda.autoinit  # initializes CUDA context

# ----------------------------
# RetinaFace helpers (pre/post)
# ----------------------------
def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i%2] + distance[:, i]
        py = points[:, i%2+1] + distance[:, i+1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)

def nms_det(dets, thresh=0.4):
    if dets.size == 0:
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

class RetinaFaceTRT:
    def __init__(self, engine_path, det_thresh=0.5, nms_thresh=0.4, input_size=(640, 640)):
        self.runner = TrtRunner(engine_path)
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        self.input_size = tuple(input_size)  # (w, h)
        self.input_mean = 127.5
        self.input_std = 128.0
        dummy = (1, 3, self.input_size[1], self.input_size[0])
        self.runner.allocate(dummy)
        self.outputs_count = len(self.runner.out_shapes)
        logger.debug(f"TRT outputs_count={self.outputs_count}")
        # default assumptions
        if self.outputs_count in (6, 9):
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
        elif self.outputs_count in (10, 15):
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
        else:
            self.fmc = None
            self._feat_stride_fpn = []
        # analyze bindings to build a robust map
        # self._head_map = self._analyze_outputs(self.runner.out_shapes, self.input_size)
        # self.use_kps = any('kps' in heads for heads in self._head_map.values())
        # logger.info(f"Head map strides={sorted(self._head_map.keys())}, use_kps={self.use_kps}")
        self.center_cache = {}

        logger.debug(f"fmc={self.fmc}")
        logger.debug(f"_feat_stride_fpn={self._feat_stride_fpn}")

    def _analyze_outputs(self, out_shapes, input_size):
        # Returns: { stride: { 'scores': meta, 'bbox': meta, 'kps': meta? } }
        # meta = { 'index': int, 'layout': 'NCHW'|'NHWC', 'A': int, 'dims': int, 'H': int, 'W': int }
        input_w, input_h = input_size
        valid_strides = {8, 16, 32, 64, 128}
        head_map = {}
        for idx, shp in enumerate(out_shapes):
            # accept rank-4 only
            if len(shp) != 4:
                continue
            n, d1, d2, d3 = shp
            # try NCHW
            cand = []
            for layout in ('NCHW', 'NHWC'):
                if layout == 'NCHW':
                    C, H, W = d1, d2, d3
                else:
                    H, W, C = d1, d2, d3
                if H <= 0 or W <= 0:
                    continue
                if (input_h % H) != 0 or (input_w % W) != 0:
                    continue
                stride_h = input_h // H
                stride_w = input_w // W
                if stride_h != stride_w or stride_h not in valid_strides:
                    continue
                stride = stride_h
                # classify by channel count
                role = None
                A = 1
                dims = None
                if C % 4 == 0:
                    role = 'bbox'
                    dims = 4
                    A = C // 4
                if C % 10 == 0:
                    # prefer kps if exactly divisible by 10
                    role = 'kps'
                    dims = 10
                    A = C // 10
                # scores can be 1 or 2 per anchor; resolve last
                if role is None:
                    role = 'scores'
                    # let reshape auto-detect per-anchor dims (1 or 2) using A from bbox later
                    dims = 0  # auto
                    A = 1
                cand.append((stride, layout, role, A, dims, H, W))
            if not cand:
                continue
            # pick the first candidate; for ambiguous cases NCHW usually correct
            stride, layout, role, A, dims, H, W = cand[0]
            meta = {'index': idx, 'layout': layout, 'A': int(A), 'dims': int(dims), 'H': int(H), 'W': int(W)}
            if stride not in head_map:
                head_map[stride] = {}
            # if multiple candidates claim same role, prefer one with expected dims (bbox=4, kps=10, scores<=2)
            if role in head_map[stride]:
                prev = head_map[stride][role]
                prefer = (role == 'bbox' and dims == 4) or (role == 'kps' and dims == 10) or (role == 'scores' and dims <= 2)
                if prefer:
                    head_map[stride][role] = meta
            else:
                head_map[stride][role] = meta
        # sanity: ensure bbox and scores for each stride exist; drop incomplete strides
        for s in list(head_map.keys()):
            if 'bbox' not in head_map[s] or 'scores' not in head_map[s]:
                del head_map[s]
        # if user specified fmc/strides earlier, filter to them; else keep discovered ones
        if self._feat_stride_fpn:
            head_map = {s: head_map[s] for s in self._feat_stride_fpn if s in head_map}
        logger.debug(f"Built head_map: { {s: list(head_map[s].keys()) for s in head_map} }")
        return head_map

    def _reshape_head(self, arr, meta):
        # Returns flattened per-location per-anchor array with shape:
        #   scores: (K*A,) or (K*A,2)
        #   bbox:   (K*A,4)
        #   kps:    (K*A,kps_dim)
        layout, A, dims, H, W = meta['layout'], meta['A'], meta['dims'], meta['H'], meta['W']
        if layout == 'NCHW':
            # arr shape: (1, C, H, W)
            arr = arr.reshape(1, -1, H, W)
            arr = np.transpose(arr, (0, 2, 3, 1))  # (1,H,W,C)
        else:
            # arr shape: (1, H, W, C)
            arr = arr.reshape(1, H, W, -1)
        C = arr.shape[-1]
        # auto-derive per-anchor dims if requested (dims <= 0)
        if dims is None or dims <= 0:
            if A > 0 and C % A == 0:
                dims = C // A
            else:
                dims = C
        # final shape: (H*W*A, dims)
        out = arr.reshape(-1, C)
        if dims == 1:
            return out.reshape(-1)
        out = out.reshape(-1, dims)
        return out

    # ----------------------------
    # RetinaFace helpers (pre/post)
    # ----------------------------
    def distance2bbox(self, points, distance):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points, distance):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    def nms_det(self, dets, thresh=0.4):
        if dets.size == 0:
            return []
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def _preprocess(self, img):
        ih, iw = img.shape[:2]
        in_w, in_h = self.input_size
        im_ratio = ih / float(iw)
        model_ratio = in_h / float(in_w)
        if im_ratio > model_ratio:
            new_h = in_h
            new_w = int(round(new_h / im_ratio))
        else:
            new_w = in_w
            new_h = int(round(new_w * im_ratio))
        resized = cv2.resize(img, (new_w, new_h))
        det_img = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized
        input_size = (in_w, in_h)
        blob = cv2.dnn.blobFromImage(det_img, 1.0 / self.input_std, input_size,
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        det_scale = new_h / float(ih)
        return blob, det_scale

    def _reorder_retinaface_outputs(self, outputs):
        """
        Reorder interleaved RetinaFace heads [score,bbox,kps]* into InsightFace order:
        [scores_s8.., scores_s16.., scores_s32.., bboxes_*, kps_*]
        Applies only when outputs match expected shapes:
        - 9 heads with last-dims {1,4,10} (with kps), or
        - 6 heads with last-dims {1,4} (no kps).
        """
        try:
            if not isinstance(outputs, (list, tuple)) or len(outputs) not in (6, 9):
                return outputs
            # Collect per-output (N, D) where D is last-dim (1 for scores, 4 for bbox, 10 for kps)
            shapes = [o.shape for o in outputs]
            # Only handle rank-2 heads: (N, D)
            if any(len(s) != 2 for s in shapes):
                return outputs
            Ns = [s[0] for s in shapes]
            Ds = [s[1] for s in shapes]
            uniq_dims = sorted(set(Ds))
            # Validate expected head dims
            if len(outputs) == 9 and uniq_dims != [1, 4, 10]:
                return outputs
            if len(outputs) == 6 and uniq_dims != [1, 4]:
                return outputs

            # Group heads by their last-dimension (1=scores, 4=bbox, 10=kps)
            groups = {d: [] for d in uniq_dims}
            for o in outputs:
                groups[o.shape[1]].append(o)

            # Sort each group by N descending so strides are [s8, s16, s32]
            for d in groups:
                groups[d].sort(key=lambda a: a.shape[0], reverse=True)

            order_dims = [1, 4] + ([10] if 10 in groups else [])
            reordered = []
            for d in order_dims:
                reordered.extend(groups[d])

            return reordered
        except Exception:
            # Fail-safe: if anything unexpected, return original order
            return outputs

    def detect(self, img, max_num=0, metric='default'):
        blob, det_scale = self._preprocess(img)
        logger.debug(f"Preprocess: blob.shape={blob.shape}, det_scale={det_scale:.6f}")
        t0 = time.time()
        net_outs = self.runner.infer(blob)
        net_outs = self._reorder_retinaface_outputs(net_outs)
        t_det = time.time() - t0

        logger.debug(f"TRT detect time={t_det*1000:.2f} ms, outputs={len(net_outs)}")
        for i, o in enumerate(net_outs):
            logger.debug(f"  output[{i}] shape={o.shape}")
        # logger.debug(f"Detect net outputs={net_outs}")

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = 3
        _num_anchors = 2
        _feat_stride_fpn = [8, 16, 32]
        use_kps = True
        scores_list, bboxes_list, kpss_list = [], [], []
        for idx, stride in enumerate(_feat_stride_fpn):
            logger.debug(f"idx={idx}, stride={stride}")
            scores = net_outs[idx]
            bbox_preds = net_outs[idx+fmc]
            bbox_preds = bbox_preds * stride
            logger.debug(f"bbox_preds.spahe={bbox_preds.shape}")
            if use_kps:
                kps_preds = net_outs[idx+fmc*2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape( (-1, 2) )
                if _num_anchors>1:
                    anchor_centers = np.stack([anchor_centers]*_num_anchors, axis=1).reshape( (-1,2) )
                if len(self.center_cache)<100:
                    self.center_cache[key] = anchor_centers
            
            pos_inds = np.where(scores>=self.det_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                #kpss = kps_preds
                kpss = kpss.reshape( (kpss.shape[0], -1, 2) )
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        logger.debug(f"score_list={scores_list}")
        logger.debug(f"bboxes_list={bboxes_list}")
        logger.debug(f"kpss_list={kpss_list}")

        if not scores_list:
            logger.debug("No detections above threshold")
            return np.zeros((0, 5), dtype=np.float32), None, t_det

        scores = np.vstack(scores_list)
        order = scores.ravel().argsort()[::-1]
        bboxes = (np.vstack(bboxes_list) / det_scale).astype(np.float32, copy=False)
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        logger.debug(f"pre_det.shape={pre_det.shape}, top_score={pre_det[:,4].max():.4f}")
        pre_det = pre_det[order, :]
        keep = nms_det(pre_det, self.nms_thresh)
        logger.debug(f"NMS keep={len(keep)} of {pre_det.shape[0]} (nms_thresh={self.nms_thresh})")
        det = pre_det[keep, :]
        kpss = None
        if use_kps and kpss_list:
            kpss = (np.vstack(kpss_list) / det_scale).astype(np.float32, copy=False)
            kpss = kpss[order, :, :][keep, :, :]

        if max_num > 0 and det.shape[0] > max_num:
            logger.debug(f"Limiting detections to max_num={max_num} from {det.shape[0]}")
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            values = area if metric == 'max' else area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1][:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss, t_det
```

</details>

### 性别、年龄
<details markdown="1">
  <summary>attribute.py</summary>

```python
from common import TrtRunner
from loguru import logger
import numpy as np
import cv2
import pycuda.autoinit  # initializes CUDA context
from skimage import transform as trans

def transform(data, center, output_size, scale, rotation):
    scale_ratio = scale
    rot = float(rotation) * np.pi / 180.0
    #translation = (output_size/2-center[0]*scale_ratio, output_size/2-center[1]*scale_ratio)
    t1 = trans.SimilarityTransform(scale=scale_ratio)
    cx = center[0] * scale_ratio
    cy = center[1] * scale_ratio
    t2 = trans.SimilarityTransform(translation=(-1 * cx, -1 * cy))
    t3 = trans.SimilarityTransform(rotation=rot)
    t4 = trans.SimilarityTransform(translation=(output_size / 2,
                                                output_size / 2))
    t = t1 + t2 + t3 + t4
    M = t.params[0:2]
    cropped = cv2.warpAffine(data,
                             M, (output_size, output_size),
                             borderValue=0.0)
    return cropped, M

class AttributeTRT:
    def __init__(self, engine_path):
        self.runner = TrtRunner(engine_path)
        self.input_mean = 0.0
        self.input_std = 1.0

        in_shape = self.runner.engine.get_binding_shape(self.runner.input_indices[0])
        self.input_size = (96 if -1 in in_shape else in_shape[2], 96 if -1 in in_shape else in_shape[3])

    def infer(self, img, face):
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        _scale = self.input_size[0]  / (max(w, h)*1.5)
        aimg, M = transform(img, center, self.input_size[0], _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(aimg, 1.0/self.input_std, input_size,
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        req_shape = blob.shape
        cur_shape = tuple(self.runner.context.get_binding_shape(self.runner.input_indices[0]))
        if (not self.runner._allocated) or (cur_shape != req_shape):
            self.runner.allocate(req_shape)
        pred = self.runner.infer(blob)[0][0]
        pred = np.asarray(pred).reshape(-1)
        if pred.shape[0] == 3:
            gender = int(np.argmax(pred[:2]))
            age = int(np.round(pred[2] * 100))
            return gender, age
        return pred
```

</details>

### landmark
<details markdown="1">
  <summary>landmark.py</summary>

```python
from common import TrtRunner, affine_crop
from loguru import logger
import numpy as np
import cv2
import pycuda.autoinit  # initializes CUDA context

# ----------------------------
# Landmark, Attribute, ArcFace TRT wrappers
# ----------------------------

def trans_points(pts, M):
    # pts: (N,2) or (N,3) where last dim z not used for transform
    pts2 = pts.copy()
    xy = pts2[:, :2]
    ones = np.ones((xy.shape[0], 1), dtype=xy.dtype)
    xy1 = np.hstack([xy, ones])
    dst = xy1 @ M.T
    pts2[:, 0:2] = dst
    return pts2

class LandmarkTRT:
    def __init__(self, engine_path):
        self.runner = TrtRunner(engine_path)
        # Match InsightFace Landmark preprocessing (raw pixels)
        self.input_mean = 0.0
        self.input_std = 1.0

    def infer(self, img, bbox):
        # allocate using engine's declared input if static, else 192x192
        # Read needed input size from binding shape (assume (1,3,H,W))
        # If dynamic, we use 192x192; can adjust if your model differs
        in_shape = self.runner.engine.get_binding_shape(self.runner.input_indices[0])
        H = 192 if -1 in in_shape else in_shape[2]
        W = 192 if -1 in in_shape else in_shape[3]
        assert H == W, "Landmark input must be square"
        aimg, M = affine_crop(img, bbox, H)
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, (W, H),
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True).astype(np.float32)
        logger.debug(f"blob={blob}")
        # Ensure allocation before inference
        req_shape = blob.shape
        cur_shape = tuple(self.runner.context.get_binding_shape(self.runner.input_indices[0]))
        if (not self.runner._allocated) or (cur_shape != req_shape):
            self.runner.allocate(req_shape)
        out = self.runner.infer(blob)[0][0]
        if out.shape[0] >= 3000:
            pred = out.reshape((-1, 3))
            pred[:, 0:2] += 1
            pred[:, 0:2] *= (H // 2)
            pred[:, 2] *= (H // 2)
        else:
            pred = out.reshape((-1, 2))
            pred[:, 0:2] += 1
            pred[:, 0:2] *= (H // 2)
        IM = cv2.invertAffineTransform(M)
        pred = trans_points(pred, IM)
        return pred
```

</details>

### 特征
<details markdown="1">
  <summary>arcface.py</summary>

```python
from common import TrtRunner
from loguru import logger
import numpy as np
import cv2
import pycuda.autoinit  # initializes CUDA context

class ArcFaceTRT:
    def __init__(self, engine_path):
        self.runner = TrtRunner(engine_path)
        self.input_mean = 127.5
        self.input_std = 127.5
        # arcface template for 112x112
        self.dst5 = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)

    def align_by_5p(self, img, kps5, out_size=112):
        src = kps5.astype(np.float32)
        dst = self.dst5.copy()
        if out_size != 112:
            scale = out_size / 112.0
            dst *= scale
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        aimg = cv2.warpAffine(img, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return aimg

    def embed(self, img, kps5):
        in_shape = self.runner.engine.get_binding_shape(self.runner.input_indices[0])
        H = 112 if -1 in in_shape else in_shape[2]
        W = 112 if -1 in in_shape else in_shape[3]
        aimg = self.align_by_5p(img, kps5, out_size=H)
        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, (W, H),
                                     (self.input_mean, self.input_mean, self.input_mean), swapRB=True).astype(np.float32)
        # Ensure allocation before inference
        req_shape = blob.shape
        cur_shape = tuple(self.runner.context.get_binding_shape(self.runner.input_indices[0]))
        if (not self.runner._allocated) or (cur_shape != req_shape):
            self.runner.allocate(req_shape)
        feat = self.runner.infer(blob)[0]
        feat = feat.reshape(-1).astype(np.float32)
        n = np.linalg.norm(feat) + 1e-12
        return feat / n
```

</details>
