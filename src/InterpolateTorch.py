import torch
import math
import os

from .InterpolateArchs.DetectInterpolateArch import loadInterpolationModel
from .Util import currentDirectory
torch.set_float32_matmul_precision("high")


class InterpolateRifeTorch:
    @torch.inference_mode()
    def __init__(
        self,
        interpolateModelPath: str,
        width: int = 1920,
        height: int = 1080,
        device: str = "cuda",
        dtype: str = "float16",
        backend: str = "pytorch",
        UHDMode: bool = False,
        ensemble: bool = False,


        #trt options
        trt_min_shape: list[int] = [128, 128],
        trt_opt_shape: list[int] = [1920, 1080],
        trt_max_shape: list[int] = [1920, 1080],
        trt_workspace_size: int = 0,
        trt_max_aux_streams: int | None = None,
        trt_optimization_level: int | None = None,
        trt_cache_dir: str = currentDirectory(),
        trt_debug: bool = False,
    ):
        
        self.interpolateModel = interpolateModelPath
        self.width = width
        self.height = height
        self.device = device
        self.dtype = self.handlePrecision(dtype)
        self.backend = backend
        state_dict = torch.load(interpolateModelPath)
        scale = 1
        if UHDMode:
            scale = .5
        model = loadInterpolationModel(state_dict)
        self.flownet = (model
                        .getIFnet(scale=scale,ensemble=ensemble)
                        .eval()
                        .to(device=self.device,dtype=self.dtype)
                        .load_state_dict(state_dict=state_dict,assign=True,strict=True)
                        )
        tmp = max(32, int(32 / scale))
        self.pw = math.ceil(self.width / tmp) * tmp
        self.ph = math.ceil(self.height / tmp) * tmp
        self.padding = (0, self.pw - self.width, 0, self.ph - self.height)
        self.tenFlow_div = torch.tensor([(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0], dtype=dtype, device=device)

        tenHorizontal = torch.linspace(-1.0, 1.0, self.pw, dtype=dtype, device=device).view(1, 1, 1, self.pw).expand(-1, -1, self.ph, -1)
        tenVertical = torch.linspace(-1.0, 1.0, self.ph, dtype=dtype, device=device).view(1, 1, self.ph, 1).expand(-1, -1, -1, self.pw)
        self.backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)
        
        if self.backend == "tensorrt":
            import tensorrt
            import torch_tensorrt
            for i in range(2):
                trt_min_shape[i] = math.ceil(max(trt_min_shape[i], 1) / tmp) * tmp
                trt_opt_shape[i] = math.ceil(max(trt_opt_shape[i], 1) / tmp) * tmp
                trt_max_shape[i] = math.ceil(max(trt_max_shape[i], 1) / tmp) * tmp

            dimensions = (
                f"min-{trt_min_shape[0]}x{trt_min_shape[1]}"
                f"_opt-{trt_opt_shape[0]}x{trt_opt_shape[1]}"
                f"_max-{trt_max_shape[0]}x{trt_max_shape[1]}"
            )
            trt_engine_path = os.path.join(
                os.path.realpath(trt_cache_dir),
                (
                    f"{os.path.basename(self.interpolateModel)}"
                    + f"_{dimensions}"
                    + f"_{'fp16' if self.dtype == torch.float16 else 'fp32'}"
                    + f"_scale-{scale}"
                    + f"_ensemble-{ensemble}"
                    + f"_{torch.cuda.get_device_name(device)}"
                    + f"_trt-{tensorrt.__version__}"
                    + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                    + (f"_aux-{trt_max_aux_streams}" if trt_max_aux_streams is not None else "")
                    + (f"_level-{trt_optimization_level}" if trt_optimization_level is not None else "")
                    + ".ts"
                ),
        )
        if not os.path.isfile(trt_engine_path):
            trt_min_shape.reverse()
            trt_opt_shape.reverse()
            trt_max_shape.reverse()

            example_tensors = (
                torch.zeros((1, 3, self.ph, self.ph), dtype=dtype, device=device),
                torch.zeros((1, 3, self.ph, self.ph), dtype=dtype, device=device),
                torch.zeros((1, 1, self.ph, self.ph), dtype=dtype, device=device),
                torch.zeros((2,), dtype=dtype, device=device),
                torch.zeros((1, 2, self.ph, self.ph), dtype=dtype, device=device),
            )

            _height = torch.export.Dim("height", min=trt_min_shape[0] // tmp, max=trt_max_shape[0] // tmp)
            _width = torch.export.Dim("width", min=trt_min_shape[1] // tmp, max=trt_max_shape[1] // tmp)
            dim_height = _height * tmp
            dim_width = _width * tmp
            dynamic_shapes = {
                "img0": {2: dim_height, 3: dim_width},
                "img1": {2: dim_height, 3: dim_width},
                "timestep": {2: dim_height, 3: dim_width},
                "tenFlow_div": {0: None},
                "backwarp_tenGrid": {2: dim_height, 3: dim_width},
            }

            exported_program = torch.export.export(flownet, example_tensors, dynamic_shapes=dynamic_shapes)

            inputs = [
                torch_tensorrt.Input(
                    min_shape=[1, 3] + trt_min_shape,
                    opt_shape=[1, 3] + trt_opt_shape,
                    max_shape=[1, 3] + trt_max_shape,
                    dtype=dtype,
                    name="img0",
                ),
                torch_tensorrt.Input(
                    min_shape=[1, 3] + trt_min_shape,
                    opt_shape=[1, 3] + trt_opt_shape,
                    max_shape=[1, 3] + trt_max_shape,
                    dtype=dtype,
                    name="img1",
                ),
                torch_tensorrt.Input(
                    min_shape=[1, 1] + trt_min_shape,
                    opt_shape=[1, 1] + trt_opt_shape,
                    max_shape=[1, 1] + trt_max_shape,
                    dtype=dtype,
                    name="timestep",
                ),
                torch_tensorrt.Input(
                    shape=[2],
                    dtype=dtype,
                    name="tenFlow_div",
                ),
                torch_tensorrt.Input(
                    min_shape=[1, 2] + trt_min_shape,
                    opt_shape=[1, 2] + trt_opt_shape,
                    max_shape=[1, 2] + trt_max_shape,
                    dtype=dtype,
                    name="backwarp_tenGrid",
                ),
            ]

            flownet = torch_tensorrt.dynamo.compile(
                exported_program,
                inputs,
                enabled_precisions={dtype},
                debug=trt_debug,
                workspace_size=trt_workspace_size,
                min_block_size=1,
                max_aux_streams=trt_max_aux_streams,
                optimization_level=trt_optimization_level,
                device=device,
                assume_dynamic_shape_support=True,
            )

            torch_tensorrt.save(flownet, trt_engine_path, output_format="torchscript", inputs=example_tensors)

        flownet = torch.jit.load(trt_engine_path).eval()

    def handlePrecision(self, precision):
        if precision == "float32":
            return torch.float32
        if precision == "float16":
            return torch.float16

    @torch.inference_mode()
    def process(self,img0,img1,timestep):
        if timestep == 1:
            return img1
        if timestep == 0:
            return img0

        timestep = torch.full((1, 1, self.ph, self.pw), timestep, dtype=self.dtype, device=self.device)