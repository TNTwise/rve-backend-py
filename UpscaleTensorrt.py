from spandrel import ImageModelDescriptor, ModelLoader
import torch
import os
from src.Util import currentDirectory
import sys
from dataclasses import dataclass
class Backend:
    @dataclass
    class Eager:
        module: torch.nn.Module

    @dataclass
    class TensorRT:
        module: list[torch.nn.Module]
@torch.inference_mode()
def renderTensorRT(
        upscaleModelPath: str,
        width: int = 1920,
        height: int = 1080,
        device: str = "cuda",
        dtype: str = "float16",
        # trt options
        trt_workspace_size: int = 0,
        trt_max_aux_streams: int | None = None,
        trt_optimization_level: int = 5,
        trt_cache_dir: str = currentDirectory(),
        trt_debug: bool = False,
        ):
    model = ModelLoader(device="cpu").load_from_file(upscaleModelPath)
    model = model.model
    model.load_state_dict(state_dict=model.state_dict())
    model.cuda().eval()
    dtype = torch.float16 if dtype == "float16" else torch.float32
    
    if dtype == torch.float16:
            model.half() 
    import tensorrt
    import torch_tensorrt
    device = torch.device(device, 0)
    

    
    trt_engine_path = os.path.join(
            os.path.realpath(trt_cache_dir),
            (
                f"{os.path.basename(upscaleModelPath)}"
                + f"_{width}x{height}"
                + f"_{'fp16' if dtype == torch.float16 else 'fp32'}"
                + f"_{torch.cuda.get_device_name(device)}"
                + f"_trt-{tensorrt.__version__}"
                + (f"_workspace-{trt_workspace_size}" if trt_workspace_size > 0 else "")
                + ".ts"
            ),
        )
    if not os.path.isfile(trt_engine_path):
        import tensorrt
        import torch_tensorrt
        import torch_tensorrt.ts.logging as logging

        

        logging.set_reportable_log_level(logging.Level.Debug if trt_debug else logging.Level.Info)
        logging.set_is_colored_output_on(True)

        

        if not os.path.isfile(trt_engine_path):
            inputs = [torch.zeros((1, 3, height, width), dtype=torch.half, device="cuda")]
            module = torch.jit.trace(model, inputs)

           

            module = torch_tensorrt.compile(
                module,
                ir="ts",
                inputs=inputs,
                enabled_precisions={dtype},
                device=torch_tensorrt.Device(gpu_id=0),
                workspace_size=trt_workspace_size,
                calibrator=None,
                truncate_long_and_double=True,
                min_block_size=1,
            )

            torch.jit.save(module, trt_engine_path)

        module = torch.jit.load(trt_engine_path)
        backend = Backend.TensorRT(module)
    else:
        backend = Backend.Eager(module)

renderTensorRT(upscaleModelPath=sys.argv[1])