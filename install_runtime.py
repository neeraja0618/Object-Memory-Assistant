"""Install a matching CPU torch/torchvision pair in this project's environment."""
import subprocess
from pathlib import Path
import sys

CHECK="import torch, torchvision; from torchvision.ops import nms; nms(torch.tensor([[0.,0.,10.,10.]]), torch.tensor([.9]), .5); assert torch.__version__.split('+')[0] == '2.6.0'; assert torchvision.__version__.split('+')[0] == '0.21.0'"


def main():
    expected=Path(__file__).resolve().parent/'.venv'
    if Path(sys.prefix).resolve()!=expected.resolve():
        print('Use this project environment: .\\.venv\\Scripts\\python.exe install_runtime.py')
        return 2
    check=subprocess.run([sys.executable,'-c',CHECK],capture_output=True,text=True)
    if check.returncode==0:
        print('Matching PyTorch/Torchvision runtime already works.')
        return 0
    print('Installing matching CPU builds: torch 2.6.0 / torchvision 0.21.0.',flush=True)
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torch','torchvision'],check=True)
    subprocess.run([sys.executable,'-m','pip','install','--no-cache-dir','torch==2.6.0','torchvision==0.21.0',
                    '--index-url','https://download.pytorch.org/whl/cpu'],check=True)
    verified=subprocess.run([sys.executable,'-c',CHECK])
    return verified.returncode


if __name__=='__main__':raise SystemExit(main())
