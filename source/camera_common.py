"""Shared contracts for native monochrome camera backends."""
from pathlib import Path
import os, sys
import numpy as np

BACKENDS={'tucam':('TUCam','TUCam.dll'), 'hik':('海康 MVS','MvCameraControl_class.py'), 'toup':('图谱 ToupCam','toupcam.dll')}

def root_dir():
    return Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent.parent

def sdk_path(backend):
    root=root_dir(); filename=BACKENDS[backend][1]
    candidates=[root/'sdk'/backend/filename,root/'sdk'/filename]
    if backend=='hik':
        roots=[os.environ.get('MVCAM_SDK_PATH',''),r'C:\Program Files (x86)\MVS',r'C:\Program Files\MVS']
        for base in roots:
            if base:candidates.extend([Path(base)/'Development/Samples/Python/MvImport'/filename,Path(base)/'Samples/Python/MvImport'/filename])
    return str(next((p for p in candidates if p.is_file()),candidates[0]))

def check_value(value,limits,label):
    value=float(value);lo,hi,step=limits
    if not np.isfinite(value) or not lo<=value<=hi:raise ValueError(f'{label}需在 {lo:g}—{hi:g} 范围内')
    return value

def check_readback(target,actual,step,label):
    if abs(target-actual)>max(step*1.1,abs(target)*.0001,.001):
        raise RuntimeError(f'{label}读回不一致：请求 {target:g}，实际 {actual:g}')

def unpack_mono(data,width,height,bits,frame_bytes=None):
    """Unpacked little-endian Mono8/10/12/14/16. Keep native ADU in uint16."""
    if bits not in (8,10,11,12,14,16):raise RuntimeError('暂不支持此黑白像素格式')
    if not 0<width<=30000 or not 0<height<=30000:raise RuntimeError('无效图像尺寸')
    size=width*height*(1 if bits==8 else 2)
    if size>512*1024**2 or len(data)<size or (frame_bytes is not None and frame_bytes!=size):
        raise RuntimeError('帧长度与非压缩黑白格式不符；不接受截断帧、填充或未知格式')
    return np.frombuffer(data,dtype=np.uint8 if bits==8 else '<u2',count=width*height).reshape(height,width).astype(np.uint16,copy=True)

def open_camera(backend,path,index):
    if index<0:raise ValueError('设备序号不能为负')
    if backend=='tucam':
        from camera import TucamCamera
        return TucamCamera(path,index)
    if backend=='toup':
        from camera_toup import ToupCamera
        return ToupCamera(path,index)
    if backend=='hik':
        from camera_hik import HikCamera
        return HikCamera(path,index)
    raise ValueError('未知相机接口')
