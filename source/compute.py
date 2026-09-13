"""Optional hardware acceleration through the OpenCV OpenCL runtime.

OpenCV's UMat path uses the vendor driver, so the same implementation works
with mainstream NVIDIA cards and Intel integrated graphics when their OpenCL
runtime is installed.  Nothing is required for CPU mode and all failures
fall back to NumPy without stopping capture.
"""
from __future__ import annotations
import cv2
import numpy as np

DEVICE_CHOICES=('自动（优先 GPU）','CPU','GPU（OpenCL）')

class ComputeEngine:
    def __init__(self,requested='自动（优先 GPU）'):
        self.requested=requested if requested in DEVICE_CHOICES else DEVICE_CHOICES[0]
        self.kind='cpu';self.label='CPU';self.reason='未启用 GPU'
        self._cache={}
        self._probe()
    @staticmethod
    def probe():
        """Return a short, serialisable description of the available device."""
        result={'opencl':False,'label':'未发现可用 OpenCL GPU','vendor':'','name':''}
        try:
            if not cv2.ocl.haveOpenCL():return result
            device=cv2.ocl.Device_getDefault()
            if not device.available() or not device.compilerAvailable():return result
            vendor=str(device.vendorName());name=str(device.name())
            result.update(opencl=True,vendor=vendor,name=name,label=f'{vendor} · {name}')
        except Exception as exc:
            result['label']='OpenCL 检测失败：'+str(exc)
        return result
    def _probe(self):
        info=self.probe();self.device_info=info
        if self.requested=='CPU':
            self.kind='cpu';self.label='CPU';self.reason='按设置使用 CPU';return
        if info['opencl']:
            try:
                cv2.ocl.setUseOpenCL(True)
                self.kind='opencl';self.label='GPU / OpenCL · '+info['label'];self.reason='已启用 OpenCL'
                return
            except Exception as exc:self.reason='OpenCL 启用失败：'+str(exc)
        self.kind='cpu';self.label='CPU 回退';
        self.reason='未找到可用 OpenCL GPU，已回退 CPU' if self.requested!='CPU' else '按设置使用 CPU'
    @property
    def use_gpu(self):return self.kind=='opencl'
    def status(self):
        return dict(requested=self.requested,kind=self.kind,label=self.label,reason=self.reason,
                    vendor=self.device_info.get('vendor',''),name=self.device_info.get('name',''))
    def set_mode(self,requested):
        new=ComputeEngine(requested)
        changed=(new.kind!=self.kind or new.requested!=self.requested or new.label!=self.label)
        self.__dict__.update(new.__dict__)
        return changed
    def disable_gpu(self,reason=''):
        """Switch this engine to CPU after a runtime OpenCL failure.

        OpenCL availability can change while a driver is being initialized.
        Keeping the fallback here lets a live capture continue instead of
        turning one failed UMat operation into a stopped worker thread.
        """
        self.kind='cpu';self.label='CPU 回退';self.reason='OpenCL 运算失败，已回退 CPU'
        if reason:self.reason+='：'+str(reason)
        self._cache.clear()
    def upload(self,array):
        if not self.use_gpu:return np.asarray(array)
        return cv2.UMat(np.ascontiguousarray(array))
    def upload_cached(self,array):
        if not self.use_gpu:return np.asarray(array)
        key=(id(array),array.shape,str(array.dtype))
        value=self._cache.get(key)
        if value is None:
            value=cv2.UMat(np.ascontiguousarray(array));self._cache[key]=value
        return value
    @staticmethod
    def download(value):
        return value.get() if isinstance(value,cv2.UMat) else np.asarray(value)
    def add(self,left,right):
        if self.use_gpu:return cv2.add(left,right)
        return np.asarray(left)+np.asarray(right)
    def subtract(self,left,right):
        if self.use_gpu:return cv2.subtract(left,right)
        return np.asarray(left)-np.asarray(right)
    def multiply(self,left,scale):
        if self.use_gpu:return cv2.multiply(left,float(scale))
        return np.asarray(left)*float(scale)
    def divide(self,left,scale):
        if self.use_gpu:return cv2.divide(left,float(scale))
        return np.asarray(left)/float(scale)
    def maximum(self,left,right):
        if self.use_gpu:return cv2.max(left,right)
        return np.maximum(np.asarray(left),np.asarray(right))

def available_label():
    return ComputeEngine.probe()['label']
