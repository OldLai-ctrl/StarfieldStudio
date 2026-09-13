"""ToupCam Windows x64 ABI; pull mode with blocking WaitImageV3, RAW only."""
import ctypes as C
import os
from pathlib import Path
import numpy as np
from core import FrameMeta
from camera_common import VALID_BITS, check_value, check_readback, validate_native_frame

U=C.c_uint32; I=C.c_int32; P=C.c_void_p; US=C.c_uint16
class Resolution(C.Structure):_fields_=[('width',U),('height',U)]
class Model(C.Structure):
    _fields_=[('name',C.c_wchar_p),('flag',C.c_uint64),('maxspeed',U),('preview',U),('still',U),('maxfanspeed',U),('ioctrol',U),('xpixsz',C.c_float),('ypixsz',C.c_float),('res',Resolution*16)]
class Device(C.Structure):_fields_=[('displayname',C.c_wchar*64),('id',C.c_wchar*64),('model',C.POINTER(Model))]
class FrameInfo(C.Structure):
    _fields_=[('width',U),('height',U),('flag',U),('seq',U),('timestamp',C.c_uint64),('shutterseq',U),('expotime',U),('expogain',US),('blacklevel',US)]

class ToupCamera:
    def __init__(self,path,index=0):
        self.handle=None;self.active=False;self.dir_handle=None;self.discard=0
        if os.name!='nt' or C.sizeof(P)!=8:raise RuntimeError('图谱接口需要 Windows x64')
        p=Path(path).resolve()
        if not p.is_file():raise RuntimeError('未找到图谱接口，请选择官方 x64 toupcam.dll')
        try:
            self.dir_handle=os.add_dll_directory(str(p.parent));self.dll=C.WinDLL(str(p))
            signatures={
                'EnumV2':([C.POINTER(Device)],U),'Open':([C.c_wchar_p],P),'Close':([P],None),
                'get_MonoMode':([P],I),'get_MaxBitDepth':([P],I),
                'put_Option':([P,U,I],I),'get_Option':([P,U,C.POINTER(I)],I),
                'get_RawFormat':([P,C.POINTER(U),C.POINTER(U)],I),
                'put_AutoExpoEnable':([P,I],I),'get_AutoExpoEnable':([P,C.POINTER(I)],I),
                'get_ExpTimeRange':([P,C.POINTER(U),C.POINTER(U),C.POINTER(U)],I),
                'get_ExpoAGainRange':([P,C.POINTER(US),C.POINTER(US),C.POINTER(US)],I),
                'put_ExpoTime':([P,U],I),'get_ExpoTime':([P,C.POINTER(U)],I),
                'put_ExpoAGain':([P,US],I),'get_ExpoAGain':([P,C.POINTER(US)],I),
                'get_ResolutionNumber':([P],I),'get_Resolution':([P,U,C.POINTER(I),C.POINTER(I)],I),
                'get_eSize':([P,C.POINTER(U)],I),'put_eSize':([P,U],I),
                'get_Size':([P,C.POINTER(I),C.POINTER(I)],I),
                'StartPullModeWithCallback':([P,P,P],I),'Stop':([P],I),
                'WaitImageV3':([P,U,P,I,I,I,C.POINTER(FrameInfo)],I),
            }
            for name,(args,result) in signatures.items():
                fn=getattr(self.dll,'Toupcam_'+name);fn.argtypes=args;fn.restype=result
            devices=(Device*128)();n=self.dll.Toupcam_EnumV2(devices)
            if not 0<=index<min(n,128):raise RuntimeError(f'图谱接口发现 {n} 台相机，设备序号无效；请检查驱动及占用情况')
            device=devices[index]
            if not device.model or not device.model.contents.flag&0x10:raise RuntimeError('当前版本仅接受黑白相机，拒绝彩色/Bayer 型号')
            self.name=device.displayname;self.handle=self.dll.Toupcam_Open(device.id)
            if not self.handle:raise RuntimeError('图谱相机打开失败，请关闭其他采集软件')
            if self.dll.Toupcam_get_MonoMode(self.handle)!=0:raise RuntimeError('图谱接口未确认黑白传感器')
            depth=self.dll.Toupcam_get_MaxBitDepth(self.handle)
            if depth not in VALID_BITS:raise RuntimeError('图谱接口返回未知位深')
            # ToupCam exposes 8-bit and the sensor's maximum RAW depth via
            # TOUPCAM_OPTION_BITDEPTH (0/1).  Do not invent intermediate modes.
            self.max_bits=depth;self.bit_options=[(8,'Mono8 · 原始')]
            if depth!=8:self.bit_options.append((depth,f'Mono{depth} · 原始'))
            self.input_bits=depth
            self.call('put_Option',4,1);self.call('put_Option',6,int(self.input_bits>8))
            self.ensure_manual()
            # Remove trigger, ROI and SDK digital binning from previous client state.
            self.call('put_Option',0x0b,0)
            if hasattr(self.dll,'Toupcam_put_Roi'):
                f=self.dll.Toupcam_put_Roi;f.argtypes=[P,U,U,U,U];f.restype=I;self.check(f(self.handle,0,0,0,0),'恢复完整画面')
            self.call('put_Option',0x17,1)
            self.resolutions=[]
            count=self.dll.Toupcam_get_ResolutionNumber(self.handle)
            if not 0<count<=128:raise RuntimeError('图谱分辨率列表无效')
            for j in range(count):
                w,h=I(),I();self.call('get_Resolution',j,C.byref(w),C.byref(h));self.resolutions.append((j,f'{w.value} × {h.value}'))
            self.mode=self.get('get_eSize',U);self.bin_value=None;self.bins=[]
            self.meta=FrameMeta(camera='ToupCam '+self.name+' '+device.id)
            self.refresh()
            self.configure(exposure=min(max(100,self.exp_range[0]),self.exp_range[1]))
        except Exception:self.close();raise
    @staticmethod
    def check(ret,label):
        if ret<0:raise RuntimeError(f'图谱接口 {label} 失败：0x{ret&0xffffffff:08X}')
    def call(self,name,*args):
        result=getattr(self.dll,'Toupcam_'+name)(self.handle,*args);self.check(result,name);return result
    def get(self,name,typ):
        v=typ();self.call(name,C.byref(v));return v.value
    def ensure_manual(self):
        self.call('put_AutoExpoEnable',0)
        if self.get('get_AutoExpoEnable',I)!=0:raise RuntimeError('图谱相机自动曝光未关闭')
    def refresh(self):
        lo,hi,default=U(),U(),U();self.call('get_ExpTimeRange',C.byref(lo),C.byref(hi),C.byref(default));self.exp_range=(lo.value/1000,hi.value/1000,.001)
        lo,hi,default=US(),US(),US();self.call('get_ExpoAGainRange',C.byref(lo),C.byref(hi),C.byref(default));self.gain_range=(lo.value,hi.value,1)
        fourcc,bits=U(),U();self.call('get_RawFormat',C.byref(fourcc),C.byref(bits))
        if bits.value not in (8,10,11,12,14,16):raise RuntimeError('不支持当前图谱原始位深')
        if hasattr(self,'bit_options') and bits.value not in [v for v,_ in self.bit_options]:
            raise RuntimeError(f'图谱接口返回未选择的原始位深 Mono{bits.value}')
        self.meta.bits=bits.value;self.input_bits=bits.value;self.meta.exposure_ms=self.get('get_ExpoTime',U)/1000;self.meta.gain=self.get('get_ExpoAGain',US)
        w,h=I(),I();self.call('get_Size',C.byref(w),C.byref(h));self.shape=(h.value,w.value)
        if min(self.shape)<=0 or max(self.shape)>30000 or w.value*h.value*2>512*1024**2:raise RuntimeError('图谱帧尺寸无效')
        self.buffer=np.empty(self.shape,np.uint8 if bits.value==8 else np.uint16)
        self.meta.mode=f'ToupCam;res={self.mode};raw{bits.value};shape={self.shape}'
    def start(self):
        if self.active:return
        self.ensure_manual();self.call('StartPullModeWithCallback',None,None);self.active=True;self.discard=2
    def stop(self):
        if self.active:self.call('Stop');self.active=False
    def configure(self,exposure=None,gain=None,resolution=None,native_bin=None,input_bits=None):
        if native_bin is not None:raise ValueError('此接口使用软件 Binning')
        if input_bits is not None:
            input_bits=int(input_bits)
            if input_bits not in [v for v,_ in self.bit_options]:raise ValueError(f'图谱相机不提供 Mono{input_bits}')
        was=self.active
        if was:self.stop()
        if input_bits is not None and input_bits!=self.input_bits:
            self.call('put_Option',6,int(input_bits>8))
            self.refresh()
            if self.input_bits!=input_bits:raise RuntimeError(f'图谱位深读回不一致：请求 Mono{input_bits}，实际 Mono{self.input_bits}')
        if resolution is not None:
            if resolution not in [v for v,_ in self.resolutions]:raise ValueError('无效分辨率')
            self.call('put_eSize',resolution);self.mode=self.get('get_eSize',U)
            if self.mode!=resolution:raise RuntimeError('图谱分辨率读回不一致')
        self.refresh();self.ensure_manual()
        if exposure is not None:
            target=check_value(exposure,self.exp_range,'曝光');self.call('put_ExpoTime',round(target*1000))
            check_readback(target,self.get('get_ExpoTime',U)/1000,.001,'曝光')
        if gain is not None:
            target=check_value(gain,self.gain_range,'增益');self.call('put_ExpoAGain',round(target))
            check_readback(target,self.get('get_ExpoAGain',US),1,'增益')
        self.refresh()
        if was:self.start()
    def read(self):
        info=FrameInfo();self.call('WaitImageV3',max(3000,int(self.meta.exposure_ms*2+2000)),self.buffer.ctypes.data,0,16,-1,C.byref(info))
        if (info.height,info.width)!=self.shape:raise RuntimeError('图谱实际帧尺寸与所选模式不符')
        if self.discard:self.discard-=1;return None
        if self.get('get_AutoExpoEnable',I)!=0:raise RuntimeError('图谱自动曝光状态发生变化')
        self.meta.exposure_ms=self.get('get_ExpoTime',U)/1000;self.meta.gain=self.get('get_ExpoAGain',US)
        return validate_native_frame(self.buffer.astype(np.uint16,copy=True),self.input_bits)
    def close(self):
        try:
            if self.handle:
                try:self.stop()
                finally:self.dll.Toupcam_Close(self.handle);self.handle=None
        finally:
            if self.dir_handle:self.dir_handle.close();self.dir_handle=None
