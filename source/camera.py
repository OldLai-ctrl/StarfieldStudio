"""TUCam native RAW interface. ABI declarations verified against SDK headers."""
from __future__ import annotations
import ctypes as C
from pathlib import Path
import os, sys, time
import numpy as np
import cv2
from core import FrameMeta
from camera_common import VALID_BITS, validate_native_frame

I=C.c_int32; U=C.c_uint32; D=C.c_double; P=C.c_void_p
class Init(C.Structure): _fields_=[('count',U),('host_index',I),('path',C.c_char_p)]
class Open(C.Structure): _fields_=[('index',U),('handle',P)]
class Info(C.Structure): _fields_=[('id',I),('value',I),('text',P),('size',I)]
class Text(C.Structure): _fields_=[('id',I),('value',D),('text',P),('size',I)]
class Reg(C.Structure): _fields_=[('kind',I),('buffer',P),('size',I)]
class CapAttr(C.Structure): _fields_=[('id',I),('low',I),('high',I),('default',I),('step',I)]
class PropAttr(C.Structure): _fields_=[('id',I),('channel',I),('low',D),('high',D),('default',D),('step',D)]
class Frame(C.Structure):
    _fields_=[('signature',C.c_char*8),('header',C.c_uint16),('offset',C.c_uint16),
              ('width',C.c_uint16),('height',C.c_uint16),('stride',U),('depth',C.c_uint8),
              ('format',C.c_uint8),('channels',C.c_uint8),('element',C.c_uint8),('requested',C.c_uint8),
              ('index',U),('image_size',U),('reserved',U),('histogram_size',U),('buffer',P)]

def decode_frame(f,acquisition_bits=None):
    # Keep the legacy strict check when callers do not identify the requested
    # acquisition depth.  The camera adapter passes the selected native depth.
    explicit_bits=acquisition_bits is not None
    if acquisition_bits is None:
        if f.element!=2 or f.channels!=1 or f.depth<=8:
            raise RuntimeError(f'拒绝非高位深原始帧：depth={f.depth}, bytes={f.element}, channels={f.channels}。不会用8位图冒充16位采集。')
        acquisition_bits=16
    if acquisition_bits not in VALID_BITS:raise RuntimeError('TUCam 返回未知输入位深')
    if explicit_bits and f.format!=0x10:
        raise RuntimeError(f'拒绝非 RAW 格式帧：format=0x{f.format:02X}')
    expected_bytes=1 if acquisition_bits==8 else 2
    if f.channels!=1 or f.element not in (expected_bytes,2) or (acquisition_bits>8 and f.element!=2):
        raise RuntimeError(f'拒绝与 Mono{acquisition_bits} 不符的原始帧：depth={f.depth}, bytes={f.element}, channels={f.channels}')
    if acquisition_bits>8 and f.depth not in (2,16):
        raise RuntimeError(f'拒绝未确认的 Mono{acquisition_bits} 帧：depth={f.depth}')
    if not f.buffer or not (0<f.width<=30000 and 0<f.height<=30000) or f.stride<f.width*expected_bytes:
        raise RuntimeError('SDK 返回无效图像尺寸／缓冲区')
    if f.stride*f.height>512*1024**2:raise RuntimeError('SDK 帧尺寸超出安全范围')
    offset=int(f.offset)
    if offset<f.header:raise RuntimeError('SDK 图像偏移小于头部长度')
    # Respect SDK returned offset and row padding; never assume offset==1024.
    buf=(C.c_uint8*(f.stride*f.height)).from_address(int(f.buffer)+offset)
    dtype=np.uint8 if f.element==1 else np.uint16
    result=np.ndarray((f.height,f.width),dtype,buffer=buf,strides=(f.stride,f.element)).astype(np.uint16,copy=True)
    return validate_native_frame(result,acquisition_bits)

class TucamCamera:
    def __init__(self,dll_path,index=0):
        if os.name!='nt' or C.sizeof(P)!=8:raise RuntimeError('TUCam 接口仅支持 Windows x64')
        p=Path(dll_path).resolve()
        if not p.is_file():raise RuntimeError('请选择官方 SDK 的 x64 TUCam.dll')
        self.dir_handle=os.add_dll_directory(str(p.parent))
        self.dll=C.CDLL(str(p));self.handle=None;self.initialized=False;self.active=False;self.allocated=False
        self.frame=Frame();self.last_index=None;self.index=index;self.discard=0
        self.ready=False
        signatures={
            'TUCAM_Api_Init':[C.POINTER(Init),I],'TUCAM_Api_Uninit':[],
            'TUCAM_Reg_Read':[P,Reg],
            'TUCAM_Dev_Open':[C.POINTER(Open)],'TUCAM_Dev_Close':[P],
            'TUCAM_Dev_GetInfo':[P,C.POINTER(Info)],
            'TUCAM_Capa_GetAttr':[P,C.POINTER(CapAttr)],
            'TUCAM_Capa_GetValue':[P,I,C.POINTER(I)],'TUCAM_Capa_SetValue':[P,I,I],
            'TUCAM_Capa_GetValueText':[P,C.POINTER(Text)],
            'TUCAM_Prop_GetAttr':[P,C.POINTER(PropAttr)],
            'TUCAM_Prop_GetValue':[P,I,C.POINTER(D),I],'TUCAM_Prop_SetValue':[P,I,D,I],
            'TUCAM_Buf_Alloc':[P,C.POINTER(Frame)],'TUCAM_Buf_Release':[P],
            'TUCAM_Buf_AbortWait':[P],
            'TUCAM_Buf_WaitForFrame':[P,C.POINTER(Frame),I],
            'TUCAM_Cap_Start':[P,U],'TUCAM_Cap_Stop':[P],
        }
        for name,args in signatures.items():
            f=getattr(self.dll,name);f.argtypes=args;f.restype=U
        try:
            init=Init(0,0,None);self.check(self.dll.TUCAM_Api_Init(C.byref(init),1000),'初始化');self.initialized=True
            if not 0<=index<init.count:raise RuntimeError('TUCam 未发现所选设备。请安装对应驱动，检查连接并关闭其他采集软件。')
            op=Open(index,None);self.check(self.dll.TUCAM_Dev_Open(C.byref(op)),'打开相机');self.handle=op.handle
            self.name=self.info_text(9) or 'TUCSEN'
            sn=C.create_string_buffer(64)
            rr=self.dll.TUCAM_Reg_Read(self.handle,Reg(1,C.cast(sn,P),64))
            serial=sn.value.decode(errors='replace') if rr==1 else f'index{index}'
            self.meta=FrameMeta(camera=self.name+' '+serial)
            channels=Info(0x0C,0,None,0)
            self.check(self.dll.TUCAM_Dev_GetInfo(self.handle,C.byref(channels)),'确认黑白通道')
            if channels.value!=1:
                raise RuntimeError('接口未确认单通道黑白型号，拒绝彩色、Bayer 或未知通道类型。')
            self.exp_range=self.prop_attr(1);self.gain_range=self.prop_attr(0)
            self.resolutions=self.cap_options(0)
            # Native binning is capability-based. Unsupported controls are not fabricated.
            self.bins=self.cap_options(0x26)
            options=self.cap_options(2)
            import re
            bit_modes={}
            for value,text in options:
                m=re.search(r'(?<!\d)(8|10|11|12|14|16)(?:\s*(?:bit|bits|位))?(?!\d)',text,re.I)
                if m:bit_modes[int(m.group(1))]=value
                elif value in VALID_BITS:bit_modes[int(value)]=value
            if not bit_modes:raise RuntimeError(f'驱动未明确提供可切换位深：{options}')
            self.bit_modes=bit_modes
            self.bit_options=[(bits,f'Mono{bits} · 原始') for bits in sorted(bit_modes)]
            self.input_bits=max(bit_modes)
            self.bit_mode=bit_modes[self.input_bits]
            self.meta.exposure_ms=self.get_prop(1);self.meta.gain=self.get_prop(0)
            self.mode=self.get_cap(0);self.bin_value=self.get_cap(0x26,optional=True)
            self.meta.bits=self.input_bits
            self.meta.mode=f'res={self.mode};bin={self.bin_value};raw{self.input_bits}'
            # Some USB controls become writable only after stream initialization.
            self.start()
            if self.get_cap(2)!=self.bit_mode:
                self.set_cap(2,self.bit_mode);self.stop();self.start()
            self._read_bit_mode()
            self.set_cap(3,0)
            self.set_prop(1,min(max(100.,self.exp_range[0]),self.exp_range[1]))
            self.meta.exposure_ms=self.get_prop(1);self.meta.gain=self.get_prop(0);self.discard=2
            self.ready=True
        except Exception:
            self.close();raise
    @staticmethod
    def check(ret,operation):
        if ret==0x8000020B:raise RuntimeError('相机 USB 通信失败（0x8000020B）。请关闭其他采集软件，重新连接后重试，并确认 x64 TUCam.dll 与驱动版本匹配。')
        if ret!=1:raise RuntimeError(f'TUCam 接口：{operation}失败，代码 0x{ret:08X}')
    def info_text(self,id):
        buf=C.create_string_buffer(256);v=Info(id,0,C.cast(buf,P),256)
        return C.string_at(v.text).decode(errors='replace') if self.dll.TUCAM_Dev_GetInfo(self.handle,C.byref(v))==1 and v.text else ''
    def cap_options(self,id):
        a=CapAttr(id)
        if self.dll.TUCAM_Capa_GetAttr(self.handle,C.byref(a))!=1:return []
        if a.high-a.low>256:return []
        out=[]
        for n in range(a.low,a.high+1,max(1,a.step)):
            b=C.create_string_buffer(256);v=Text(id,n,C.cast(b,P),256)
            ret=self.dll.TUCAM_Capa_GetValueText(self.handle,C.byref(v))
            out.append((n,C.string_at(v.text).decode(errors='replace') if ret==1 and v.text else str(n)))
        return out
    def prop_attr(self,id):
        a=PropAttr(id,0);self.check(self.dll.TUCAM_Prop_GetAttr(self.handle,C.byref(a)),'读取参数范围')
        return (a.low,a.high,a.step)
    def get_prop(self,id):
        d=D();self.check(self.dll.TUCAM_Prop_GetValue(self.handle,id,C.byref(d),0),'读取参数');return d.value
    def set_prop(self,id,value):self.check(self.dll.TUCAM_Prop_SetValue(self.handle,id,value,0),'设置参数')
    def get_cap(self,id,optional=False):
        v=I();r=self.dll.TUCAM_Capa_GetValue(self.handle,id,C.byref(v))
        if optional and r!=1:return None
        self.check(r,'读取模式');return v.value
    def set_cap(self,id,v):self.check(self.dll.TUCAM_Capa_SetValue(self.handle,id,v),'设置模式')
    def _read_bit_mode(self):
        actual=self.get_cap(2)
        for bits,value in self.bit_modes.items():
            if value==actual:
                self.input_bits=bits;self.bit_mode=value;self.meta.bits=bits
                return bits
        raise RuntimeError(f'TUCam 位深模式读回不一致：{actual}，可用 {self.bit_modes}')
    def ensure_manual(self):
        self.set_cap(3,0)
        if self.get_cap(3)!=0:raise RuntimeError('相机自动曝光未能关闭，已停止使用未确认的参数')
    def start(self):
        if self.active:return
        # TUCam's request field is the RAW format selector (0x10), not the
        # sensor bit count; the selected bit depth is controlled by capability 2.
        self.frame=Frame();self.frame.requested=0x10;self.frame.reserved=1
        self.check(self.dll.TUCAM_Buf_Alloc(self.handle,C.byref(self.frame)),'分配原始缓冲区');self.allocated=True
        self.check(self.dll.TUCAM_Cap_Start(self.handle,0),'开始采集');self.active=True;self.last_index=None
        self.ensure_manual()
        if self.ready:
            self.set_prop(1,self.meta.exposure_ms);self.set_prop(0,self.meta.gain)
        self.discard=2
    def stop(self):
        if self.handle:
            if self.active:self.dll.TUCAM_Cap_Stop(self.handle);self.active=False
            if self.allocated:self.dll.TUCAM_Buf_Release(self.handle);self.allocated=False
    def configure(self,exposure=None,gain=None,resolution=None,native_bin=None,input_bits=None):
        running=self.active
        if input_bits is not None:
            input_bits=int(input_bits)
            if input_bits not in self.bit_modes:raise ValueError(f'TUCam 不提供 Mono{input_bits}')
        format_change=(resolution is not None and resolution!=self.mode) or (native_bin is not None and native_bin!=self.bin_value) or (input_bits is not None and input_bits!=self.input_bits)
        target_exp=self.meta.exposure_ms if exposure is None else exposure
        target_gain=self.meta.gain if gain is None else gain
        if running and format_change:self.stop()
        try:
            if resolution is not None:self.set_cap(0,resolution);self.mode=self.get_cap(0)
            if native_bin is not None:self.set_cap(0x26,native_bin);self.bin_value=self.get_cap(0x26)
            if input_bits is not None and input_bits!=self.input_bits:
                self.bit_mode=self.bit_modes[input_bits];self.set_cap(2,self.bit_mode);self._read_bit_mode()
            if running and format_change:self.start()
            self.ensure_manual()
            self.set_prop(1,target_exp);self.set_prop(0,target_gain)
            self.exp_range=self.prop_attr(1);self.gain_range=self.prop_attr(0)
            self.meta.exposure_ms=self.get_prop(1);self.meta.gain=self.get_prop(0)
            if abs(self.meta.exposure_ms-target_exp)>max(self.exp_range[2]*1.1,.01):raise RuntimeError('曝光写入后读回不一致')
            if abs(self.meta.gain-target_gain)>max(self.gain_range[2]*1.1,.001):raise RuntimeError('增益写入后读回不一致')
            if resolution is not None and self.mode!=resolution:raise RuntimeError('分辨率设置未被相机接受')
            self.meta.mode=f'res={self.mode};bin={self.bin_value};raw{self.input_bits}'
        finally:
            if running:self.start()
            self.discard=2
    def read(self):
        ret=self.dll.TUCAM_Buf_WaitForFrame(self.handle,C.byref(self.frame),max(3000,int(self.meta.exposure_ms*2+2000)))
        self.check(ret,'等待图像')
        # uiIndex may identify a fixed buffer slot rather than a frame sequence.
        # A successful blocking WaitForFrame is the new-frame event; do not deduplicate by slot.
        self.last_index=self.frame.index
        if self.discard:self.discard-=1;return None
        a=decode_frame(self.frame,self.input_bits);self.meta.bits=self.input_bits
        import re
        label=next((t for v,t in self.resolutions if v==self.mode),'')
        dims=re.search(r'(\d+)\s*[x×]\s*(\d+)',label)
        if dims and a.shape!=(int(dims[2]),int(dims[1])):
            raise RuntimeError(f'相机实际输出 {a.shape[1]}×{a.shape[0]}，与选择的 {label} 不符，已暂停')
        if self.get_cap(3)!=0:raise RuntimeError('检测到相机自行重新开启自动曝光，已暂停采集')
        self.meta.exposure_ms=self.get_prop(1);self.meta.gain=self.get_prop(0)
        return a
    def close(self):
        self.stop()
        if self.handle:self.dll.TUCAM_Dev_Close(self.handle);self.handle=None
        if self.initialized:self.dll.TUCAM_Api_Uninit();self.initialized=False
        if getattr(self,'dir_handle',None):self.dir_handle.close();self.dir_handle=None

class SimCamera:
    def __init__(self,*args):
        self.name='模拟星野 · 非相机数据';self.meta=FrameMeta();self.exp_range=(.018,15000,.001)
        self.gain_range=(1,16,.1);self.resolutions=[(0,'640 × 480'),(1,'1280 × 960'),(2,'5472 × 3648')]
        self.bins=[];self.mode=0;self.bin_value=None;self.scene='星野';self.active=False
        self.bit_options=[(8,'Mono8 · 模拟输入'),(14,'Mono14 · 模拟输入'),(16,'Mono16 · 模拟输入')]
        self.input_bits=16
        self.meta.bits=self.input_bits
        self.rng=np.random.default_rng(42);self.next=0;self.make_scene()
    def make_scene(self):
        w,h=[(640,480),(1280,960),(5472,3648)][self.mode]
        self.meta.mode=f'{w}x{h}';y,x=np.mgrid[-1:1:complex(h),-1:1:complex(w)]
        self.flat=np.maximum(.3,1-.45*(x*x+y*y)).astype(np.float32)
        a=np.zeros((h,w),np.float32)
        for _ in range(220):a[self.rng.integers(5,h-5),self.rng.integers(5,w-5)]=self.rng.uniform(1000,40000)
        self.stars=cv2.GaussianBlur(a,(0,0),.7)+70+180*np.exp(-((y-.3*np.sin(3*x))/.18)**2)
        self.bias=(400+12*np.sin(x*80)).astype(np.float32)
        self.hot=np.zeros((h,w),np.float32)
        for _ in range(50):self.hot[self.rng.integers(h),self.rng.integers(w)]=self.rng.uniform(100,4000)
    def configure(self,exposure=None,gain=None,resolution=None,native_bin=None,input_bits=None):
        if exposure is not None:self.meta.exposure_ms=float(np.clip(exposure,*self.exp_range[:2]))
        if gain is not None:self.meta.gain=float(np.clip(gain,*self.gain_range[:2]))
        if resolution is not None:self.mode=resolution;self.make_scene()
        if input_bits is not None:
            input_bits=int(input_bits)
            if input_bits not in [v for v,_ in self.bit_options]:raise ValueError('模拟器不提供所选输入位深')
            self.input_bits=input_bits;self.meta.bits=input_bits
    def start(self):self.active=True;self.next=time.monotonic()
    def stop(self):self.active=False
    def close(self):self.stop()
    def read(self):
        delay=max(.05,self.meta.exposure_ms/1000)
        now=time.monotonic()
        if self.next>now:time.sleep(self.next-now)
        self.next=time.monotonic()+delay
        signal=self.stars if self.scene=='星野' else (np.full(self.flat.shape,18000,np.float32) if self.scene=='平场光源' else np.zeros_like(self.flat))
        scale=self.meta.exposure_ms/100*self.meta.gain
        image=signal*self.flat*scale+self.bias+self.hot*self.meta.exposure_ms/1000
        noise=self.rng.normal(0,12*np.sqrt(self.meta.gain),image.shape).astype(np.float32)
        return np.clip(image+noise,0,(1<<self.input_bits)-1).astype(np.uint16)

def default_sdk():
    base=Path(sys.executable).parent if getattr(sys,'frozen',False) else Path(__file__).resolve().parent.parent
    candidates=[base/'sdk'/'TUCam.dll',base/'_internal'/'sdk'/'TUCam.dll']
    return str(next((p for p in candidates if p.exists()),candidates[0]))
