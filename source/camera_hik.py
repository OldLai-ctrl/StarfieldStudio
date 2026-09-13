"""Hikrobot USB3/GigE adapter using the installed official MVS Python bindings.

The vendor defines and owns all device/frame structures. No guessed ABI layouts.
Only unpacked Mono formats are accepted; payload and dimensions are checked.
"""
import ctypes as C
import importlib
import os, sys
from pathlib import Path
from core import FrameMeta
from camera_common import VALID_BITS, check_value, check_readback, unpack_mono

MONO={0x01080001:8,0x01100003:10,0x01100005:12,0x01100025:14,0x01100007:16}

def load_mvs(path):
    if os.name!='nt' or C.sizeof(C.c_void_p)!=8:raise RuntimeError('海康接口需要 Windows x64')
    p=Path(path).resolve()
    if p.is_dir():p=p/'MvCameraControl_class.py'
    if not p.is_file() or p.name!='MvCameraControl_class.py':
        raise RuntimeError('请安装海康 MVS 开发包，选择 Development/Samples/Python/MvImport/MvCameraControl_class.py；保留同目录其他接口文件')
    old=sys.modules.get('MvCameraControl_class')
    if old and Path(old.__file__).resolve()!=p:raise RuntimeError('切换海康 SDK 版本后请重启程序')
    handles=[]
    roots=[p.parent,Path(r'C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64'),Path(r'C:\Program Files\Common Files\MVS\Runtime\Win64_x64')]
    for parent in p.parents:
        if parent.name=='MVS':roots.extend([parent/'Runtime/Win64_x64',parent/'Development/Bin/Win64'])
    try:
        for root in roots:
            if root.is_dir():handles.append(os.add_dll_directory(str(root)))
        sys.path.insert(0,str(p.parent))
        try:module=importlib.import_module('MvCameraControl_class')
        finally:sys.path.remove(str(p.parent))
        return module,handles
    except Exception as e:
        for h in handles:h.close()
        raise RuntimeError('海康 MVS 加载失败，请确认已安装完整 x64 运行库和 Python 示例：'+str(e)) from e

class HikCamera:
    def __init__(self,path,index=0):
        self.cam=None;self.created=False;self.opened=False;self.active=False;self.handles=[];self.initialized=False
        try:
            self.sdk,self.handles=load_mvs(path);m=self.sdk
            if hasattr(m.MvCamera,'MV_CC_Initialize'):
                self.check(m.MvCamera.MV_CC_Initialize(),'初始化');self.initialized=True
            devices=m.MV_CC_DEVICE_INFO_LIST()
            self.check(m.MvCamera.MV_CC_EnumDevices(m.MV_GIGE_DEVICE|m.MV_USB_DEVICE,devices),'枚举')
            if not 0<=index<devices.nDeviceNum:raise RuntimeError(f'海康接口发现 {devices.nDeviceNum} 台 USB3/GigE 相机，设备序号无效；请检查 MVS 驱动及连接')
            info=C.cast(devices.pDeviceInfo[index],C.POINTER(m.MV_CC_DEVICE_INFO)).contents
            self.cam=m.MvCamera();self.check(self.cam.MV_CC_CreateHandle(info),'创建句柄');self.created=True
            self.check(self.cam.MV_CC_OpenDevice(),'打开');self.opened=True
            details=info.SpecialInfo.stGigEInfo if info.nTLayerType==m.MV_GIGE_DEVICE else info.SpecialInfo.stUsb3VInfo
            decode=lambda value:bytes(value).split(b'\0',1)[0].decode('utf-8',errors='replace')
            self.name=decode(details.chModelName) or '海康黑白相机';serial=decode(details.chSerialNumber)
            self.meta=FrameMeta(camera='MVS '+self.name+' '+serial)
            if info.nTLayerType==m.MV_GIGE_DEVICE and hasattr(self.cam,'MV_CC_GetOptimalPacketSize'):
                packet=self.cam.MV_CC_GetOptimalPacketSize()
                if packet>0:self.check(self.cam.MV_CC_SetIntValue('GevSCPSPacketSize',packet),'网络包长度')
            pixel=self.enum('PixelFormat')
            supported=list(pixel.nSupportValue)[:pixel.nSupportedNum]
            # Bayer sensors also offer Mono8 conversions; reject a color-capable list.
            if any((v&0xffff) in (8,9,10,11,12,13,14,15,16,17,18,19,0x2a,0x2b,0x2c,0x2d) or (v&0xff000000)==0x02000000 for v in supported):
                raise RuntimeError('海康设备提供彩色/Bayer 格式，当前版本仅接受黑白相机')
            formats=[v for v in supported if v in MONO]
            if not formats:raise RuntimeError('此海康相机未提供受支持的非压缩 Mono8/10/12/14/16 格式')
            self.pixel_by_bits={}
            for value in formats:self.pixel_by_bits.setdefault(MONO[value],value)
            self.bit_options=[(bits,f'Mono{bits} · 原始') for bits in sorted(self.pixel_by_bits)]
            self.input_bits=max(self.pixel_by_bits)
            self.pixel=self.pixel_by_bits[self.input_bits];self.check(self.cam.MV_CC_SetEnumValue('PixelFormat',self.pixel),'原始像素格式')
            if self.enum('PixelFormat').nCurValue!=self.pixel:raise RuntimeError('海康像素格式读回不一致')
            self.meta.bits=self.input_bits
            self.ensure_manual();self.check(self.cam.MV_CC_SetEnumValue('TriggerMode',0),'连续采集')
            self.check(self.cam.MV_CC_SetEnumValue('AcquisitionMode',2),'连续采集模式')
            # Optional frame-rate limiting may be absent. Do not require it for cameras without this node.
            if hasattr(self.cam,'MV_CC_SetBoolValue'):self.cam.MV_CC_SetBoolValue('AcquisitionFrameRateEnable',False)
            self.check(self.cam.MV_CC_SetImageNodeNum(2),'采集缓存')
            self.bins=[];self.bin_value=None;self.mode=0
            self.shapes=[(self.integer('Width').nCurValue,self.integer('Height').nCurValue)]
            self.resolutions=[(0,f'{self.shapes[0][0]} × {self.shapes[0][1]} · 当前画幅')]
            # ROI sizes are computed from real limits and increments, never displayed as binning.
            width,height=self.integer('Width'),self.integer('Height')
            for factor in (1,2,4):
                def aligned(a):return int(a.nMin+max(0,(int(a.nMax)//factor-int(a.nMin)))//max(1,int(a.nInc))*max(1,int(a.nInc)))
                shape=(aligned(width),aligned(height))
                if shape not in self.shapes:
                    self.shapes.append(shape);self.resolutions.append((len(self.shapes)-1,f'{shape[0]} × {shape[1]} · 相机 ROI'))
            self.refresh();self.configure(exposure=min(max(100,self.exp_range[0]),self.exp_range[1]))
        except Exception:self.close();raise
    @staticmethod
    def check(ret,label):
        if ret!=0:raise RuntimeError(f'海康接口 {label} 失败：0x{ret&0xffffffff:08X}')
    def enum(self,name):
        v=self.sdk.MVCC_ENUMVALUE();self.check(self.cam.MV_CC_GetEnumValue(name,v),name);return v
    def integer(self,name):
        v=self.sdk.MVCC_INTVALUE();self.check(self.cam.MV_CC_GetIntValue(name,v),name);return v
    def floating(self,name):
        v=self.sdk.MVCC_FLOATVALUE();self.check(self.cam.MV_CC_GetFloatValue(name,v),name);return v
    def ensure_manual(self):
        for name in ('ExposureAuto','GainAuto'):
            self.check(self.cam.MV_CC_SetEnumValue(name,0),name)
            if self.enum(name).nCurValue!=0:raise RuntimeError('海康自动曝光或自动增益未关闭')
    def refresh(self):
        exposure,gain=self.floating('ExposureTime'),self.floating('Gain')
        self.exp_range=(exposure.fMin/1000,exposure.fMax/1000,.001);self.gain_range=(gain.fMin,gain.fMax,.001)
        self.meta.exposure_ms=exposure.fCurValue/1000;self.meta.gain=gain.fCurValue
        self.shape=(self.integer('Height').nCurValue,self.integer('Width').nCurValue)
        self.meta.mode=f'MVS;Mono{self.meta.bits};shape={self.shape};res={self.mode}'
        size=self.integer('PayloadSize').nCurValue
        if not 0<size<=512*1024**2:raise RuntimeError('海康帧大小无效')
        self.buffer=(C.c_ubyte*size)();self.payload=size
    def start(self):
        if self.active:return
        self.ensure_manual();self.check(self.cam.MV_CC_StartGrabbing(),'开始采集');self.active=True;self.discard=2
    def stop(self):
        if self.active:self.check(self.cam.MV_CC_StopGrabbing(),'停止采集');self.active=False
    def configure(self,exposure=None,gain=None,resolution=None,native_bin=None,input_bits=None):
        if native_bin is not None:raise ValueError('此海康接口使用软件 Binning')
        if input_bits is not None:
            input_bits=int(input_bits)
            if input_bits not in self.pixel_by_bits:raise ValueError(f'海康相机不提供 Mono{input_bits}')
        was=self.active
        if was:self.stop()
        if input_bits is not None and input_bits!=self.input_bits:
            pixel=self.pixel_by_bits[input_bits]
            self.check(self.cam.MV_CC_SetEnumValue('PixelFormat',pixel),'原始像素格式')
            actual=self.enum('PixelFormat').nCurValue
            if actual!=pixel:raise RuntimeError('海康像素位深读回不一致')
            self.pixel=pixel;self.input_bits=input_bits;self.meta.bits=input_bits
        if resolution is not None:
            if not 0<=resolution<len(self.shapes):raise ValueError('无效画幅')
            for name in ('OffsetX','OffsetY'):
                self.check(self.cam.MV_CC_SetIntValue(name,0),name)
            w,h=self.shapes[resolution]
            self.check(self.cam.MV_CC_SetIntValue('Width',w),'画幅宽度');self.check(self.cam.MV_CC_SetIntValue('Height',h),'画幅高度')
            if (self.integer('Width').nCurValue,self.integer('Height').nCurValue)!=(w,h):raise RuntimeError('海康画幅读回不一致')
            self.mode=resolution
        self.refresh();self.ensure_manual()
        for name,target,limits,scale in [('ExposureTime',exposure,self.exp_range,1000),('Gain',gain,self.gain_range,1)]:
            if target is None:continue
            target=check_value(target,limits,name);self.check(self.cam.MV_CC_SetFloatValue(name,target*scale),name)
            check_readback(target,self.floating(name).fCurValue/scale,limits[2],name)
        self.refresh()
        if was:self.start()
    def read(self):
        info=self.sdk.MV_FRAME_OUT_INFO_EX()
        self.check(self.cam.MV_CC_GetOneFrameTimeout(self.buffer,self.payload,info,max(3000,int(self.meta.exposure_ms*2+2000))),'等待图像')
        if info.enPixelType!=self.pixel or (info.nHeight,info.nWidth)!=self.shape:raise RuntimeError('海康实际帧格式或尺寸与设置不符')
        if self.discard:self.discard-=1;return None
        if self.enum('ExposureAuto').nCurValue!=0 or self.enum('GainAuto').nCurValue!=0:raise RuntimeError('海康自动参数状态发生变化')
        self.meta.exposure_ms=self.floating('ExposureTime').fCurValue/1000;self.meta.gain=self.floating('Gain').fCurValue
        return unpack_mono(self.buffer,info.nWidth,info.nHeight,self.meta.bits,info.nFrameLen)
    def close(self):
        try:
            if self.cam:
                try:self.stop()
                finally:
                    if self.opened:self.cam.MV_CC_CloseDevice();self.opened=False
                    if self.created:self.cam.MV_CC_DestroyHandle();self.created=False
        finally:
            if self.initialized:self.sdk.MvCamera.MV_CC_Finalize();self.initialized=False
            for h in self.handles:h.close()
            self.handles=[]
