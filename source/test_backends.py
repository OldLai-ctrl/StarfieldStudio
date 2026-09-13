"""Contract tests use fake SDK devices; these are not physical camera certification."""
import ctypes as C
from types import SimpleNamespace as NS
import numpy as np
import pytest
from camera_common import unpack_mono, open_camera
from core import histogram
import camera_hik as hik
import camera_toup as toup

@pytest.mark.parametrize('bits',[8,10,12,14,16])
def test_native_levels_are_not_scaled_or_truncated(bits):
    values=np.array([0,1,(1<<bits)-1],dtype=np.uint8 if bits==8 else '<u2')
    result=unpack_mono(values.tobytes(),3,1,bits,values.nbytes)
    assert result.dtype==np.uint16
    np.testing.assert_array_equal(result[0],values)
    counts,stats=histogram(result,bits=bits)
    assert counts.sum()==3 and stats['hist_high']==1<<bits
    assert stats['saturation']==pytest.approx(100/3)

def test_reject_short_padded_and_unknown_frames():
    for raw,bits,length in [(b'\0',16,1),(b'\0'*9,16,9),(b'\0'*8,9,8)]:
        with pytest.raises(RuntimeError):unpack_mono(raw,2,2,bits,length)
    with pytest.raises(ValueError):open_camera('unregistered','',0)

class FakeHik:
    def __init__(self):
        self.ints={'Width':8,'Height':6,'PayloadSize':96,'OffsetX':0,'OffsetY':0}
        self.enums={'PixelFormat':0x01100005,'ExposureAuto':1,'GainAuto':1,'TriggerMode':1,'AcquisitionMode':2}
        self.floats={'ExposureTime':100000.,'Gain':0.}
        self.active=False;self.closed=False;self.destroyed=False;self.ignore=None;self.bad_frame=False
    def MV_CC_CreateHandle(self,info):return 0
    def MV_CC_OpenDevice(self):return 0
    def MV_CC_CloseDevice(self):self.closed=True;return 0
    def MV_CC_DestroyHandle(self):self.destroyed=True;return 0
    def MV_CC_GetEnumValue(self,k,v):
        v.nCurValue=self.enums[k];v.nSupportValue=[0x01080001,0x01100005] if k=='PixelFormat' else [0,1,2];v.nSupportedNum=len(v.nSupportValue);return 0
    def MV_CC_SetEnumValue(self,k,v):self.enums[k]=v;return 0
    def MV_CC_GetIntValue(self,k,v):
        if k=='PayloadSize':self.ints[k]=self.ints['Width']*self.ints['Height']*2
        v.nCurValue=self.ints[k];v.nMin=2;v.nInc=2;v.nMax={'Width':8,'Height':6}.get(k,1000);return 0
    def MV_CC_SetIntValue(self,k,v):self.ints[k]=v;return 0
    def MV_CC_GetFloatValue(self,k,v):
        v.fCurValue=self.floats[k];v.fMin=5000 if k=='ExposureTime' else 0;v.fMax=5000000 if k=='ExposureTime' else 24;return 0
    def MV_CC_SetFloatValue(self,k,v):
        if k!=self.ignore:self.floats[k]=v
        return 0
    def MV_CC_SetBoolValue(self,*args):return 0
    def MV_CC_SetImageNodeNum(self,*args):return 0
    def MV_CC_StartGrabbing(self):self.active=True;return 0
    def MV_CC_StopGrabbing(self):self.active=False;return 0
    def MV_CC_GetOneFrameTimeout(self,buffer,size,info,timeout):
        assert self.active and timeout>=3000
        info.nWidth=self.ints['Width'];info.nHeight=self.ints['Height'];info.nFrameLen=info.nWidth*info.nHeight*2
        info.enPixelType=0x01080009 if self.bad_frame else self.enums['PixelFormat']
        a=np.ctypeslib.as_array(buffer).view('<u2');a[:]=4095;return 0

def hik_fixture(monkeypatch):
    device_type=type('Device',(C.Structure,),{'_fields_':[('unused',C.c_int)]})
    device=device_type();pointer=C.pointer(device);cam=FakeHik()
    details=NS(chModelName=b'Test mono',chSerialNumber=b'TEST')
    # cast returns a vendor device info object in the adapter, not in actual SDK frame decoding.
    original_cast=C.cast
    monkeypatch.setattr(hik.C,'cast',lambda p,t:NS(contents=NS(nTLayerType=4,SpecialInfo=NS(stUsb3VInfo=details))) if p is pointer else original_cast(p,t))
    class Factory:
        @staticmethod
        def MV_CC_EnumDevices(mask,v):v.nDeviceNum=1;v.pDeviceInfo=[pointer];return 0
        def __new__(cls):return cam
    sdk=NS(MvCamera=Factory,MV_CC_DEVICE_INFO_LIST=NS,MV_CC_DEVICE_INFO=device_type,MV_GIGE_DEVICE=1,MV_USB_DEVICE=4,
        MVCC_ENUMVALUE=NS,MVCC_INTVALUE=NS,MVCC_FLOATVALUE=NS,MV_FRAME_OUT_INFO_EX=NS)
    monkeypatch.setattr(hik,'load_mvs',lambda path:(sdk,[]))
    return cam

def test_hik_connect_controls_roi_raw_and_cleanup(monkeypatch):
    fake=hik_fixture(monkeypatch);cam=hik.HikCamera('fake')
    try:
        assert cam.meta.bits==12 and cam.meta.gain==0
        assert fake.enums['ExposureAuto']==fake.enums['GainAuto']==0
        cam.start();cam.configure(exposure=5000,gain=12,resolution=1)
        assert cam.meta.exposure_ms==5000 and cam.meta.gain==12 and fake.active
        cam.read();cam.read();frame=cam.read()
        assert frame.shape==cam.shape and frame.max()==4095
        fake.bad_frame=True
        with pytest.raises(RuntimeError,match='格式'):cam.read()
    finally:cam.close()
    assert fake.closed and fake.destroyed and not fake.active

def test_hik_rejects_failed_write_and_closes_failed_open(monkeypatch):
    fake=hik_fixture(monkeypatch);cam=hik.HikCamera('fake');cam.start();fake.ignore='Gain'
    try:
        with pytest.raises(RuntimeError,match='读回'):cam.configure(gain=12)
        assert not fake.active
    finally:cam.close()
    with pytest.raises(RuntimeError,match='序号'):hik.HikCamera('fake',5)

class Function:
    def __init__(self,fn):self.fn=fn
    def __call__(self,*args):return self.fn(*args)

class FakeToup:
    def __init__(self):
        self.model=toup.Model();self.model.flag=0x10;self.closed=False;self.active=False;self.mode=0;self.auto=1
        self.exp=100000;self.gain=100;self.options={};self.ignore=False;self.calls=[]
    def __getattr__(self,name):
        def fn(*a):
            op=name.removeprefix('Toupcam_');self.calls.append(op)
            if op=='EnumV2':a[0][0].displayname='Test mono';a[0][0].id='test';a[0][0].model=C.pointer(self.model);return 1
            if op=='Open':return 123
            if op=='Close':self.closed=True;return
            if op=='get_MaxBitDepth':return 12
            if op=='get_ResolutionNumber':return 2
            if op=='get_MonoMode':return 0
            if op=='put_Option':self.options[a[1]]=a[2]
            elif op=='put_AutoExpoEnable':self.auto=a[1]
            elif op=='get_AutoExpoEnable':a[1]._obj.value=self.auto
            elif op=='get_ExpTimeRange':a[1]._obj.value=5000;a[2]._obj.value=5000000
            elif op=='get_ExpoAGainRange':a[1]._obj.value=100;a[2]._obj.value=1000
            elif op=='get_RawFormat':a[1]._obj.value=0x59595959;a[2]._obj.value=12
            elif op=='get_ExpoTime':a[1]._obj.value=self.exp
            elif op=='get_ExpoAGain':a[1]._obj.value=self.gain
            elif op=='put_ExpoTime':self.exp=a[1]
            elif op=='put_ExpoAGain':
                if not self.ignore:self.gain=a[1]
            elif op=='get_eSize':a[1]._obj.value=self.mode
            elif op=='put_eSize':self.mode=a[1]
            elif op in ('get_Size','get_Resolution'):
                mode=self.mode if op=='get_Size' else a[1];a[-2]._obj.value=8//(mode+1);a[-1]._obj.value=6//(mode+1)
            elif op=='StartPullModeWithCallback':self.active=True
            elif op=='Stop':self.active=False
            elif op=='WaitImageV3':
                assert self.active and a[5]==-1
                info=a[-1]._obj;info.width=8//(self.mode+1);info.height=6//(self.mode+1)
                np.ctypeslib.as_array((C.c_uint16*(info.width*info.height)).from_address(a[2]))[:]=4095
            return 0
        return Function(fn)

def test_toup_raw_lifecycle_and_controls(monkeypatch,tmp_path):
    fake=FakeToup();monkeypatch.setattr(toup.C,'WinDLL',lambda path:fake)
    dll=tmp_path/'toupcam.dll';dll.touch();cam=toup.ToupCamera(str(dll))
    try:
        assert fake.options[4]==1 and fake.options[6]==1 and fake.auto==0
        cam.start();cam.configure(exposure=5,gain=200,resolution=1)
        assert cam.shape==(3,4) and cam.meta.bits==12 and cam.meta.exposure_ms==5
        cam.read();cam.read();frame=cam.read();assert frame.max()==4095 and frame.dtype==np.uint16
        fake.ignore=True
        with pytest.raises(RuntimeError,match='读回'):cam.configure(gain=300)
        assert not fake.active
    finally:cam.close()
    assert fake.closed

def test_toup_rejects_color_before_open(monkeypatch,tmp_path):
    fake=FakeToup();fake.model.flag=0;monkeypatch.setattr(toup.C,'WinDLL',lambda path:fake)
    dll=tmp_path/'toupcam.dll';dll.touch()
    with pytest.raises(RuntimeError,match='彩色'):toup.ToupCamera(str(dll))
    assert 'Open' not in fake.calls

def test_backend_selection_and_config_roundtrip(monkeypatch,tmp_path):
    from PySide6.QtWidgets import QApplication,QFileDialog
    from main import MainWindow
    import json
    app=QApplication.instance() or QApplication([]);window=MainWindow()
    try:
        window.backend.setCurrentIndex(window.backend.findData('hik'));window.dll.setText('C:/test/MvCameraControl_class.py')
        path=str(tmp_path/'settings.json');monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**kw:(path,''));window.save_config()
        assert json.loads((tmp_path/'settings.json').read_text(encoding='utf-8'))['backend']=='hik'
        window.backend.setCurrentIndex(0)
        monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a,**kw:(path,''));window.load_config()
        assert window.backend.currentData()=='hik' and window.dll.text().endswith('.py')
    finally:
        window.worker.send('quit');window.worker.join(2);window.close();app.processEvents()
