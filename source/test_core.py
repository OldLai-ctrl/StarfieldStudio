import ctypes as C
import struct, json, time, queue
from dataclasses import replace
import numpy as np
import pytest
import tifffile
from core import *
from camera import Frame, decode_frame, SimCamera
from camera_common import validate_native_frame
from worker import CaptureWorker
from compute import ComputeEngine, DEVICE_CHOICES

def test_compute_engine_cpu_mode_and_probe_are_serializable():
    engine=ComputeEngine('CPU')
    status=engine.status()
    assert DEVICE_CHOICES[1]=='CPU'
    assert status['requested']=='CPU' and status['kind']=='cpu' and not engine.use_gpu
    info=ComputeEngine.probe()
    assert set(('opencl','label','vendor','name'))<=set(info)
    assert isinstance(info['opencl'],bool)

def test_opencl_rolling_accumulation_when_runtime_is_available():
    engine=ComputeEngine('GPU（OpenCL）')
    if not engine.use_gpu:
        pytest.skip('当前测试机没有可用 OpenCL GPU')
    r=RollingIntegrator(window_unit='帧数',frame_limit=3,memory_mb=16,backend=engine)
    for i in range(1,5):r.push(np.full((3,4),i,np.float32),i)
    assert len(r.frames)==3 and r.gpu
    np.testing.assert_allclose(r.result('积分'),np.full((3,4),9,np.float32))
    np.testing.assert_allclose(r.result('平均'),np.full((3,4),3,np.float32))

def test_opencl_engine_can_fall_back_without_losing_window():
    engine=ComputeEngine('GPU（OpenCL）')
    if not engine.use_gpu:
        pytest.skip('当前测试机没有可用 OpenCL GPU')
    r=RollingIntegrator(window_unit='帧数',frame_limit=2,memory_mb=16,backend=engine)
    r.push(np.full((2,2),4,np.float32),1)
    original_add=engine.add
    engine.add=lambda left,right: (_ for _ in ()).throw(RuntimeError('driver reset'))
    try:r.push(np.full((2,2),5,np.float32),2)
    finally:engine.add=original_add
    assert not r.gpu and r.backend.kind=='cpu' and len(r.frames)==2
    np.testing.assert_allclose(r.result('平均'),np.full((2,2),4.5,np.float32))

def test_rolling_maximum_keeps_latest_window_only():
    r=RollingIntegrator(window_unit='帧数',frame_limit=3,memory_mb=16)
    r.set_max_enabled(True)
    values=[[[1,8],[4,2]],[[3,5],[2,7]],[[2,6],[9,1]],[[0,4],[6,10]]]
    for i,a in enumerate(values):r.push(np.asarray(a,np.float32),i)
    assert len(r.frames)==3
    np.testing.assert_allclose(r.result('最大值'),[[3,6],[9,10]])
    assert r.total_bytes>r.total_nbytes

def test_gpu_rolling_maximum_when_opencl_is_available():
    engine=ComputeEngine('GPU（OpenCL）')
    if not engine.use_gpu:
        pytest.skip('当前测试机没有可用 OpenCL GPU')
    r=RollingIntegrator(window_unit='帧数',frame_limit=3,memory_mb=16,backend=engine)
    r.set_max_enabled(True)
    for i in range(4):r.push(np.full((2,2),i,np.float32),i)
    np.testing.assert_allclose(r.result('最大值'),np.full((2,2),3,np.float32))

def test_exact_rolling_window():
    r=RollingIntegrator(2)
    for t,v in [(0,1),(.5,3),(1.9,8),(2,12)]:r.push(np.full((2,3),v),t)
    assert [t for t,a in r.frames]==[.5,1.9,2]
    np.testing.assert_allclose(r.result(),23/3)
    np.testing.assert_allclose(r.result('积分'),23)
    r.push(np.full((2,3),7),9);assert len(r.frames)==1
    np.testing.assert_allclose(r.result(),7)
    with pytest.raises(ValueError):r.push(np.zeros((2,3)),9)

def test_frame_count_window_keeps_exact_latest_frames():
    r=RollingIntegrator(window_unit='帧数',frame_limit=3,memory_mb=16)
    for i in range(6):r.push(np.full((2,2),i,np.float32),i)
    assert [int(a[0,0]) for _,a in r.frames]==[3,4,5]
    np.testing.assert_allclose(r.result('平均'),4)
    r.set_window('时间',seconds=2,frames=3);r.clear()
    for t in (0,.5,2.5):r.push(np.ones((1,1)),t)
    assert len(r.frames)==1

def test_memory_budget_no_silent_window_shortening():
    r=RollingIntegrator(10,memory_mb=.001)
    with pytest.raises(MemoryError):r.push(np.ones((100,100)),1)

def test_auto_budget_expands_without_losing_window(monkeypatch):
    import core
    monkeypatch.setattr(core,'physical_memory',lambda:(32*1024**3,12*1024**3))
    r=RollingIntegrator(3,0);r.limit=1
    for i in range(40):r.push(np.full((10,10),i,np.float32),i/10)
    assert r.auto and r.limit>1
    expected=list(range(10,40))
    assert len(r.frames)==len(expected)
    np.testing.assert_allclose(r.result(),np.mean(expected))

def test_auto_budget_preserves_system_reserve(monkeypatch):
    import core
    monkeypatch.setattr(core,'physical_memory',lambda:(8*1024**3,1024**3))
    r=RollingIntegrator(3,0);r.limit=1
    with pytest.raises(MemoryError,match='物理内存不足'):r.push(np.ones((10,10)),1)

def test_calibration_dark_does_not_double_subtract_bias():
    m=FrameMeta();lib=CalibrationLibrary();a=np.ones((4,4),np.float32)
    lib.add(Master('bias',a*10,m));lib.add(Master('dark',a*15,m));lib.add(Master('flat',a*2,m))
    np.testing.assert_allclose(lib.correct(a*25,m,True,True,True),5)
    np.testing.assert_allclose(lib.correct(a*5,m,False,True),-5)
    with pytest.raises(CalibrationError):lib.correct(a,replace(m,exposure_ms=200),dark=True)
    with pytest.raises(CalibrationError):lib.correct(a,replace(m,gain=2),bias=True)

def test_flat_requires_offset_and_normalizes(tmp_path):
    lib=CalibrationLibrary();m=FrameMeta();a=np.array([[100,200],[300,400]],np.float32)
    with pytest.raises(CalibrationError):lib.make_flat(a,m,10)
    lib.add(Master('bias',np.full((2,2),10,np.float32),m))
    f=lib.make_flat(a+10,m,10);np.testing.assert_allclose(f.image,a/250)
    lib.add(f);p=tmp_path/'库.npz';lib.save(p);copy=CalibrationLibrary();copy.load(p)
    np.testing.assert_allclose(copy.masters[-1].image,f.image)
    with pytest.raises(CalibrationError):lib.make_flat(np.full((2,2),65535),m,10)

def test_target_exposure_and_dark_steps():
    assert exposure_target(100,10,40,5,5000)==200
    assert exposure_target(100,101,100,5,5000)==100
    assert exposure_target(100,10,40,5,5000,[50,100,200,400])==200
    assert exposure_target(4000,1,100,5,5000)==5000

def test_binning_float_precision():
    a=np.arange(16,dtype=np.float32).reshape(4,4)
    np.testing.assert_allclose(bin_image(a,2),[[2.5,4.5],[10.5,12.5]])

def test_processed_histogram_keeps_negative_and_integral_overflow():
    a=np.array([[-1000,0,100],[65535,100000,200000]],np.float32)
    counts,stats=histogram(a,'processed')
    assert counts.sum()==a.size
    assert stats['hist_low']<=-1000 and stats['hist_high']>200000
    counts,stats=histogram(np.array([[0,127,255]],np.uint8),'display')
    assert counts.sum()==3 and stats['hist_high']==256

def test_switch_sum_mean_preserves_window_and_actual_exposure_scale():
    w=CaptureWorker();w.meta=FrameMeta(exposure_ms=200);w.raw=np.full((4,4),200,np.uint16)
    w.roll.push(np.full((4,4),100,np.float32),1)
    w.roll.push(np.full((4,4),100,np.float32),2)
    w.publish();np.testing.assert_equal(w.processed,200)
    w.command('settings',{'values':{'mode':'积分'}})
    assert len(w.roll.frames)==2;np.testing.assert_equal(w.processed,400)
    assert w.latest['mode']=='积分'
    w.command('settings',{'values':{'mode':'平均'}})
    assert len(w.roll.frames)==2;np.testing.assert_equal(w.processed,200)

def test_settings_ack_does_not_rewrite_other_controls():
    w=CaptureWorker();w.command('settings',{'values':{'mode':'积分'}})
    e=w.events.get_nowait();assert e['values']=={'mode':'积分'}

def test_raw_offset_padding_and_reject_8bit():
    b=(C.c_uint8*100)();f=Frame();f.header=8;f.offset=16;f.width=3;f.height=2;f.stride=8
    f.depth=16;f.element=2;f.channels=1;f.buffer=C.addressof(b)
    a=np.ndarray((2,3),dtype=np.uint16,buffer=b,offset=16,strides=(8,2));a[:]=[[0,1024,65535],[42,100,4095]]
    np.testing.assert_equal(decode_frame(f),a)
    f.depth=8
    with pytest.raises(RuntimeError):decode_frame(f)
    f.depth=2;f.format=0x10
    with pytest.raises(RuntimeError):decode_frame(f)
    np.testing.assert_equal(decode_frame(f,16),a)
    b=(C.c_uint8*32)();g=Frame();g.header=8;g.offset=8;g.width=3;g.height=2;g.stride=4;g.depth=8;g.format=0x10;g.element=1;g.channels=1;g.buffer=C.addressof(b)
    np.ndarray((2,3),dtype=np.uint8,buffer=b,offset=8,strides=(4,1))[:]=[[0,127,255],[4,8,16]]
    np.testing.assert_equal(decode_frame(g,8),[[0,127,255],[4,8,16]])
    with pytest.raises(RuntimeError,match='超过'):
        validate_native_frame(np.array([[4096]],np.uint16),12)

def test_simulator_exposes_switchable_input_bits():
    c=SimCamera();c.configure(input_bits=14);a=c.read()
    assert c.input_bits==c.meta.bits==14 and int(a.max())<=16383
    c.configure(input_bits=8);a=c.read();assert int(a.max())<=255
    assert C.sizeof(Frame)==56

def test_ser_and_fits_readback(tmp_path):
    a=np.array([[0,32768],[65535,1]],np.uint16);p=tmp_path/'video.ser';s=SerWriter(p,a.shape)
    s.append(a);s.append(a);s.close();b=p.read_bytes()
    assert len(b)==178+16;assert struct.unpack_from('<I',b,38)[0]==2
    np.testing.assert_equal(np.frombuffer(b,dtype='<u2',offset=178).reshape(2,2,2)[0],a)
    p=tmp_path/'frame.fits';im=a.astype(np.float32)-100;write_fits(p,im,FrameMeta())
    b=p.read_bytes();assert len(b)%2880==0
    np.testing.assert_equal(np.frombuffer(b,dtype='>f4',offset=2880,count=4).reshape(2,2),im)

def wait_for(worker,predicate,timeout=6):
    start=time.monotonic()
    while time.monotonic()-start<timeout:
        if predicate():return
        time.sleep(.025)
    raise AssertionError('Timed out')

def test_worker_end_to_end(tmp_path):
    w=CaptureWorker();w.start()
    try:
        w.send('connect',sim=True,dll='')
        wait_for(w,lambda:w.latest is not None)
        assert w.raw.dtype==np.uint16
        w.send('settings',values={'seconds':.2})
        time.sleep(.6);assert w.latest['span']<.2
        w.send('master',master_kind='bias',count=3)
        wait_for(w,lambda:any(m.kind=='bias' for m in w.library.masters))
        w.send('master',master_kind='dark',count=3)
        wait_for(w,lambda:any(m.kind=='dark' for m in w.library.masters))
        w.send('master',master_kind='flat',count=3)
        wait_for(w,lambda:any(m.kind=='flat' for m in w.library.masters))
        w.send('settings',values={'dark':True,'bias':True,'flat':True})
        before=w.serial;wait_for(w,lambda:w.serial>before+2)
        assert not w.paused
        p=tmp_path/'raw.tif';w.send('save',path=str(p),format='raw');wait_for(w,p.exists)
        assert tifffile.imread(p).dtype==np.uint16
        p=tmp_path/'record.ser';w.send('record',path=str(p));time.sleep(.4);w.send('record')
        wait_for(w,lambda:w.rec is None);assert struct.unpack_from('<I',p.read_bytes(),38)[0]>=2
        w.send('pause');wait_for(w,lambda:w.paused);n=w.serial;time.sleep(.3);assert w.serial==n
        w.send('pause');wait_for(w,lambda:w.serial>n)
        w.send('settings',values={'gain':2})
        wait_for(w,lambda:w.paused)
        errors=[]
        while not w.events.empty():
            e=w.events.get_nowait()
            if e['kind']=='error':errors.append(e['text'])
        assert len(errors)==1 and '匹配' in errors[0]
    finally:
        w.send('quit');w.join(5);assert not w.is_alive()
