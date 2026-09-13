import numpy as np
import pytest
from processing import *
from worker import CaptureWorker
from core import FrameMeta

def test_auto_priority_and_secondary_control():
    args=(100,1,1000,10000,(5,5000,.01),(1,257,1))
    e,g,_=auto_step(*args,'优先曝光时长');assert e>100 and g==1
    e,g,_=auto_step(*args,'优先增益');assert e==100 and g>1
    e,g,_=auto_step(5000,1,1000,10000,(5,5000,.01),(1,257,1),'优先曝光时长');assert e==5000 and g>1
    e,g,_=auto_step(100,257,1000,10000,(5,5000,.01),(1,257,1),'优先增益');assert e>100 and g==257
    assert auto_step(*args,'手动')[:2]==(100,1)

def test_calibrated_auto_never_uses_unmatched_parameters():
    e,g,_=auto_step(100,1,1000,10000,(5,5000,.01),(1,257,1),'优先增益',[(100,1),(100,16),(200,1)])
    assert (e,g)==(100,16)
    e,g,_=auto_step(100,1,1000,10000,(5,5000,.01),(1,257,1),'优先增益',[(100,1)])
    assert (e,g)==(100,1)

def test_roi_measures_actual_selected_pixels():
    image=np.zeros((100,100),np.uint16);image[:,:50]=1000;image[:,50:]=50000
    assert measure_brightness(image,(0,0,.5,1))==1000
    assert measure_brightness(image,(.5,0,.5,1))==50000
    with pytest.raises(ValueError):validate_roi((-.1,0,.5,1))

def test_palette_absolute_gray_values_and_clamped_endpoints():
    pts=[[0,'#000000'],[100,'#ff0000'],[200,'#ffffff']]
    rgb=custom_color(np.array([[-10,0,100,200,500]],np.float32),pts)
    np.testing.assert_array_equal(rgb[0,0],[0,0,0]);np.testing.assert_array_equal(rgb[0,2],[255,0,0]);np.testing.assert_array_equal(rgb[0,4],[255,255,255])
    with pytest.raises(ValueError):validate_points([[0,'#000000'],[0,'#ffffff']])

def test_pixel_math_is_local_and_respects_gray_mask():
    a=np.arange(16,dtype=np.float32).reshape(4,4)
    settings=dict(math_op='加常数',math_roi=(.5,0,.5,1),math_low=5,math_high=10,math_k=100)
    out=pixel_math(a,settings);expected=a.copy();expected[1,2:4]+=100;expected[2,2]+=100
    np.testing.assert_array_equal(out,expected)
    assert pixel_math(a,{'math_op':'关闭'}) is a

def test_window_reference_and_nonlinear_math_preserve_values():
    a=np.array([[-4,0],[4,9]],np.float32)
    np.testing.assert_array_equal(pixel_math(a,{'math_op':'减去窗口平均'},np.ones_like(a)),a-1)
    np.testing.assert_array_equal(pixel_math(a,{'math_op':'与参考帧平均'},reference=np.ones_like(a)),(a+1)/2)
    np.testing.assert_array_equal(pixel_math(a,{'math_op':'平方根','math_scale':1}),[[-2,0],[2,3]])
    assert np.isfinite(pixel_math(a,{'math_op':'对数'})).all()

def test_local_window_replaces_only_selected_region():
    w=CaptureWorker();w.meta=FrameMeta();w.raw=np.ones((4,4),np.uint16);w.single=np.ones((4,4),np.float32)
    w.settings.update(mode='关闭',math_op='窗口积分',math_roi=(.5,.5,.5,.5));w.window_local=True
    w.roll.push(np.full((2,2),3,np.float32),1);w.roll.push(np.full((2,2),5,np.float32),2)
    w.publish();expected=np.ones((4,4),np.float32);expected[2:,2:]=8
    np.testing.assert_array_equal(w.processed,expected)
