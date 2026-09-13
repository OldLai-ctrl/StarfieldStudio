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

def test_camera_raw_curve_and_point_curve_are_bounded_and_continuous():
    x=np.linspace(0,1,257,dtype=np.float32)
    np.testing.assert_allclose(apply_tone_curve(x,'关闭'),x)
    param=apply_tone_curve(x,'Camera Raw 参数曲线',(65,-30,25,-70))
    points=apply_tone_curve(x,'点曲线',points=((0.,0.),(.25,.1),(.5,.7),(.75,.8),(1.,1.)))
    assert param.min()>=0 and param.max()<=1 and points.min()>=0 and points.max()<=1
    assert param[1] != x[1] and points[64] < x[64]

def test_display_adjustments_change_preview_without_touching_input():
    from core import display_rgb
    image=np.tile(np.linspace(0,1000,64,dtype=np.float32),(64,1))
    original=image.copy()
    base=display_rgb(image,white=1000)
    adjusted=display_rgb(image,white=1000,contrast=80,sharpen=180,sharpen_radius=2,
                         curve_mode='Camera Raw 参数曲线',curve_params=(30,0,0,-20))
    assert not np.array_equal(base,adjusted)
    np.testing.assert_array_equal(image,original)

def test_live_denoise_is_low_cost_and_preview_only():
    image=np.full((21,21),100,dtype=np.float32);image[10,10]=10000;original=image.copy()
    median=apply_denoise(image,'中值 3×3（去孤立噪点）',100)
    soft=apply_denoise(image,'中值 3×3（去孤立噪点）',25)
    gaussian=apply_denoise(image,'高斯 3×3（轻度平滑）',50)
    assert median[10,10]==100 and 100<soft[10,10]<10000 and gaussian[10,10]<10000
    np.testing.assert_array_equal(image,original)
    np.testing.assert_array_equal(apply_denoise(image,'关闭',100),image)

def test_local_window_replaces_only_selected_region():
    w=CaptureWorker();w.meta=FrameMeta();w.raw=np.ones((4,4),np.uint16);w.single=np.ones((4,4),np.float32)
    w.settings.update(mode='关闭',math_op='窗口积分',math_roi=(.5,.5,.5,.5));w.window_local=True
    w.roll.push(np.full((2,2),3,np.float32),1);w.roll.push(np.full((2,2),5,np.float32),2)
    w.publish();expected=np.ones((4,4),np.float32);expected[2:,2:]=8
    np.testing.assert_array_equal(w.processed,expected)
