"""Bounded metering, camera auto-control, lookup colors and optional local pixel math."""
from functools import lru_cache
import math
import numpy as np
import cv2

WINDOW_OPS={'窗口平均','窗口积分','减去窗口平均','加上窗口平均'}
REF_OPS={'减去参考帧','加上参考帧','与参考帧平均'}
MATH_OPS=['关闭','加常数','乘系数','窗口平均','窗口积分','减去窗口平均','加上窗口平均',
          '减去参考帧','加上参考帧','与参考帧平均','幂律','对数','平方根','绝对值','限制范围']
DENOISE_MODES=('关闭','中值 3×3（去孤立噪点）','高斯 3×3（轻度平滑）')
LOWLIGHT_MODES=('关闭','自适应弱光提亮','弱光提亮 + 轻度降噪','星点/流星保护')
DEFAULT_POINTS=[[0,'#000000'],[1000,'#143d8f'],[4000,'#23cda8'],[16000,'#ffe56c'],[65535,'#ffffff']]
DEFAULT_CURVE_POINTS=[[0.,0.],[.25,.25],[.5,.5],[.75,.75],[1.,1.]]
CURVE_MODES=('关闭','Camera Raw 参数曲线','点曲线')

def validate_roi(roi):
    if roi is None:return None
    if not isinstance(roi,(list,tuple)) or len(roi)!=4:raise ValueError('选区格式无效')
    x,y,w,h=map(float,roi)
    if not all(math.isfinite(v) for v in (x,y,w,h)) or x<0 or y<0 or w<=0 or h<=0 or x+w>1.000001 or y+h>1.000001:
        raise ValueError('选区必须在画面内，且宽高大于零')
    return (x,y,w,h)

def roi_slices(shape,roi):
    h,w=shape
    if roi is None:return slice(0,h),slice(0,w)
    x,y,rw,rh=validate_roi(roi);x0=min(w-1,int(x*w));y0=min(h-1,int(y*h))
    return slice(y0,min(h,max(y0+1,int((y+rh)*h)))),slice(x0,min(w,max(x0+1,int((x+rw)*w))))

def measure_brightness(raw,roi=None):
    area=raw[roi_slices(raw.shape,roi)]
    stride=max(1,int(np.ceil(np.sqrt(area.size/200000))))
    sample=area[::stride,::stride]
    return float(np.percentile(sample,90))

def auto_step(exposure,gain,measured,target,exp_range,gain_range,priority,allowed_pairs=None,allowed_gains=None,gain_slope=None,lock_exposure=False,lock_gain=False):
    if priority=='手动':return exposure,gain,'手动'
    if abs(measured-target)<=max(1,target*.08):return exposure,gain,'目标范围内'
    direction=1 if measured<target else -1
    elo,ehi,estep=exp_range;glo,ghi,gstep=gain_range
    etol=max(.01,abs(estep)*.55);gtol=max(.0001,abs(gstep)*.55)
    desired_exp=float(np.clip(exposure*np.clip(target/max(measured,1),.5,2),elo,ehi))
    # Gain units are vendor codes, not a linear multiplier (especially near maximum).
    delta=max(gstep,min((ghi-glo)*.045,max(gstep,(ghi-glo)*.018*abs(math.log(max(target,1)/max(measured,1))))))
    desired_gain=float(np.clip(gain+direction*delta,glo,ghi))
    if gain_slope is not None and gain_slope>1e-6:
        desired_gain=float(np.clip(gain+np.clip(math.log(max(target,1)/max(measured,1))/gain_slope,-delta,delta),glo,ghi))
    if gstep>0:desired_gain=float(np.clip(glo+round((desired_gain-glo)/gstep)*gstep,glo,ghi))
    if allowed_gains is not None:
        candidates=[g for g in allowed_gains if glo-gtol<=g<=ghi+gtol and (g-gain)*direction>gtol]
        desired_gain=min(candidates,key=lambda g:abs(g-desired_gain)) if candidates else gain
    order=['exposure','gain'] if priority=='优先曝光时长' else ['gain','exposure']
    if lock_exposure:order=[which for which in order if which!='exposure']
    if lock_gain:order=[which for which in order if which!='gain']
    if not order:return exposure,gain,'曝光和增益已锁定'
    if allowed_pairs is not None:
        pairs=[(float(e),float(g)) for e,g in allowed_pairs if elo-etol<=e<=ehi+etol and glo-gtol<=g<=ghi+gtol]
        for which in order:
            if which=='exposure':
                candidates=[(e,g) for e,g in pairs if abs(g-gain)<=gtol and (e-exposure)*direction>etol]
                if candidates:return (*min(candidates,key=lambda p:abs(p[0]-desired_exp)),'使用匹配的校正档位')
            else:
                candidates=[(e,g) for e,g in pairs if abs(e-exposure)<=etol and (g-gain)*direction>gtol]
                if candidates:return (*min(candidates,key=lambda p:abs(p[1]-desired_gain)),'使用匹配的校正档位')
        return exposure,gain,'校正库没有可继续调整的匹配档位'
    for which in order:
        if which=='exposure' and (desired_exp-exposure)*direction>etol:return desired_exp,gain,'正在调整曝光'
        if which=='gain' and (desired_gain-gain)*direction>gtol:return exposure,desired_gain,'正在调整增益'
    return exposure,gain,'已到曝光与增益边界'

def validate_points(points):
    if not isinstance(points,(list,tuple)) or not 2<=len(points)<=64:raise ValueError('伪彩需要2至64个灰度—颜色控制点')
    result=[]
    for value,color in points:
        value=float(value)
        if not math.isfinite(value):raise ValueError('灰度值必须为有限数值')
        if not isinstance(color,str) or len(color)!=7 or color[0]!='#':raise ValueError('颜色格式应为 #RRGGBB')
        int(color[1:],16);result.append((value,color.lower()))
    result.sort()
    if any(result[i][0]==result[i-1][0] for i in range(1,len(result))):raise ValueError('控制点灰度值不能重复')
    return tuple(result)

@lru_cache(maxsize=8)
def color_table(points):
    points=validate_points(points);x=np.array([v for v,c in points]);colors=np.array([[int(c[i:i+2],16) for i in (1,3,5)] for v,c in points])
    grid=np.linspace(x[0],x[-1],65536)
    table=np.stack([np.interp(grid,x,colors[:,i]) for i in range(3)],axis=1)
    return x[0],x[-1],np.rint(table).astype(np.uint8)

def custom_color(image,points):
    lo,hi,lut=color_table(validate_points(points))
    index=np.clip((image-lo)*(65535/(hi-lo)),0,65535).astype(np.uint16)
    coordinates=np.empty((*index.shape,2),np.int16)
    coordinates[:,:,0]=index&255;coordinates[:,:,1]=index>>8
    return cv2.remap(lut.reshape(256,256,3),coordinates,None,cv2.INTER_NEAREST)

def validate_curve_points(points):
    """Validate a normalized point curve, keeping the endpoints pinned to 0 and 1."""
    if not isinstance(points,(list,tuple)) or not 2<=len(points)<=32:
        raise ValueError('曲线需要2至32个控制点')
    result=[]
    for point in points:
        if not isinstance(point,(list,tuple)) or len(point)!=2:
            raise ValueError('曲线控制点格式无效')
        x,y=map(float,point)
        if not math.isfinite(x) or not math.isfinite(y) or not 0<=x<=1 or not 0<=y<=1:
            raise ValueError('曲线控制点必须在0%至100%范围内')
        result.append((x,y))
    result.sort()
    if any(result[i][0]==result[i-1][0] for i in range(1,len(result))):
        raise ValueError('曲线控制点的输入值不能重复')
    if abs(result[0][0])>1e-6 or abs(result[-1][0]-1)>1e-6:
        raise ValueError('曲线必须从0%输入开始并在100%输入结束')
    return tuple(result)

def camera_raw_curve(x,shadows=0.,darks=0.,lights=0.,highlights=0.):
    """A fast four-zone tone curve modeled on Camera Raw's parametric curve."""
    x=np.asarray(x,dtype=np.float32)
    shadow=np.square(np.clip(1-x/.25,0,1))
    dark=np.square(np.clip(1-np.abs(x-.25)/.25,0,1))
    light=np.square(np.clip(1-np.abs(x-.75)/.25,0,1))
    highlight=np.square(np.clip((x-.75)/.25,0,1))
    y=x + (.35*float(shadows)/100)*shadow + (.20*float(darks)/100)*dark
    y=y + (.20*float(lights)/100)*light + (.35*float(highlights)/100)*highlight
    return np.clip(y,0,1).astype(np.float32,copy=False)

def apply_tone_curve(x,mode='关闭',params=None,points=None):
    """Apply a normalized Camera Raw-like curve with a small cached-size LUT."""
    if mode=='关闭':return np.asarray(x,dtype=np.float32)
    arr=np.clip(np.asarray(x,dtype=np.float32),0,1)
    grid_size=4097
    grid=np.linspace(0,1,grid_size,dtype=np.float32)
    if mode=='Camera Raw 参数曲线':
        p=tuple(params or (0.,0.,0.,0.))
        if len(p)!=4:raise ValueError('Camera Raw 曲线参数无效')
        lut=camera_raw_curve(grid,*p)
    elif mode=='点曲线':
        pts=validate_curve_points(points if points is not None else DEFAULT_CURVE_POINTS)
        lut=np.interp(grid,[p[0] for p in pts],[p[1] for p in pts]).astype(np.float32)
    else:raise ValueError('未知曲线模式')
    index=arr*(grid_size-1)
    lo=np.floor(index).astype(np.int32)
    hi=np.minimum(lo+1,grid_size-1)
    frac=index-lo
    return lut[lo]*(1-frac)+lut[hi]*frac

def _gpu_clip(value,low=0.,high=1.):
    """Clip a float UMat without downloading it to the CPU."""
    return cv2.min(cv2.max(value,float(low)),float(high))

def _gpu_denoise(value,mode,alpha,backend):
    if mode=='中值 3×3（去孤立噪点）':filtered=cv2.medianBlur(value,3)
    else:filtered=cv2.GaussianBlur(value,(3,3),0.6)
    backend.note('GPU即时降噪')
    return cv2.add(value,cv2.multiply(cv2.subtract(filtered,value),float(alpha)))

def _curve_lut_8(mode,params,points):
    grid=np.linspace(0,1,256,dtype=np.float32)
    if mode=='Camera Raw 参数曲线':return np.rint(camera_raw_curve(grid,*(tuple(params or (0.,0.,0.,0.))))*255).astype(np.uint8)
    if mode=='点曲线':
        pts=validate_curve_points(points if points is not None else DEFAULT_CURVE_POINTS)
        return np.rint(np.interp(grid,[p[0] for p in pts],[p[1] for p in pts])*255).astype(np.uint8)
    raise ValueError('未知曲线模式')

def apply_display_adjustments(x,contrast=0.,sharpen=0.,sharpen_radius=1.,curve_mode='关闭',curve_params=None,curve_points=None,backend=None):
    """Apply preview-only tone and unsharp adjustments to normalized luminance.

    When an OpenCL backend is selected, the array operations and filters stay
    on a UMat until the final preview download.  A 256-entry LUT is sufficient
    for the final 8-bit display curve and avoids a CPU round-trip per pixel.
    """
    out=np.clip(np.asarray(x,dtype=np.float32),0,1)
    c=float(contrast);amount=float(sharpen)
    if backend is not None and backend.use_gpu:
        u=backend.upload(out)
        if abs(c)>1e-6:
            u=_gpu_clip(cv2.add(cv2.multiply(cv2.subtract(u,.5),1+c/100.),.5),0,1);backend.note('GPU对比度')
        if curve_mode!='关闭':
            lut=_curve_lut_8(curve_mode,curve_params,curve_points)
            q=cv2.convertScaleAbs(u,alpha=255)
            # Keep the LUT result in float32.  Multiplying an 8-bit UMat by
            # 1/255 without an explicit dtype rounds the curve to only 0/1.
            u=cv2.multiply(cv2.LUT(q,lut),1/255.,dtype=cv2.CV_32F);backend.note('GPU曲线')
        if amount>1e-6:
            radius=max(.1,float(sharpen_radius));blur=cv2.GaussianBlur(u,(0,0),radius)
            u=_gpu_clip(cv2.add(u,cv2.multiply(cv2.subtract(u,blur),amount/100.)),0,1);backend.note('GPU锐化')
        return backend.download(u).astype(np.float32,copy=False)
    if abs(c)>1e-6:out=np.clip((out-.5)*(1+c/100.)+.5,0,1)
    out=apply_tone_curve(out,curve_mode,curve_params,curve_points)
    if amount>1e-6:
        radius=max(.1,float(sharpen_radius));blur=cv2.GaussianBlur(out,(0,0),radius)
        out=np.clip(out+(amount/100.)*(out-blur),0,1)
    return out

def apply_denoise(a,mode='关闭',amount=100.,backend=None,return_device=False):
    """Apply a small preview-time spatial denoise filter.

    The filter is intentionally local and stateless so it adds no rolling
    memory and remains suitable for a live OBS feed.  It is applied after
    stacking and does not alter the raw or processed file buffers.
    """
    if mode not in DENOISE_MODES:raise ValueError('未知即时降噪模式')
    alpha=float(np.clip(amount,0,100))/100
    is_device=backend is not None and isinstance(a,cv2.UMat)
    out=None if is_device else np.asarray(a,dtype=np.float32)
    if mode=='关闭' or alpha<=0:return a if is_device and return_device else (backend.download(a).astype(np.float32,copy=False) if is_device else out)
    if is_device:
        value=a
    else:
        if out.ndim!=2:raise ValueError('即时降噪只接受二维灰度图像')
        value=backend.upload(out) if backend is not None and backend.use_gpu else None
    if backend is not None and backend.use_gpu:
        result=_gpu_denoise(value,mode,alpha,backend)
        return result if return_device else backend.download(result).astype(np.float32,copy=False)
    if out is None:out=np.asarray(a,dtype=np.float32)
    if out.ndim!=2:raise ValueError('即时降噪只接受二维灰度图像')
    if mode=='中值 3×3（去孤立噪点）':filtered=cv2.medianBlur(out,3)
    else:filtered=cv2.GaussianBlur(out,(3,3),0.6)
    return out+(filtered-out)*alpha

def apply_lowlight(a,mode='关闭',strength=50.,backend=None,return_device=False):
    """Apply a fast, preview-only low-light tone preset.

    The public descriptions of modern low-light camera pipelines point to a
    combination of sensor/ISP processing, adaptive tone mapping and 3-D
    denoise rather than one portable filter.  This function deliberately
    implements only the part that is safe for an incoming monochrome frame:
    robust shadow lifting, optional small spatial smoothing, and a highlight
    preserving curve for stars and meteors.  It never changes the raw frame or
    the rolling accumulator.
    """
    if mode not in LOWLIGHT_MODES:raise ValueError('未知弱光增强模式')
    amount=float(np.clip(strength,0,100))/100
    is_device=backend is not None and isinstance(a,cv2.UMat)
    out=backend.download(a).astype(np.float32,copy=False) if is_device else np.asarray(a,dtype=np.float32)
    if mode=='关闭' or amount<=0:return a if is_device and return_device else out
    if out.ndim!=2:raise ValueError('弱光增强只接受二维灰度图像')
    # Percentiles are estimated from at most about 200k samples so the
    # operation stays cheap at live-camera frame rates.
    stride=max(1,int(np.ceil(np.sqrt(out.size/200000))))
    sample=out[::stride,::stride]
    sample=sample[np.isfinite(sample)]
    if sample.size==0:return out.copy()
    black=float(np.percentile(sample,1.0));white=float(np.percentile(sample,99.7))
    if not (math.isfinite(black) and math.isfinite(white)) or white<=black+1e-6:
        return out.copy()
    span=max(white-black,1e-6)
    # A restrained gamma lift makes the faint background visible without the
    # severe posterisation caused by a hard display stretch.
    gamma=1.0/(1.0+1.35*amount)
    if backend is not None and backend.use_gpu:
        u=backend.upload(out)
        normalized=_gpu_clip(cv2.divide(cv2.subtract(u,black),span),0,1)
        lifted=cv2.pow(normalized,gamma)
        if mode=='星点/流星保护':
            shadow_weight=cv2.pow(_gpu_clip(cv2.divide(cv2.subtract(.78,normalized),.78),0,1),2)
            lifted=cv2.add(normalized,cv2.multiply(cv2.subtract(lifted,normalized),shadow_weight))
            # max() keeps values above the robust white estimate unchanged.
            lifted=cv2.max(lifted,cv2.divide(cv2.subtract(u,black),span))
        result=cv2.add(cv2.multiply(lifted,span),black)
        if mode=='弱光提亮 + 轻度降噪':result=_gpu_denoise(result,'高斯 3×3（轻度平滑）',.25+.45*amount,backend)
        backend.note('GPU弱光增强')
        return result if return_device else backend.download(result).astype(np.float32,copy=False)
    normalized=np.clip((out-black)/span,0,1)
    lifted=np.power(normalized,gamma,dtype=np.float32)
    if mode=='星点/流星保护':
        shadow_weight=np.square(np.clip((.78-normalized)/.78,0,1))
        lifted=normalized+(lifted-normalized)*shadow_weight
        original_norm=(out-black)/span
        lifted=np.where(original_norm>1,original_norm,lifted)
    result=black+lifted*span
    if mode=='弱光提亮 + 轻度降噪':result=apply_denoise(result,'高斯 3×3（轻度平滑）',25+45*amount)
    return np.asarray(result,dtype=np.float32)

def apply_mirror(a,horizontal=False,vertical=False,backend=None,return_device=False):
    """Flip the preview frame while leaving the captured raw data untouched."""
    horizontal=bool(horizontal);vertical=bool(vertical)
    is_device=backend is not None and isinstance(a,cv2.UMat)
    out=None if is_device else np.asarray(a)
    if not horizontal and not vertical:return a if is_device and return_device else (backend.download(a) if is_device else out)
    if backend is not None and backend.use_gpu:
        code=-1 if horizontal and vertical else 1 if horizontal else 0
        value=a if is_device else backend.upload(out)
        result=cv2.flip(value,code);backend.note('GPU画面镜像')
        return result if return_device else backend.download(result).astype(np.float32 if out is None else out.dtype,copy=False)
    if out is None:out=np.asarray(a)
    if horizontal:out=np.flip(out,axis=1)
    if vertical:out=np.flip(out,axis=0)
    return np.ascontiguousarray(out)

def pixel_math(base,settings,window_average=None,window_sum=None,reference=None,window_is_local=False,backend=None):
    op=settings.get('math_op','关闭')
    if op=='关闭':return base
    if op not in MATH_OPS:raise ValueError('未知像素运算')
    region=roi_slices(base.shape,settings.get('math_roi'));src=base[region]
    low=settings.get('math_low',-1e9);high=settings.get('math_high',1e9)
    if high<low:raise ValueError('像素灰度范围的下限不能大于上限')
    mask=(src>=low)&(src<=high)
    if not np.any(mask):return base
    x=src[mask];k=float(settings.get('math_k',1));scale=max(.001,float(settings.get('math_scale',65535)))
    gpu_ops={'窗口平均','窗口积分','减去窗口平均','加上窗口平均',*REF_OPS,'加常数','乘系数','幂律','对数','平方根','绝对值','限制范围'}
    if backend is not None and backend.use_gpu and op in gpu_ops:
        # Upload and calculate only the selected region.  The previous path
        # evaluated a full-frame candidate and masked it afterwards, which
        # wasted GPU time and made the selection appear misleading in a live
        # preview.
        u=backend.upload(src)
        candidate=u;sign_restore=False
        if op in WINDOW_OPS or op in REF_OPS:
            data=window_sum if op=='窗口积分' else window_average if op in WINDOW_OPS else reference
            if data is None:raise ValueError('窗口运算正在等待图像')
            candidate_data=backend.upload(data if window_is_local and op in WINDOW_OPS else data[region])
            if op in ('窗口平均','窗口积分'):candidate=candidate_data
            elif op=='减去窗口平均' or op=='减去参考帧':candidate=cv2.subtract(u,candidate_data)
            elif op=='加上窗口平均' or op=='加上参考帧':candidate=cv2.add(u,candidate_data)
            else:candidate=cv2.multiply(cv2.add(u,candidate_data),.5)
        elif op=='加常数':candidate=cv2.add(u,k)
        elif op=='乘系数':candidate=cv2.multiply(u,k)
        elif op=='绝对值':candidate=cv2.absdiff(u,0)
        elif op=='限制范围':candidate=_gpu_clip(u,settings.get('math_clip_low',0),settings.get('math_clip_high',65535))
        elif op in ('幂律','对数','平方根'):
            absolute=cv2.absdiff(u,0);scaled=cv2.divide(absolute,scale)
            if op=='幂律':candidate=cv2.multiply(cv2.pow(scaled,np.clip(k,.05,8)),scale)
            elif op=='对数':candidate=cv2.multiply(cv2.log(cv2.add(scaled,1)),scale)
            else:candidate=cv2.multiply(cv2.sqrt(scaled),scale)
            sign_restore=True
        result_gpu=backend.download(candidate).astype(np.float32,copy=False)
        if sign_restore:result_gpu=np.copysign(result_gpu,src)
        out=base.copy();out_region=out[region];out_region[mask]=result_gpu[mask];backend.note('GPU像素运算 · 选区')
        return out
    if op in WINDOW_OPS:
        data=window_sum if op=='窗口积分' else window_average
        if data is None:raise ValueError('窗口运算正在等待图像')
        y=data[mask] if window_is_local else data[region][mask]
        if op=='减去窗口平均':y=x-y
        elif op=='加上窗口平均':y=x+y
    elif op in REF_OPS:
        if reference is None or reference.shape!=base.shape:raise ValueError('请先记录一张同尺寸的参考帧')
        y=reference[region][mask]
        if op=='减去参考帧':y=x-y
        elif op=='加上参考帧':y=x+y
        else:y=(x+y)*.5
    elif op=='加常数':y=x+k
    elif op=='乘系数':y=x*k
    elif op=='幂律':y=np.sign(x)*np.power(np.abs(x)/scale,np.clip(k,.05,8))*scale
    elif op=='对数':y=np.sign(x)*np.log1p(np.abs(x)/scale)*scale
    elif op=='平方根':y=np.sign(x)*np.sqrt(np.abs(x)/scale)*scale
    elif op=='绝对值':y=np.abs(x)
    else:y=np.clip(x,settings.get('math_clip_low',0),settings.get('math_clip_high',65535))
    if not np.isfinite(y).all():raise ValueError('运算结果溢出，请减小系数或调整尺度')
    out=base.copy();out[region][mask]=y
    return out
