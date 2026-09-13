"""Linear-light calibration and finite-window integration. No display transforms here."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
import json
import numpy as np
import cv2

@dataclass
class FrameMeta:
    camera: str = 'SIMULATOR'
    exposure_ms: float = 100.0
    gain: float = 1.0
    mode: str = '640x480'
    bits: int = 16
    temperature: float | None = None

@dataclass
class Master:
    kind: str
    image: np.ndarray
    meta: FrameMeta
    count: int = 1

class CalibrationError(ValueError): pass

class CalibrationLibrary:
    def __init__(self): self.masters: list[Master] = []

    def compatible(self, master, meta, shape, exposure=False):
        m=master.meta
        return (master.image.shape==shape and m.camera==meta.camera and
                m.mode==meta.mode and m.bits==meta.bits and
                abs(m.gain-meta.gain)<1e-6 and
                (not exposure or abs(m.exposure_ms-meta.exposure_ms)<=max(.01,meta.exposure_ms*.001)) and
                (m.temperature is None or meta.temperature is None or abs(m.temperature-meta.temperature)<=5))

    def find(self, kind, meta, shape, exposure=False):
        return next((m for m in reversed(self.masters) if m.kind==kind and self.compatible(m,meta,shape,exposure)),None)

    def add(self, master):
        if master.image.ndim!=2 or not np.isfinite(master.image).all():
            raise CalibrationError('校正帧必须为有限数值的二维灰度图像')
        self.masters=[m for m in self.masters if not (m.kind==master.kind and self.compatible(m,master.meta,master.image.shape,True))]
        self.masters.append(master)

    def correct(self, raw, meta, dark=False, bias=False, flat=False, backend=None):
        if backend is not None and backend.use_gpu:
            # Calibration masters are uploaded once and the arithmetic stays
            # in the selected OpenCL device until the corrected frame is
            # needed by the rolling window.
            out=backend.upload(raw.astype(np.float32,copy=False))
            if dark:
                d=self.find('dark',meta,raw.shape,True)
                if d is None: raise CalibrationError('没有匹配当前相机、分辨率、增益、曝光和温度的暗场；输出已冻结')
                out=backend.subtract(out,backend.upload_cached(d.image))
            elif bias:
                b=self.find('bias',meta,raw.shape)
                if b is None: raise CalibrationError('没有匹配的偏置帧；输出已冻结')
                out=backend.subtract(out,backend.upload_cached(b.image))
            if flat:
                f=self.find('flat',meta,raw.shape)
                if f is None: raise CalibrationError('没有匹配的平场；输出已冻结')
                out=backend.divide(out,backend.upload_cached(f.image))
            backend.note('GPU校正')
            return backend.download(out).astype(np.float32,copy=False)
        out=raw.astype(np.float32)
        if dark:
            d=self.find('dark',meta,raw.shape,True)
            if d is None: raise CalibrationError('没有匹配当前相机、分辨率、增益、曝光和温度的暗场；输出已冻结')
            out-=d.image  # Raw master dark already contains bias. Never subtract it twice.
        elif bias:
            b=self.find('bias',meta,raw.shape)
            if b is None: raise CalibrationError('没有匹配的偏置帧；输出已冻结')
            out-=b.image
        if flat:
            f=self.find('flat',meta,raw.shape)
            if f is None: raise CalibrationError('没有匹配的平场；输出已冻结')
            np.divide(out,f.image,out=out)
        return out

    def make_flat(self, raw_average, meta, count):
        base=raw_average.astype(np.float32).copy()
        d=self.find('dark',meta,base.shape,True)
        b=self.find('bias',meta,base.shape)
        if d is not None: base-=d.image
        elif b is not None: base-=b.image
        else: raise CalibrationError('制作平场前请先拍摄匹配偏置，或相同曝光的暗场')
        med=float(np.median(base))
        if med<=10 or np.any(base<=max(1,med*.01)):
            raise CalibrationError('平场过暗或存在无效像素，请增加均匀照明后重拍')
        if np.max(raw_average)>=65530: raise CalibrationError('平场存在饱和像素，请降低曝光')
        return Master('flat',base/float(np.mean(base,dtype=np.float64)),meta,count)

    def dark_exposures(self, meta, shape, low=5, high=5000):
        return sorted(set(m.meta.exposure_ms for m in self.masters if m.kind=='dark' and
                          self.compatible(m,meta,shape) and low<=m.meta.exposure_ms<=high))

    def save(self,path):
        data={f'image_{i}':m.image for i,m in enumerate(self.masters)}
        manifest=[dict(kind=m.kind,meta=asdict(m.meta),count=m.count) for m in self.masters]
        data['manifest']=np.array(json.dumps(manifest,ensure_ascii=False))
        with open(path,'wb') as f: np.savez_compressed(f,**data)

    def load(self,path):
        with np.load(path,allow_pickle=False) as z:
            manifest=json.loads(str(z['manifest']))
            if len(manifest)>200: raise CalibrationError('校正库条目过多')
            fresh=CalibrationLibrary()
            for i,m in enumerate(manifest):
                if m['kind'] not in ('dark','bias','flat'): raise CalibrationError('无效校正类型')
                a=z[f'image_{i}'].astype(np.float32)
                if m['kind']=='flat' and np.any(a<=0): raise CalibrationError('平场含零或负值')
                fresh.add(Master(m['kind'],a,FrameMeta(**m['meta']),int(m['count'])))
        self.masters=fresh.masters

def physical_memory():
    """Return total/available physical bytes; no optional dependency."""
    import ctypes, os, struct
    if os.name!='nt':return (0,0)
    buf=ctypes.create_string_buffer(64);struct.pack_into('I',buf,0,64)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(buf):return (0,0)
    return struct.unpack_from('QQ',buf,8)

class RollingIntegrator:
    """Finite FIFO using either elapsed time or a fixed number of frames."""
    def __init__(self, seconds=3.0, memory_mb=768, window_unit='时间', frame_limit=30, backend=None):
        self.seconds=float(seconds);self.window_unit='时间';self.frame_limit=int(frame_limit)
        self.backend=backend;self.gpu=bool(getattr(backend,'use_gpu',False));self.total_nbytes=0;self._shape=None
        self.max_enabled=False;self.max_block_size=8
        self.set_window(window_unit,seconds,frame_limit)
        self.set_budget(memory_mb)
        self.clear()
    def set_backend(self,backend):
        changed=(self.gpu!=bool(getattr(backend,'use_gpu',False)))
        self.backend=backend;self.gpu=bool(getattr(backend,'use_gpu',False))
        if changed:self.clear()
        return changed
    def set_window(self,unit='时间',seconds=None,frames=None):
        if unit not in ('时间','帧数'):raise ValueError('滚动窗口单位只能是时间或帧数')
        if seconds is not None:
            seconds=float(seconds)
            if not np.isfinite(seconds) or seconds<=0:raise ValueError('窗口时长必须大于0秒')
            self.seconds=seconds
        if frames is not None:
            frames=int(frames)
            if frames<1:raise ValueError('窗口帧数必须至少为1')
            self.frame_limit=frames
        self.window_unit=unit
    def set_budget(self,memory_mb):
        self.auto=memory_mb==0
        self.limit=int((1024 if self.auto else memory_mb)*1024**2)
    def clear(self):
        self.frames=deque(); self.frame_means=deque(); self.total=None; self.total_nbytes=0; self._shape=None; self.bytes=0; self.last_time=None
        self.max_blocks=deque();self.max_nbytes=0
    def _max_reduce(self,values):
        if not values:return None
        result=values[0]
        for value in values[1:]:
            result=self.backend.maximum(result,value) if self.gpu else np.maximum(result,value)
        return result
    def _max_append(self,item):
        """Add a shared frame reference to the small block-max structure."""
        if not self.max_enabled:return
        stamp,frame=item
        if not self.max_blocks or len(self.max_blocks[-1]['frames'])>=self.max_block_size:
            self.max_blocks.append({'frames':deque([item]),'maximum':frame})
            self.max_nbytes+=self._frame_nbytes(frame)
            return
        block=self.max_blocks[-1]
        updated=self.backend.maximum(block['maximum'],frame) if self.gpu else np.maximum(block['maximum'],frame)
        block['frames'].append(item);block['maximum']=updated
    def _max_remove_oldest(self):
        if not self.max_enabled or not self.max_blocks:return
        block=self.max_blocks[0]
        if len(block['frames'])>1:
            updated=self._max_reduce([frame for _,frame in list(block['frames'])[1:]])
        else:updated=None
        block['frames'].popleft()
        if updated is None:
            self.max_blocks.popleft()
            self.max_nbytes-=self._frame_nbytes(block['maximum'])
        else:block['maximum']=updated
    def _frame_nbytes(self,frame):
        if self.gpu:return int(np.prod(self._shape))*4
        return np.asarray(frame).nbytes
    @staticmethod
    def _sample_mean(frame):
        arr=np.asarray(frame)
        stride=max(1,int(np.ceil(np.sqrt(arr.size/200000))))
        return float(np.mean(arr[::stride,::stride],dtype=np.float64))
    def _rebuild_max_blocks(self):
        self.max_blocks=deque();self.max_nbytes=0
        if not self.max_enabled:return
        for item in self.frames:self._max_append(item)
    def set_max_enabled(self,enabled):
        enabled=bool(enabled)
        if enabled==self.max_enabled:return
        self.max_enabled=enabled
        if not enabled:
            self.max_blocks=deque();self.max_nbytes=0
            return
        try:self._rebuild_max_blocks()
        except Exception as exc:
            if not self.gpu:raise
            self._fallback_cpu(str(exc))
    def _fallback_cpu(self,reason):
        """Migrate a live OpenCL window to NumPy without dropping its frames."""
        if not self.gpu:return
        backend=self.backend
        try:
            total=backend.download(self.total) if self.total is not None else None
            frames=deque((stamp,backend.download(frame).astype(np.float32,copy=False)) for stamp,frame in self.frames)
        except Exception as exc:
            # If downloading the broken device buffer also fails, a safe reset
            # is preferable to publishing a corrupted accumulation.
            self.clear()
            if hasattr(backend,'disable_gpu'):backend.disable_gpu(f'{reason}；设备缓存读取失败：{exc}')
            else:self.gpu=False
            return
        self.frames=frames
        # Means are kept separately so adaptive stacking can choose a short
        # suffix without downloading or re-scanning every cached frame.
        if len(self.frame_means)!=len(frames):
            self.frame_means=deque(self._sample_mean(frame) for _,frame in frames)
        # The OpenCL accumulator is float32; keeping that dtype during the
        # migration preserves its values and avoids doubling the cache budget
        # just because the driver was reset.
        self.total=None if total is None else np.asarray(total,dtype=np.float32)
        self.total_nbytes=0 if self.total is None else self.total.nbytes
        self.gpu=False
        if hasattr(backend,'disable_gpu'):backend.disable_gpu(reason)
        self._rebuild_max_blocks()
    def _push_once(self,source,stamp):
        """Append one frame; GPU operations are kept transactional."""
        prepared=np.asarray(source,dtype=np.float32)
        frame_bytes=prepared.nbytes
        if self.total is not None and self._shape!=source.shape:self.clear()
        if self.total is None:
            self.total=self.backend.upload(np.zeros(source.shape,np.float32)) if self.gpu else np.zeros(source.shape,np.float64)
            self._shape=source.shape;self.total_nbytes=source.size*(4 if self.gpu else 8)
        if self.window_unit=='帧数':
            while len(self.frames)>=self.frame_limit:
                old_stamp,old=self.frames[0]
                updated=self.backend.subtract(self.total,old) if self.gpu else self.total-old
                self._max_remove_oldest()
                self.frames.popleft();self.frame_means.popleft();self.total=updated;self.bytes-=frame_bytes
        else:
            while self.frames and self.frames[0][0]<=stamp-self.seconds+1e-9:
                old_stamp,old=self.frames[0]
                updated=self.backend.subtract(self.total,old) if self.gpu else self.total-old
                self._max_remove_oldest()
                self.frames.popleft();self.frame_means.popleft();self.total=updated;self.bytes-=frame_bytes
        a=self.backend.upload(prepared) if self.gpu else np.array(prepared,copy=True)
        new_block=self.max_enabled and (not self.max_blocks or len(self.max_blocks[-1]['frames'])>=self.max_block_size)
        if self.max_enabled and not new_block:
            last=self.max_blocks[-1]
            new_max=self.backend.maximum(last['maximum'],a) if self.gpu else np.maximum(last['maximum'],a)
        else:new_max=a
        required=self.bytes+frame_bytes+self.total_nbytes+self.max_nbytes+(frame_bytes if new_block else 0)
        if required>self.limit and self.auto:
            total,available=physical_memory()
            extra=required-(self.bytes+self.total_nbytes+self.max_nbytes)
            # Leave at least 2 GiB and half currently free RAM to the OS and other apps.
            if total and available-extra>=max(2*1024**3,available*.5):
                self.limit=required
        if required>self.limit:
            reason='可用物理内存不足，自动扩展已停止' if self.auto else f'手动缓存上限 {self.limit/1024**2:.0f} MB 已用满（可将缓存设为 0 自动扩展）'
            raise MemoryError(reason+'；未缩短窗口或降低画质。请减少窗口时长或处理尺寸后恢复采集。')
        updated=self.backend.add(self.total,a) if self.gpu else self.total+a
        item=(stamp,a)
        if self.max_enabled:
            if new_block:
                self.max_blocks.append({'frames':deque([item]),'maximum':new_max});self.max_nbytes+=frame_bytes
            else:
                self.max_blocks[-1]['frames'].append(item);self.max_blocks[-1]['maximum']=new_max
        self.frames.append(item);self.frame_means.append(self._sample_mean(prepared));self.total=updated;self.bytes+=frame_bytes;self.last_time=stamp
    def push(self, frame, stamp):
        if self.last_time is not None and stamp<=self.last_time:
            raise ValueError('Frame timestamps must strictly increase')
        source=np.asarray(frame)
        try:
            self._push_once(source,stamp)
        except MemoryError:
            raise
        except Exception as exc:
            if not self.gpu:raise
            self._fallback_cpu(str(exc))
            self._push_once(source,stamp)
    def _max_result(self):
        if not self.max_enabled:self.set_max_enabled(True)
        if not self.max_blocks:return None
        value=self._max_reduce([block['maximum'] for block in self.max_blocks])
        return self.backend.download(value) if self.gpu else value
    def result(self,mode='平均'):
        if self.total is None: return None
        if mode=='最大值':
            try:value=self._max_result()
            except Exception as exc:
                if not self.gpu:raise
                self._fallback_cpu(str(exc));value=self._max_result()
            return None if value is None else np.asarray(value,dtype=np.float32)
        if self.gpu:
            try:total=self.backend.download(self.total)
            except Exception as exc:
                self._fallback_cpu(str(exc));total=self.total
        else:total=self.total
        return (total/(len(self.frames) if mode=='平均' else 1)).astype(np.float32)
    def adaptive_result(self,target,scale=1.0):
        """Return the shortest recent integral whose mean reaches ``target``.

        Frames are stored in a common exposure scale.  ``scale`` converts the
        cumulative value back to the current camera ADU scale for comparison.
        The configured time/frame window remains the hard upper bound.
        """
        if self.total is None or not self.frames:return None,0,0.0
        target=float(target);scale=float(scale)
        if not np.isfinite(target) or target<0:raise ValueError('叠加目标亮度必须是非负有限数值')
        cumulative=0.0;count=len(self.frame_means)
        for index,mean in enumerate(reversed(self.frame_means),1):
            cumulative+=float(mean)*scale
            if cumulative>=target:
                count=index;break
        selected=list(self.frames)[-count:]
        try:
            if count==len(self.frames):
                result=self.backend.download(self.total).astype(np.float32,copy=False) if self.gpu else np.asarray(self.total,dtype=np.float32)
            elif count==1:
                result=self.backend.download(selected[0][1]).astype(np.float32,copy=False) if self.gpu else np.asarray(selected[0][1],dtype=np.float32)
            elif self.gpu:
                value=self.backend.upload(np.zeros(self._shape,np.float32))
                for _,frame in selected:value=self.backend.add(value,frame)
                result=self.backend.download(value).astype(np.float32,copy=False)
            else:
                result=np.sum([frame for _,frame in selected],axis=0,dtype=np.float64).astype(np.float32)
        except Exception as exc:
            if not self.gpu:raise
            self._fallback_cpu(str(exc));return self.adaptive_result(target,scale)
        span=selected[-1][0]-selected[0][0] if len(selected)>1 else 0.0
        return result,count,float(span)
    @property
    def total_shape(self):
        return self._shape
    @property
    def total_bytes(self):
        return self.total_nbytes+self.max_nbytes
    @property
    def span(self):
        return self.frames[-1][0]-self.frames[0][0] if len(self.frames)>1 else 0.0

def bin_image(a,factor,backend=None):
    if factor==1:return a
    h,w=a.shape; h-=h%factor; w-=w%factor
    if backend is not None and backend.use_gpu:
        value=backend.upload(a[:h,:w]);result=cv2.resize(value,(w//factor,h//factor),interpolation=cv2.INTER_AREA)
        backend.note('GPU软件Binning')
        return backend.download(result).astype(np.float32,copy=False)
    return cv2.resize(a[:h,:w],(w//factor,h//factor),interpolation=cv2.INTER_AREA)

def exposure_target(current, measured, target, low, high, allowed=None):
    """Bounded AE in linear sensor space. Ignore ≤8% target error."""
    if abs(measured-target)<=max(1,target*.08):return current
    ratio=np.clip(target/max(measured,1),.5,2.0)
    desired=float(np.clip(current*ratio,low,high))
    if allowed:
        candidates=[v for v in allowed if low<=v<=high]
        if not candidates:return current
        desired=min(candidates,key=lambda v:abs(np.log(v/max(desired,.001))))
    return desired

def display_rgb(a,black=0,white=65535,gamma=1,palette='灰度',max_width=1920,custom_points=None,
                contrast=0.,sharpen=0.,sharpen_radius=1.,curve_mode='关闭',curve_params=None,curve_points=None,backend=None):
    """Convert a processed mono frame to the preview image and apply display-only adjustments."""
    if a.shape[1]>max_width:
        size=(max_width,max(1,round(a.shape[0]*max_width/a.shape[1])))
        if backend is not None and backend.use_gpu:
            a=backend.download(cv2.resize(backend.upload(a),size,interpolation=cv2.INTER_AREA)).astype(np.float32,copy=False);backend.note('GPU预览缩放')
        else:a=cv2.resize(a,size,interpolation=cv2.INTER_AREA)
    from processing import apply_display_adjustments,DEFAULT_CURVE_POINTS
    if palette=='自定义':
        from processing import custom_color,DEFAULT_POINTS,validate_points
        points=custom_points if custom_points is not None else DEFAULT_POINTS
        points=tuple(points)
        # Custom pseudo-color retains its absolute gray control points. New tone
        # adjustments operate within that LUT's input range so existing palettes
        # keep their meaning when all adjustment sliders are at zero.
        lo,hi=validate_points(points)[0][0],validate_points(points)[-1][0]
        z=np.clip((a-lo)/max(hi-lo,1),0,1)
        z=apply_display_adjustments(z,contrast,sharpen,sharpen_radius,curve_mode,curve_params,curve_points,backend)
        return custom_color(z*(hi-lo)+lo,points)
    x=np.clip((a-black)/max(white-black,1),0,1)
    x=np.power(x,1/max(gamma,.05))
    x=apply_display_adjustments(x,contrast,sharpen,sharpen_radius,curve_mode,curve_params,
                                curve_points if curve_points is not None else DEFAULT_CURVE_POINTS,backend)
    gray=np.ascontiguousarray(np.rint(x*255).astype(np.uint8))
    maps={'火焰':cv2.COLORMAP_INFERNO,'青蓝':cv2.COLORMAP_OCEAN,'科学色':cv2.COLORMAP_VIRIDIS}
    if palette in maps:return cv2.cvtColor(cv2.applyColorMap(gray,maps[palette]),cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

def histogram(a,domain='sensor',bits=16):
    stride=max(1,int(np.sqrt(a.size/300000)))
    vals=a[::stride,::stride]
    if domain=='processed':low=min(0.,float(vals.min()));high=max(65535.,float(vals.max()))+1
    elif domain=='display':low,high=0.,256.
    else:low,high=0.,float(1<<bits)
    counts,edges=np.histogram(vals,bins=256,range=(low,high))
    return counts,dict(min=float(vals.min()),max=float(vals.max()),p01=float(np.percentile(vals,1)),p995=float(np.percentile(vals,99.5)),
                       hist_low=low,hist_high=high,hist_samples=int(vals.size),
                       mean=float(np.mean(vals)),saturation=float(np.mean(vals>=((1<<bits)-1))*100),
                       focus=float(cv2.Laplacian(vals.astype(np.float32),cv2.CV_32F).var()))

def write_fits(path,a,meta):
    """Minimal standards-compliant single-image FLOAT32 FITS (preserves sums/negatives)."""
    cards=[]
    def card(k,v):
        if isinstance(v,bool): s='T' if v else 'F'
        elif isinstance(v,str): s="'"+v.replace("'","''")[:60]+"'"
        else:s=str(v)
        cards.append(f'{k:<8}= {s:>20}'.ljust(80)[:80])
    card('SIMPLE',True);card('BITPIX',-32);card('NAXIS',2)
    card('NAXIS1',a.shape[1]);card('NAXIS2',a.shape[0])
    card('EXPTIME',meta.exposure_ms/1000);card('GAIN',meta.gain)
    cards.append('END'.ljust(80))
    header=''.join(cards).encode('ascii',errors='replace'); header+=b' '*((-len(header))%2880)
    data=np.asarray(a,dtype='>f4').tobytes(); data+=b'\0'*((-len(data))%2880)
    Path(path).write_bytes(header+data)

class SerWriter:
    """SER v3 little-endian mono16, raw sensor data only; finalize count on close."""
    def __init__(self,path,shape):
        import struct
        self.f=open(path,'wb');self.count=0;self.shape=shape
        self.f.write(struct.pack('<14s7I40s40s40sQQ',b'LUCAM-RECORDER',0,0,0,shape[1],shape[0],16,0,
                                 b'',b'StarfieldStudio',b'',0,0))
    def append(self,a):
        if a.shape!=self.shape or a.dtype!=np.uint16:raise ValueError('SER requires unchanged mono16 format')
        self.f.write(a.astype('<u2',copy=False).tobytes());self.count+=1
    def close(self):
        import struct
        if not self.f.closed:
            self.f.seek(38);self.f.write(struct.pack('<I',self.count));self.f.close()
