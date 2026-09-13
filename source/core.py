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

    def correct(self, raw, meta, dark=False, bias=False, flat=False):
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
    """Strict (now-window, now] FIFO. Every unique acquired frame enters once."""
    def __init__(self, seconds=3.0, memory_mb=768):
        self.seconds=seconds; self.set_budget(memory_mb)
        self.clear()
    def set_budget(self,memory_mb):
        self.auto=memory_mb==0
        self.limit=int((1024 if self.auto else memory_mb)*1024**2)
    def clear(self):
        self.frames=deque(); self.total=None; self.bytes=0; self.last_time=None
    def push(self, frame, stamp):
        if self.last_time is not None and stamp<=self.last_time:
            raise ValueError('Frame timestamps must strictly increase')
        if self.total is not None and self.total.shape!=frame.shape: self.clear()
        if self.total is None: self.total=np.zeros(frame.shape,np.float64)
        while self.frames and self.frames[0][0]<=stamp-self.seconds+1e-9:
            _,old=self.frames.popleft(); self.total-=old; self.bytes-=old.nbytes
        a=np.array(frame,dtype=np.float32,copy=True)
        required=self.bytes+a.nbytes+self.total.nbytes
        if required>self.limit and self.auto:
            total,available=physical_memory()
            extra=required-(self.bytes+self.total.nbytes)
            # Leave at least 2 GiB and half currently free RAM to the OS and other apps.
            if total and available-extra>=max(2*1024**3,available*.5):
                self.limit=required
        if required>self.limit:
            reason='可用物理内存不足，自动扩展已停止' if self.auto else f'手动缓存上限 {self.limit/1024**2:.0f} MB 已用满（可将缓存设为 0 自动扩展）'
            raise MemoryError(reason+'；未缩短窗口或降低画质。请减少窗口时长或处理尺寸后恢复采集。')
        self.frames.append((stamp,a)); self.total+=a; self.bytes+=a.nbytes; self.last_time=stamp
    def result(self,mode='平均'):
        if self.total is None: return None
        return (self.total/(len(self.frames) if mode=='平均' else 1)).astype(np.float32)
    @property
    def span(self):
        return self.frames[-1][0]-self.frames[0][0] if len(self.frames)>1 else 0.0

def bin_image(a,factor):
    if factor==1:return a
    h,w=a.shape; h-=h%factor; w-=w%factor
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

def display_rgb(a,black=0,white=65535,gamma=1,palette='灰度',max_width=1920,custom_points=None):
    if a.shape[1]>max_width:
        a=cv2.resize(a,(max_width,max(1,round(a.shape[0]*max_width/a.shape[1]))),interpolation=cv2.INTER_AREA)
    if palette=='自定义':
        from processing import custom_color,DEFAULT_POINTS
        return custom_color(a,custom_points if custom_points is not None else DEFAULT_POINTS)
    x=np.clip((a-black)/max(white-black,1),0,1)
    x=np.power(x,1/max(gamma,.05))
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
    return counts,dict(p01=float(np.percentile(vals,1)),p995=float(np.percentile(vals,99.5)),
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
