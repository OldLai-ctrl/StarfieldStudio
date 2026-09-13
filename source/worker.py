from __future__ import annotations
import threading, queue, time, json, traceback, math
import copy
from pathlib import Path
from dataclasses import replace, asdict
import numpy as np
import tifffile
from core import *
from camera import TucamCamera, SimCamera
from camera_common import open_camera
from processing import *

DEFAULTS=dict(exposure=100.,gain=1.,seconds=3.,mode='平均',software_bin=1,
              dark=False,bias=False,flat=False,auto=False,target=20.,ae_low=5.,ae_high=5000.,
              black=0.,white=8000.,gamma=1.,palette='灰度',stretch=False,memory=0,hist_source='相机原始灰度',
              ae_mode='手动',ae_gain_low=1.,ae_gain_high=257.,ae_roi=None,ae_show=True,
              custom_points=DEFAULT_POINTS,math_op='关闭',math_roi=None,math_show=True,
              math_low=-1e9,math_high=1e9,math_k=1.,math_scale=65535.,math_clip_low=0.,math_clip_high=65535.,
              contrast=0.,sharpen=0.,sharpen_radius=1.,curve_mode='关闭',
              curve_shadows=0.,curve_darks=0.,curve_lights=0.,curve_highlights=0.,
              curve_points=DEFAULT_CURVE_POINTS)

class CaptureWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.commands=queue.Queue();self.events=queue.Queue();self.lock=threading.Lock()
        self.latest=None;self.serial=0;self.cam=None;self.running=True;self.paused=False
        self.settings=copy.deepcopy(DEFAULTS);self.library=CalibrationLibrary();self.roll=RollingIntegrator(3,0)
        self.raw=None;self.processed=None;self.meta=None;self.rec=None;self.master=None;self.last_ae=0
        self.single=None;self.last_frame_time=None;self.window_local=False;self.pre_math=None;self.reference=None
        self.ae_status='手动';self.ae_previous=None
    def send(self,action,**kw):self.commands.put((action,kw))
    def event(self,kind,**kw):self.events.put(dict(kind=kind,**kw))
    def info(self):
        c=self.cam
        self.event('connected',name=c.name,exposure=c.meta.exposure_ms,gain=c.meta.gain,
                   exp_range=c.exp_range,gain_range=c.gain_range,resolutions=c.resolutions,bins=c.bins,
                   resolution=c.mode,native_bin=c.bin_value,sim=isinstance(c,SimCamera))
    def end_record(self):
        if self.rec:
            self.rec.close();self.rec=None;self.event('log',text='原始 SER 视频已完成保存')
        self.event('record',active=False)
    def disconnect(self):
        self.end_record()
        if self.master:self.cancel_master()
        if self.cam:self.cam.close();self.cam=None
        self.roll.clear();self.raw=None;self.processed=None;self.single=None;self.pre_math=None;self.reference=None
        with self.lock:self.latest=None
        self.event('disconnected')
    def cancel_master(self):
        if not self.master:return
        m=self.master;self.master=None
        if self.cam:
            self.cam.configure(exposure=m['restore'])
            if isinstance(self.cam,SimCamera):self.cam.scene=m['scene']
        self.roll.clear();self.event('master',active=False,text='校正帧采集结束')
    def command(self,action,k):
        if action=='quit':self.running=False;return
        if action=='disconnect':self.disconnect();return
        if action=='connect':
            self.disconnect()
            try:
                self.cam=SimCamera() if k['sim'] else open_camera(k.get('backend','tucam'),k['dll'],k.get('index',0))
                self.cam.start()
            except Exception:
                self.disconnect();raise
            self.paused=False
            self.settings.update(exposure=self.cam.meta.exposure_ms,gain=self.cam.meta.gain,
                ae_low=max(self.cam.exp_range[0],min(5,self.cam.exp_range[1])),ae_high=max(self.cam.exp_range[0],min(5000,self.cam.exp_range[1])),
                ae_gain_low=self.cam.gain_range[0],ae_gain_high=self.cam.gain_range[1],auto=False,ae_mode='手动')
            self.info();self.event('settings',values={k:self.settings[k] for k in ('ae_low','ae_high','ae_gain_low','ae_gain_high','auto','ae_mode')});return
        if action=='library_save':self.library.save(k['path']);self.event('log',text='校正库已保存');return
        if action=='library_load':
            self.library.load(k['path']);self.roll.clear();self.event('log',text=f'已加载 {len(self.library.masters)} 个校正主帧');return
        if action=='settings':
            new=k['values'].copy()
            if self.master:raise ValueError('正在拍摄校正帧，请结束后再修改设置')
            if 'ae_mode' in new:
                if new['ae_mode'] not in ('手动','优先曝光时长','优先增益'):raise ValueError('未知自动曝光策略')
                new['auto']=new['ae_mode']!='手动'
            elif 'auto' in new:new['ae_mode']='优先曝光时长' if new['auto'] else '手动'
            if ('exposure' in new or 'gain' in new) and 'auto' not in new:new.update(auto=False,ae_mode='手动')
            for name in ('ae_roi','math_roi'):
                if name in new:new[name]=validate_roi(new[name])
            if 'custom_points' in new:new['custom_points']=validate_points(new['custom_points'])
            if 'curve_points' in new:new['curve_points']=validate_curve_points(new['curve_points'])
            if 'curve_mode' in new and new['curve_mode'] not in CURVE_MODES:raise ValueError('未知曲线模式')
            for name,low,high in [('contrast',-100,100),('sharpen',0,300),('sharpen_radius',.1,20),
                                  ('curve_shadows',-100,100),('curve_darks',-100,100),
                                  ('curve_lights',-100,100),('curve_highlights',-100,100)]:
                if name in new and (not np.isfinite(new[name]) or not low<=float(new[name])<=high):
                    raise ValueError(f'{name}超出允许范围')
            if 'math_op' in new and new['math_op'] not in MATH_OPS:raise ValueError('未知像素运算')
            if new.get('math_op') in REF_OPS and self.reference is None:raise ValueError('请先点击“记录参考帧”，再选择参考帧运算')
            proposed={**self.settings,**new}
            if proposed['ae_low']>proposed['ae_high'] or proposed['ae_gain_low']>proposed['ae_gain_high']:raise ValueError('自动曝光／增益下限不能大于上限')
            if proposed['math_low']>proposed['math_high'] or proposed['math_clip_low']>proposed['math_clip_high']:raise ValueError('像素范围下限不能大于上限')
            if proposed['math_op']=='幂律' and not .05<=proposed['math_k']<=8:raise ValueError('幂律指数请设置在0.05至8之间')
            if new.get('auto'):self.end_record()
            if any(self.settings.get(n)!=v for n,v in new.items() if n in ('gain','software_bin','dark','bias','flat')):self.roll.clear()
            if ('mode' in new and (new['mode']=='关闭' or self.settings['mode']=='关闭')) or ('math_roi' in new) or ('math_op' in new and self.settings['mode']=='关闭'):self.roll.clear()
            if self.cam:
                hw={n:v for n,v in new.items() if n in ('exposure','gain','resolution','native_bin')}
                if hw:
                    self.end_record();self.cam.configure(**hw)
                    if 'gain' in hw or 'resolution' in hw or 'native_bin' in hw:self.roll.clear()
                    new.update(exposure=self.cam.meta.exposure_ms,gain=self.cam.meta.gain)
                    if 'resolution' in hw:new['resolution']=self.cam.mode
                elif 'auto' in new and hasattr(self.cam,'ensure_manual'):self.cam.ensure_manual()
            self.settings.update(new);self.roll.seconds=self.settings['seconds']
            if 'ae_mode' in new:self.ae_status=new['ae_mode'];self.ae_previous=None
            if 'memory' in new:self.roll.set_budget(self.settings['memory'])
            self.event('settings',values=new)
            if (self.roll.total is not None or self.single is not None) and not any(n in new for n in ('exposure','gain','resolution','native_bin')):self.publish()
            return
        if action=='reference':
            if self.pre_math is None:raise ValueError('还没有可记录的画面')
            self.reference=self.pre_math.copy();self.event('log',text='已记录运算前、拉伸前的参考帧');return
        if action=='auto_levels':
            if self.processed is None:raise ValueError('还没有可拉伸的图像')
            _,stats=histogram(self.processed,'processed')
            v=dict(black=stats['p01'],white=max(stats['p01']+1,stats['p995']),stretch=False)
            self.settings.update(v);self.event('settings',values=v);self.publish();return
        if not self.cam:raise ValueError('请先连接相机或模拟器')
        if action=='pause':
            if self.master:self.cancel_master()
            self.paused=not self.paused
            self.cam.stop() if self.paused else self.cam.start()
            self.roll.clear();self.event('paused',value=self.paused);return
        if action=='reset':self.roll.clear();return
        if action=='scene':
            if isinstance(self.cam,SimCamera):self.cam.scene=k['scene'];self.roll.clear()
            return
        if action=='master':
            if self.master:raise ValueError('校正帧正在采集中')
            if self.paused:raise ValueError('请先恢复采集')
            self.end_record();self.settings.update(auto=False,ae_mode='手动');self.event('settings',values={'auto':False,'ae_mode':'手动'})
            kind=k['master_kind'];exp=self.cam.exp_range[0] if kind=='bias' else self.cam.meta.exposure_ms
            scene=self.cam.scene if isinstance(self.cam,SimCamera) else None
            self.master=dict(kind=kind,count=k.get('count',32),n=0,total=None,restore=self.cam.meta.exposure_ms,scene=scene)
            if isinstance(self.cam,SimCamera):self.cam.scene='平场光源' if kind=='flat' else '暗场'
            self.cam.configure(exposure=exp);self.roll.clear();return
        if action=='cancel_master':self.cancel_master();return
        if action=='record':
            if self.rec:self.end_record();return
            if self.raw is None:raise ValueError('还没有收到图像')
            if self.master:raise ValueError('请先结束校正帧采集')
            self.settings.update(auto=False,ae_mode='手动');self.event('settings',values={'auto':False,'ae_mode':'手动'})
            self.rec=SerWriter(k['path'],self.raw.shape)
            Path(k['path']+'.json').write_text(json.dumps(asdict(self.meta),ensure_ascii=False,indent=2),encoding='utf-8')
            self.event('record',active=True);return
        if action=='save':
            if self.master:raise ValueError('请结束校正帧采集后再保存图像')
            if self.raw is None or self.processed is None:raise ValueError('还没有有效图像')
            path=k['path'];fmt=k['format']
            if fmt=='raw':tifffile.imwrite(path,self.raw,metadata=asdict(self.meta))
            elif fmt=='fits':write_fits(path,self.processed,self.meta)
            elif fmt=='float':tifffile.imwrite(path,self.processed,metadata=dict(stack_mode=self.settings['mode'],stack_frames=len(self.roll.frames),**asdict(self.meta)))
            else:
                s=self.settings
                import cv2
                rgb=display_rgb(self.processed,s['black'],s['white'],s['gamma'],s['palette'],max_width=30000,custom_points=s['custom_points'],
                                contrast=s['contrast'],sharpen=s['sharpen'],sharpen_radius=s['sharpen_radius'],
                                curve_mode=s['curve_mode'],curve_params=(s['curve_shadows'],s['curve_darks'],s['curve_lights'],s['curve_highlights']),
                                curve_points=s['curve_points'])
                ok,data=cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                if not ok:raise ValueError('PNG 编码失败')
                data.tofile(path)
            self.event('log',text='已保存：'+str(path));return
    def publish(self):
        if self.meta is None:return
        s=self.settings;meta=self.meta;raw=self.raw
        # Cache in a common exposure scale, but present ADU at the current actual exposure.
        if s['mode']=='关闭':
            if self.single is None:return
            result=self.single
        else:
            if self.roll.total is None:return
            result=self.roll.result(s['mode'])*(meta.exposure_ms/100)
        self.pre_math=result
        avg=total=None
        if s['math_op'] in WINDOW_OPS:
            if self.roll.total is None:return
            if s['math_op']=='窗口积分':total=self.roll.result('积分')*(meta.exposure_ms/100)
            else:avg=self.roll.result('平均')*(meta.exposure_ms/100)
        result=pixel_math(result,s,avg,total,self.reference,self.window_local);self.processed=result
        counts,stats=histogram(result,'processed')
        if s['stretch']:
            s['black']=stats['p01'];s['white']=max(s['black']+1,stats['p995'])
        view=result
        if s.get('preview_crop'):
            x,y,cw,ch=s['preview_crop'];h,w=result.shape
            view=result[int(y*h):max(int(y*h)+1,int((y+ch)*h)),int(x*w):max(int(x*w)+1,int((x+cw)*w))]
        rgb=display_rgb(view,s['black'],s['white'],s['gamma'],s['palette'],custom_points=s['custom_points'],
                        contrast=s['contrast'],sharpen=s['sharpen'],sharpen_radius=s['sharpen_radius'],
                        curve_mode=s['curve_mode'],curve_params=(s['curve_shadows'],s['curve_darks'],s['curve_lights'],s['curve_highlights']),
                        curve_points=s['curve_points'])
        hs=s['hist_source']
        if hs in ('原始16位','相机原始灰度'):counts,hstats=histogram(raw,'sensor',meta.bits)
        elif hs=='显示灰度':
            gray=display_rgb(view,s['black'],s['white'],s['gamma'],'灰度',contrast=s['contrast'],sharpen=s['sharpen'],
                             sharpen_radius=s['sharpen_radius'],curve_mode=s['curve_mode'],
                             curve_params=(s['curve_shadows'],s['curve_darks'],s['curve_lights'],s['curve_highlights']),
                             curve_points=s['curve_points'])[:,:,0]
            counts,hstats=histogram(gray,'display')
        else:hstats=stats
        white=s['custom_points'][-1][0] if s['palette']=='自定义' else s['white']
        stats['white_clip']=float(np.mean(view[::4,::4]>=white)*100)
        with self.lock:
            self.serial+=1;self.latest=dict(serial=self.serial,rgb=rgb,counts=counts,stats=stats,
                hist_low=hstats['hist_low'],hist_high=hstats['hist_high'],hist_source=hs,
                frames=len(self.roll.frames),span=self.roll.span,stamp=self.last_frame_time or self.roll.last_time,shape=raw.shape,
                processed_shape=result.shape,mode=s['mode'],auto=s['auto'],ae_mode=s['ae_mode'],ae_status=self.ae_status,math_op=s['math_op'],stretch=s['stretch'],
                exposure=meta.exposure_ms,gain=meta.gain,bits=meta.bits,black=s['black'],white=s['white'],
                memory=(self.roll.bytes+(self.roll.total.nbytes if self.roll.total is not None else 0))/1024**2,sim=isinstance(self.cam,SimCamera))
    def calibration_choices(self):
        s=self.settings;m=self.meta;shape=self.raw.shape
        if not (s['dark'] or s['bias'] or s['flat']):return None,None
        gains={m.gain}|{x.meta.gain for x in self.library.masters if x.meta.camera==m.camera and x.meta.mode==m.mode and x.image.shape==shape}
        valid=[];pairs=[]
        for g in gains:
            candidate=replace(m,gain=g)
            if s['flat'] and self.library.find('flat',candidate,shape) is None:continue
            if s['bias'] and not s['dark'] and self.library.find('bias',candidate,shape) is None:continue
            valid.append(g)
            if s['dark']:
                pairs.extend((d.meta.exposure_ms,g) for d in self.library.masters if d.kind=='dark' and self.library.compatible(d,candidate,shape))
        return (pairs,None) if s['dark'] else (None,valid)
    def run(self):
        while self.running:
            try:
                while True:
                    try:action,k=self.commands.get_nowait()
                    except queue.Empty:break
                    try:self.command(action,k)
                    except Exception as e:
                        self.event('error',text=str(e))
                        if action=='settings':
                            self.event('settings',values={name:self.settings[name] for name in k['values'] if name in self.settings})
                            if self.cam and any(name in k['values'] for name in ('exposure','gain','resolution','native_bin')):
                                self.cam.stop();self.paused=True;self.roll.clear();self.event('paused',value=True)
                if not self.running:break
                if not self.cam or self.paused:time.sleep(.03);continue
                raw=self.cam.read()
                if raw is None:continue
                stamp=time.monotonic();meta=replace(self.cam.meta)
                self.raw=raw;self.meta=meta;self.last_frame_time=stamp
                if self.rec:self.rec.append(raw)
                if self.master:
                    m=self.master
                    if m['total'] is None:m['total']=np.zeros(raw.shape,np.float64)
                    m['total']+=raw;m['n']+=1
                    self.event('master',active=True,text=f"{m['kind']}：{m['n']} / {m['count']}")
                    if m['n']>=m['count']:
                        avg=(m['total']/m['n']).astype(np.float32)
                        try:
                            master=self.library.make_flat(avg,meta,m['n']) if m['kind']=='flat' else Master(m['kind'],avg,meta,m['n'])
                            self.library.add(master);self.event('log',text=f"已加入 {m['kind']}：{meta.exposure_ms:g} ms / 增益 {meta.gain:g}")
                        finally:self.cancel_master()
                    continue
                s=self.settings
                corrected=self.library.correct(raw,meta,s['dark'],s['bias'],s['flat'])
                self.single=bin_image(corrected,s['software_bin'])
                if s['mode']!='关闭' or s['math_op'] in WINDOW_OPS:
                    self.window_local=s['mode']=='关闭' and s['math_roi'] is not None
                    data=self.single[roi_slices(self.single.shape,s['math_roi'])] if self.window_local else self.single
                    linear=data*(100/max(meta.exposure_ms,.001));self.roll.push(linear,stamp)
                else:
                    if self.roll.total is not None:self.roll.clear()
                    self.window_local=False
                self.publish()
                if s['auto'] and stamp-self.last_ae>max(.5,meta.exposure_ms/1000):
                    self.last_ae=stamp
                    measured=measure_brightness(raw,s['ae_roi']);pairs,gains=self.calibration_choices()
                    slope=None
                    if self.ae_previous:
                        pe,pg,pm=self.ae_previous
                        if abs(pe-meta.exposure_ms)<.05 and abs(pg-meta.gain)>.5 and measured>0 and pm>0:
                            slope=math.log(measured/pm)/(meta.gain-pg)
                    self.ae_previous=(meta.exposure_ms,meta.gain,measured)
                    er=(max(s['ae_low'],self.cam.exp_range[0]),min(s['ae_high'],self.cam.exp_range[1]),self.cam.exp_range[2])
                    gr=(max(s['ae_gain_low'],self.cam.gain_range[0]),min(s['ae_gain_high'],self.cam.gain_range[1]),self.cam.gain_range[2])
                    exp,gain,self.ae_status=auto_step(meta.exposure_ms,meta.gain,measured,s['target']/100*((1<<meta.bits)-1),er,gr,s['ae_mode'],pairs,gains,slope)
                    if abs(exp-meta.exposure_ms)>.01 or abs(gain-meta.gain)>.0001:
                        gain_changed=abs(gain-meta.gain)>.0001
                        self.cam.configure(exposure=exp,gain=gain)
                        if gain_changed:self.roll.clear()
                        s.update(exposure=self.cam.meta.exposure_ms,gain=self.cam.meta.gain)
                        changed={}
                        if abs(s['exposure']-meta.exposure_ms)>.01:changed['exposure']=s['exposure']
                        if gain_changed:changed['gain']=s['gain']
                        self.event('telemetry',values=changed)
            except Exception as e:
                self.roll.clear();self.processed=None;self.event('error',text=str(e));self.paused=True
                if self.cam:self.cam.stop()
                self.end_record();self.event('paused',value=True)
        try:self.disconnect()
        except Exception:pass
