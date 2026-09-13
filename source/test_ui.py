import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import time
from pathlib import Path
from PySide6.QtWidgets import QApplication, QFileDialog
from PySide6.QtGui import QIcon, QPixmap
from main import MainWindow, STYLE

def pump(app,condition,timeout=5):
    start=time.monotonic()
    while time.monotonic()-start<timeout:
        app.processEvents();time.sleep(.02)
        if condition():return
    raise AssertionError('UI condition timed out')

def test_controls_preview_pause_config_crop(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([]);app.setStyleSheet(STYLE)
    icon=QIcon(str(Path(__file__).with_name('starfield.ico')))
    assert not icon.isNull() and {s.width() for s in icon.availableSizes()} >= {16,32,256}
    w=MainWindow();w.show()
    try:
        w.source.setCurrentIndex(1);w.connect_btn.click()
        pump(app,lambda:w.latest is not None)
        assert w.connected and '640' in w.live_label.text()
        assert w.capture_state.wordWrap()
        w.show_apply_status();assert '设置已提交' in w.statusBar().currentMessage()
        w._apply_status_timer.timeout.emit();assert w.statusBar().currentMessage()==''
        section=w.sections['连接设备'];section.header.click();app.processEvents();assert not section.body.isVisible()
        section.header.click();app.processEvents();assert section.body.isVisible()
        w.controls['mirror_horizontal'].click();pump(app,lambda:w.worker.settings['mirror_horizontal'])
        pump(app,lambda:w.latest.get('mirror_horizontal'))
        assert w.latest['mirror_horizontal']
        w.controls['curve_mode'].setCurrentIndex(w.controls['curve_mode'].findData('Camera Raw 参数曲线'));w.controls['curve_mode'].activated.emit(w.controls['curve_mode'].currentIndex())
        w.controls['contrast'].slider.setValue(1700);pump(app,lambda:abs(w.worker.settings['contrast']-70)<.2)
        assert w.curve_widget.mode=='Camera Raw 参数曲线'
        w.controls['seconds'].setValue(.25);w.apply_processing()
        w.set_crop((.1,.1,.5,.5));pump(app,lambda:w.worker.settings.get('preview_crop') is not None);pump(app,lambda:w.output.pix.width()==320 and w.output.pix.height()==240)
        assert w.output.pix.width()==320 and w.output.pix.height()==240
        w.clear_crop();n=w.seen;pump(app,lambda:w.seen>n+2)
        assert w.output.pix.width()==640
        w.pause_btn.click();pump(app,lambda:w.worker.paused)
        pump(app,lambda:w.output.message=='采集已暂停')
        assert not w.output.pix.isNull()
        n=w.seen;w.pause_btn.click();pump(app,lambda:w.seen>n)
        p=str(tmp_path/'config.json')
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(p,''))
        w.save_config();assert w.config_label.text()=='配置：config';w.controls['seconds'].setValue(8)
        monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a,**k:(p,''))
        w.load_config();assert w.controls['seconds'].value()==.25 and w.config_label.text()=='配置：config'
        w.output.show();app.processEvents();assert w.output.isVisible()
    finally:
        w.worker.send('quit');pump(app,lambda:not w.worker.is_alive());w.close();app.processEvents()

def test_preview_region_mapping_follows_mirror_and_crop():
    app=QApplication.instance() or QApplication([]);app.setStyleSheet(STYLE)
    from main import Preview
    view=Preview();view.item.setPixmap(QPixmap(100,100));view.image_region=(.25,0,.5,1);view.mirror_horizontal=True
    view.regions['ae']=((.4,.1,.2,.2),True);view.draw_regions()
    rect=view.region_items['ae'].rect()
    assert abs(rect.x()-30)<.01 and abs(rect.y()-10)<.01 and abs(rect.width()-40)<.01 and abs(rect.height()-20)<.01

def test_auto_can_be_disabled_and_live_controls_apply():
    app=QApplication.instance() or QApplication([]);w=MainWindow();w.show()
    try:
        w.source.setCurrentIndex(1);w.connect_camera();pump(app,lambda:w.latest is not None)
        w.worker.send('scene',scene='平场光源')
        c=w.controls['ae_mode'];c.setCurrentIndex(1);c.activated.emit(1);pump(app,lambda:w.worker.settings['auto'])
        pump(app,lambda:abs(w.worker.cam.meta.exposure_ms-100)>1)
        # Former bug: periodic exposure telemetry overwrote unsubmitted gain/mode/AE controls.
        w.controls['gain'].setValue(3)
        w.worker.event('telemetry',values={'exposure':75})
        w.poll();assert w.controls['gain'].value()==3
        c=w.controls['ae_mode'];c.setCurrentIndex(0);c.activated.emit(0);pump(app,lambda:not w.worker.settings['auto'])
        stop_exp=w.worker.cam.meta.exposure_ms;n=w.seen
        pump(app,lambda:w.seen>n+6)
        assert w.worker.cam.meta.exposure_ms==stop_exp and w.controls['ae_mode'].currentData()=='手动'
        w.controls['gain'].editingFinished.emit();pump(app,lambda:w.latest['gain']==3)
        c=w.controls['mode'];i=c.findData('积分');c.setCurrentIndex(i);c.activated.emit(i)
        pump(app,lambda:w.latest['mode']=='积分')
        i=c.findData('平均');c.setCurrentIndex(i);c.activated.emit(i);pump(app,lambda:w.latest['mode']=='平均')
        i=c.findData('最大值');c.setCurrentIndex(i);c.activated.emit(i);pump(app,lambda:w.latest['mode']=='最大值' and w.latest['effective_mode']=='最大值')
        d=w.controls['denoise_mode'];d.setCurrentIndex(d.findData('中值 3×3（去孤立噪点）'));d.activated.emit(d.currentIndex())
        w.controls['denoise_amount'].slider.setValue(3000);pump(app,lambda:w.worker.settings['denoise_mode'].startswith('中值') and w.worker.settings['denoise_amount']>29)
        low=w.controls['lowlight_mode'];low.setCurrentIndex(low.findData('星点/流星保护'));low.activated.emit(low.currentIndex())
        w.controls['lowlight_strength'].slider.setValue(6500);pump(app,lambda:w.worker.settings['lowlight_mode']=='星点/流星保护' and w.worker.settings['lowlight_strength']>64)
        pump(app,lambda:w.latest['lowlight_mode']=='星点/流星保护')
        assert w.latest['lowlight_mode']=='星点/流星保护'
        c=w.controls['resolution'];c.setCurrentIndex(1);c.activated.emit(1)
        pump(app,lambda:w.latest['shape']==(960,1280))
        assert not w.worker.settings['auto']
    finally:
        w.worker.send('quit');pump(app,lambda:not w.worker.is_alive());w.close();app.processEvents()

def test_metering_overlay_hidden_and_single_frame_has_no_cache():
    app=QApplication.instance() or QApplication([]);w=MainWindow();w.show()
    try:
        w.source.setCurrentIndex(1);w.connect_camera();pump(app,lambda:w.latest is not None)
        w.set_region('ae',(.1,.2,.3,.4));pump(app,lambda:w.worker.settings['ae_roi'] is not None)
        assert w.preview.region_items['ae'].isVisible()
        w.controls['ae_show'].click();app.processEvents()
        assert not w.preview.region_items['ae'].isVisible()
        assert w.worker.settings['ae_roi']==(.1,.2,.3,.4)
        c=w.controls['mode'];c.setCurrentIndex(0);c.activated.emit(0)
        pump(app,lambda:w.latest['mode']=='关闭' and w.latest['frames']==0)
        assert w.latest['memory']==0 and w.worker.roll.total is None
        w.set_crop((.25,.25,.5,.5));w.set_region('math',(0,0,.5,.5))
        assert w.advanced['math_roi']==(.25,.25,.25,.25)
    finally:
        w.worker.send('quit');pump(app,lambda:not w.worker.is_alive());w.close();app.processEvents()

def test_normal_wait_does_not_black_out_obs():
    app=QApplication.instance() or QApplication([]);w=MainWindow();w.show()
    try:
        w.source.setCurrentIndex(1);w.connect_camera();pump(app,lambda:w.latest is not None)
        w.worker.send('settings',values={'mode':'关闭','seconds':.1,'exposure':1500})
        pump(app,lambda:w.latest['exposure']==1500)
        start=time.monotonic()
        while time.monotonic()-start<.5:app.processEvents();time.sleep(.02)
        assert time.monotonic()-w.last_stamp>.1
        assert not w.output.pix.isNull() and '暂无' not in w.output.message
    finally:
        w.worker.send('quit');pump(app,lambda:not w.worker.is_alive());w.close();app.processEvents()

def test_bit_depth_frame_window_and_brightness_trigger():
    app=QApplication.instance() or QApplication([]);w=MainWindow();w.show()
    try:
        w.source.setCurrentIndex(1);w.connect_camera();pump(app,lambda:w.latest is not None)
        bits=w.controls['input_bits'];bits.setCurrentIndex(bits.findData(14));bits.activated.emit(bits.currentIndex())
        pump(app,lambda:w.latest is not None and w.latest['bits']==14)
        unit=w.controls['window_unit'];unit.setCurrentIndex(unit.findData('帧数'));unit.activated.emit(unit.currentIndex())
        w.controls['window_frames'].setValue(4);w.apply_processing()
        pump(app,lambda:w.worker.settings['window_unit']=='帧数' and w.worker.settings['window_frames']==4)
        device=w.controls['compute_device']
        assert device.findData('CPU')>=0 and device.findData('GPU（OpenCL）')>=0
        device.setCurrentIndex(device.findData('CPU'));device.activated.emit(device.currentIndex())
        pump(app,lambda:w.worker.compute.kind=='cpu' and w.latest.get('compute_kind')=='cpu')
        w.worker.send('settings',values={'mode':'关闭','trigger_condition':'平均亮度高于','trigger_threshold':0,'trigger_mode':'积分'})
        pump(app,lambda:w.latest is not None and w.latest['effective_mode']=='积分')
        assert w.latest['frames']<=4 and w.latest['raw_limit']==16383
    finally:
        w.worker.send('quit');pump(app,lambda:not w.worker.is_alive());w.close();app.processEvents()
