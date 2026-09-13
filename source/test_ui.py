import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import time
from PySide6.QtWidgets import QApplication, QFileDialog
from main import MainWindow, STYLE

def pump(app,condition,timeout=5):
    start=time.monotonic()
    while time.monotonic()-start<timeout:
        app.processEvents();time.sleep(.02)
        if condition():return
    raise AssertionError('UI condition timed out')

def test_controls_preview_pause_config_crop(tmp_path,monkeypatch):
    app=QApplication.instance() or QApplication([]);app.setStyleSheet(STYLE)
    w=MainWindow();w.show()
    try:
        w.source.setCurrentIndex(1);w.connect_btn.click()
        pump(app,lambda:w.latest is not None)
        assert w.connected and '640' in w.live_label.text()
        w.controls['curve_mode'].setCurrentIndex(w.controls['curve_mode'].findData('Camera Raw 参数曲线'));w.controls['curve_mode'].activated.emit(w.controls['curve_mode'].currentIndex())
        w.controls['contrast'].slider.setValue(1700);pump(app,lambda:abs(w.worker.settings['contrast']-70)<.2)
        assert w.curve_widget.mode=='Camera Raw 参数曲线'
        w.controls['seconds'].setValue(.25);w.apply_processing()
        w.set_crop((.1,.1,.5,.5));pump(app,lambda:w.worker.settings.get('preview_crop') is not None);n=w.seen;pump(app,lambda:w.seen>n+2)
        assert w.output.pix.width()==320 and w.output.pix.height()==240
        w.clear_crop();n=w.seen;pump(app,lambda:w.seen>n+2)
        assert w.output.pix.width()==640
        w.pause_btn.click();pump(app,lambda:w.worker.paused)
        pump(app,lambda:w.output.message=='采集已暂停')
        assert not w.output.pix.isNull()
        n=w.seen;w.pause_btn.click();pump(app,lambda:w.seen>n)
        p=str(tmp_path/'config.json')
        monkeypatch.setattr(QFileDialog,'getSaveFileName',lambda *a,**k:(p,''))
        w.save_config();w.controls['seconds'].setValue(8)
        monkeypatch.setattr(QFileDialog,'getOpenFileName',lambda *a,**k:(p,''))
        w.load_config();assert w.controls['seconds'].value()==.25
        w.output.show();app.processEvents();assert w.output.isVisible()
    finally:
        w.worker.send('quit');pump(app,lambda:not w.worker.is_alive());w.close();app.processEvents()

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
