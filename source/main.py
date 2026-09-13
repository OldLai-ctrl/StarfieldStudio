from __future__ import annotations
import sys, os, json, time, queue, copy
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QTimer, QRectF, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QPainterPath, QAction, QFontDatabase, QFont, QLinearGradient
from PySide6.QtWidgets import *
from worker import CaptureWorker, DEFAULTS
from compute import DEVICE_CHOICES
from camera import default_sdk
from camera_common import BACKENDS, sdk_path
from processing import *

STYLE='''
QWidget { background:#141b25; color:#dce5ee; font-family:"Microsoft YaHei UI"; font-size:12px; }
QMainWindow,QGraphicsView { background:#0b1017; }
QGroupBox { border:1px solid #344154; border-radius:6px; margin-top:12px; padding:12px 8px 8px; font-weight:600; }
QGroupBox::title { subcontrol-origin:margin; left:10px; color:#8ddfd8; }
QPushButton { background:#263447; border:1px solid #415168; border-radius:4px; padding:7px 10px; }
QPushButton:hover { background:#34526a; } QPushButton:disabled {color:#687383;}
QPushButton#primary { background:#17796e; border-color:#31b8a5; font-weight:600; }
QLineEdit,QDoubleSpinBox,QSpinBox,QComboBox { background:#0f1620; border:1px solid #39485a; border-radius:3px; padding:5px; }
QTabWidget::pane {border:0;} QTabBar::tab {padding:10px 12px;background:#1e2938;}
QTabBar::tab:selected {background:#304355;color:#91e6dc;}
QScrollArea {border:0;} QPlainTextEdit {background:#0f1620;border:0;}
QToolBar {spacing:8px;padding:7px;border-bottom:1px solid #354154;}
QCheckBox {spacing:7px;} QStatusBar {background:#1c2938;}
'''

class SliderControl(QWidget):
    """A compact continuous slider with a precise numeric readout."""
    valueChanged=Signal(float)
    editingFinished=Signal()
    def __init__(self,low,high,value,decimals=1,suffix=''):
        super().__init__();self._low=float(low);self._high=float(high);self._guard=False
        self.slider=QSlider(Qt.Orientation.Horizontal);self.spin=QDoubleSpinBox()
        self.spin.setRange(self._low,self._high);self.spin.setDecimals(decimals);self.spin.setSingleStep(10**(-decimals));self.spin.setValue(value)
        if suffix:self.spin.setSuffix(suffix)
        self._set_slider_steps()
        row=QHBoxLayout(self);row.setContentsMargins(0,0,0,0);row.setSpacing(6);row.addWidget(self.slider,1);row.addWidget(self.spin)
        self.slider.valueChanged.connect(self._slider_changed);self.spin.valueChanged.connect(self._spin_changed);self.spin.editingFinished.connect(self.editingFinished)
        self.setValue(value)
    def _set_slider_steps(self):
        self._steps=max(1000,int(round((self._high-self._low)*10**self.spin.decimals())))
        self.slider.setRange(0,self._steps)
    def _to_position(self,value):
        return int(round((float(value)-self._low)/(self._high-self._low)*self._steps)) if self._high>self._low else 0
    def _from_position(self,value):return self._low+(self._high-self._low)*int(value)/self._steps
    def _slider_changed(self,value):
        if self._guard:return
        self._guard=True;self.spin.setValue(self._from_position(value));self._guard=False;self.valueChanged.emit(self.value())
    def _spin_changed(self,value):
        if self._guard:return
        self._guard=True;self.slider.setValue(self._to_position(value));self._guard=False;self.valueChanged.emit(float(value))
    def value(self):return float(self.spin.value())
    def setValue(self,value):
        self._guard=True;self.spin.setValue(float(value));self.slider.setValue(self._to_position(value));self._guard=False
    def setRange(self,low,high):
        self._guard=True;self._low=float(low);self._high=float(high);self.spin.setRange(self._low,self._high);self._set_slider_steps();self.slider.setValue(self._to_position(self.spin.value()));self._guard=False
    def setSingleStep(self,step):self.spin.setSingleStep(step)
    def hasFocus(self):return self.spin.hasFocus() or self.slider.hasFocus()
    def blockSignals(self,block):
        super().blockSignals(block);self.spin.blockSignals(block);self.slider.blockSignals(block)

class CurveWidget(QWidget):
    def __init__(self):
        super().__init__();self.mode='关闭';self.params=(0.,0.,0.,0.);self.points=DEFAULT_CURVE_POINTS;self.setMinimumHeight(150);self.setToolTip('预览当前色调曲线；横轴为输入亮度，纵轴为输出亮度')
    def set_curve(self,mode,params,points):self.mode=mode;self.params=tuple(params);self.points=points;self.update()
    def paintEvent(self,e):
        p=QPainter(self);p.fillRect(self.rect(),QColor('#0e151e'));r=self.rect().adjusted(30,12,-15,-26)
        p.setPen(QPen(QColor('#293c4b'),1))
        for i in range(5):
            x=r.left()+r.width()*i/4;y=r.top()+r.height()*i/4;p.drawLine(int(x),r.top(),int(x),r.bottom());p.drawLine(r.left(),int(y),r.right(),int(y))
        p.setPen(QPen(QColor('#68798a'),1,Qt.PenStyle.DashLine));p.drawLine(r.bottomLeft(),r.topRight())
        grid=np.linspace(0,1,256)
        if self.mode=='Camera Raw 参数曲线':vals=camera_raw_curve(grid,*self.params)
        elif self.mode=='点曲线':
            try:pts=validate_curve_points(self.points);vals=np.interp(grid,[x for x,y in pts],[y for x,y in pts])
            except Exception:vals=grid
        else:vals=grid
        path=QPainterPath();path.moveTo(r.left(),r.bottom()-float(vals[0])*r.height())
        for x,y in zip(grid[1:],vals[1:]):path.lineTo(r.left()+float(x)*r.width(),r.bottom()-float(y)*r.height())
        p.setPen(QPen(QColor('#f0c56c') if self.mode!='关闭' else QColor('#6f8798'),2));p.drawPath(path)
        p.setPen(QColor('#99aabe'));p.drawText(4,self.height()-8,'输出');p.drawText(self.width()-32,self.height()-8,'输入')

class CurveDialog(QDialog):
    def __init__(self,points,parent=None):
        super().__init__(parent);self.setWindowTitle('点曲线 · Camera Raw 风格');self.resize(560,480)
        layout=QVBoxLayout(self);note=QLabel('编辑输入亮度到输出亮度的控制点。输入和输出均为百分比；曲线自动在点之间平滑插值。参数曲线模式可直接使用主界面的四个滑块。');note.setWordWrap(True);layout.addWidget(note)
        self.table=QTableWidget(0,2);self.table.setHorizontalHeaderLabels(['输入亮度 %','输出亮度 %']);self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);layout.addWidget(self.table,1)
        for x,y in points:self.add_point(x,y)
        row=QHBoxLayout();add=QPushButton('添加控制点');add.clicked.connect(lambda:self.add_point(.5,.5));row.addWidget(add);delete=QPushButton('删除选中行');delete.clicked.connect(lambda:self.table.removeRow(self.table.currentRow()));row.addWidget(delete);layout.addLayout(row)
        self.graph=CurveWidget();self.graph.set_curve('点曲线',(),points);layout.addWidget(self.graph)
        self.table.itemChanged.connect(lambda *_:self.refresh())
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
    def add_point(self,x,y):
        i=self.table.rowCount();self.table.insertRow(i);self.table.setItem(i,0,QTableWidgetItem(f'{float(x)*100:g}'));self.table.setItem(i,1,QTableWidgetItem(f'{float(y)*100:g}'))
    def points(self):
        return validate_curve_points([[float(self.table.item(i,0).text())/100,float(self.table.item(i,1).text())/100] for i in range(self.table.rowCount())])
    def refresh(self):
        try:self.graph.set_curve('点曲线',(),self.points())
        except Exception:pass
    def accept(self):
        try:self.points()
        except Exception as e:QMessageBox.warning(self,'曲线无效',str(e));return
        super().accept()

class Preview(QGraphicsView):
    cropChanged=Signal(object)
    regionSelected=Signal(str,object)
    def __init__(self):
        super().__init__();self.scene_=QGraphicsScene(self);self.setScene(self.scene_)
        self.item=self.scene_.addPixmap(QPixmap());self.fit=True;self.cropping=False;self.origin=None
        self.box=self.scene_.addRect(QRectF(),QPen(QColor('#53dcca'),2));self.box.setZValue(3)
        self.selection_kind='crop';self.regions={};self.image_region=(0,0,1,1)
        self.region_items={k:self.scene_.addRect(QRectF(),QPen(QColor(c),2,Qt.PenStyle.DashLine)) for k,c in [('ae','#52eac6'),('math','#ffc569')]}
        for r in self.region_items.values():r.setZValue(4)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setMinimumSize(320,240)
    def set_image(self,pix):
        self.item.setPixmap(pix);self.scene_.setSceneRect(QRectF(pix.rect()))
        self.draw_regions()
        if self.fit:self.fitInView(self.item,Qt.AspectRatioMode.KeepAspectRatio)
    def fit_image(self):self.fit=True;self.fitInView(self.item,Qt.AspectRatioMode.KeepAspectRatio)
    def draw_regions(self):
        b=self.item.boundingRect();cx,cy,cw,ch=self.image_region
        for key,item in self.region_items.items():
            roi,visible=self.regions.get(key,(None,False))
            if roi is None or not visible:item.hide();continue
            x,y,w,h=roi
            r=QRectF((x-cx)/cw*b.width(),(y-cy)/ch*b.height(),w/cw*b.width(),h/ch*b.height()).intersected(b)
            item.setRect(r);item.setVisible(not r.isEmpty())
    def wheelEvent(self,e):
        self.fit=False;v=1.2 if e.angleDelta().y()>0 else 1/1.2
        if .03<self.transform().m11()*v<50:self.scale(v,v)
    def resizeEvent(self,e):
        super().resizeEvent(e)
        if self.fit:self.fit_image()
    def mousePressEvent(self,e):
        if self.cropping and e.button()==Qt.MouseButton.LeftButton:
            self.origin=self.mapToScene(e.position().toPoint());e.accept();return
        super().mousePressEvent(e)
    def mouseMoveEvent(self,e):
        if self.origin is not None:
            self.box.setRect(QRectF(self.origin,self.mapToScene(e.position().toPoint())).normalized().intersected(self.item.boundingRect()));return
        super().mouseMoveEvent(e)
    def mouseReleaseEvent(self,e):
        if self.origin is not None:
            r=self.box.rect();self.origin=None;self.cropping=False
            if r.width()>4 and r.height()>4:
                b=self.item.boundingRect();selection=(r.x()/b.width(),r.y()/b.height(),r.width()/b.width(),r.height()/b.height())
                if self.selection_kind=='crop':self.cropChanged.emit(selection)
                else:self.regionSelected.emit(self.selection_kind,selection)
            self.box.setRect(QRectF())
            return
        super().mouseReleaseEvent(e)

class Histogram(QWidget):
    def __init__(self):super().__init__();self.counts=np.zeros(256);self.low=0;self.high=65536;self.setMinimumHeight(180)
    def paintEvent(self,e):
        p=QPainter(self);p.fillRect(self.rect(),QColor('#0e151e'));r=self.rect().adjusted(15,20,-12,-26)
        p.setPen(QColor('#293c4b'))
        for i in range(1,4):p.drawLine(r.left(),r.top()+r.height()*i//4,r.right(),r.top()+r.height()*i//4)
        v=np.log10(self.counts+1);v/=max(float(v.max()),1)
        path=QPainterPath();path.moveTo(r.left(),r.bottom())
        for i,x in enumerate(v):path.lineTo(r.left()+r.width()*i/255,r.bottom()-float(x)*r.height())
        path.lineTo(r.right(),r.bottom());p.fillPath(path,QColor('#225b61'));p.setPen(QPen(QColor('#6fdfcd'),1));p.drawPath(path)
        p.setPen(QColor('#99aabe'));p.drawText(15,self.height()-7,f'{self.low:g}');p.drawText(self.width()-85,self.height()-7,f'{self.high-1:.6g}')
        p.drawText(15,14,'像素数 · 对数显示')

class OutputWindow(QWidget):
    def __init__(self):
        super().__init__();self.setWindowTitle('Starfield Studio — OBS');self.resize(960,640)
        self.pix=QPixmap();self.message='等待有效画面';self.setStyleSheet('background:black;')
    def paintEvent(self,e):
        p=QPainter(self);p.fillRect(self.rect(),Qt.GlobalColor.black)
        if not self.pix.isNull():
            size=self.pix.size().scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio)
            p.drawPixmap((self.width()-size.width())//2,(self.height()-size.height())//2,self.pix.scaled(size,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        if self.message:
            p.fillRect(0,0,self.width(),35,QColor(0,0,0,200));p.setPen(QColor('#ffcf7d'));p.drawText(12,23,self.message)

class PaletteDialog(QDialog):
    def __init__(self,points,parent=None):
        super().__init__(parent);self.setWindowTitle('自定义伪彩 · 灰度与颜色');self.resize(580,470)
        layout=QVBoxLayout(self);note=QLabel('按处理后的实际灰度映射颜色，控制点之间平滑过渡。双击颜色格可打开取色器。此映射不经过显示黑白点或伽马。');note.setWordWrap(True);layout.addWidget(note)
        self.table=QTableWidget(0,2);self.table.setHorizontalHeaderLabels(['灰度值','颜色 #RRGGBB · 双击取色'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);layout.addWidget(self.table)
        self.table.cellDoubleClicked.connect(self.pick_color)
        for value,color in points:self.add_point(value,color)
        row=QHBoxLayout();add=QPushButton('添加控制点');add.clicked.connect(lambda:self.add_point(1000,'#ffffff'));row.addWidget(add)
        delete=QPushButton('删除选中行');delete.clicked.connect(lambda:self.table.removeRow(self.table.currentRow()));row.addWidget(delete)
        layout.addLayout(row);self.gradient=QLabel('');self.gradient.setFixedHeight(30);layout.addWidget(self.gradient)
        preview=QPushButton('预览渐变');preview.clicked.connect(self.preview_colors);layout.addWidget(preview)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
        self.preview_colors()
    def add_point(self,value,color):
        i=self.table.rowCount();self.table.insertRow(i);self.table.setItem(i,0,QTableWidgetItem(f'{value:g}'))
        item=QTableWidgetItem(color);item.setBackground(QColor(color));item.setForeground(QColor('#ffffff') if QColor(color).lightness()<128 else QColor('#000000'));self.table.setItem(i,1,item)
    def pick_color(self,row,col):
        if col!=1:return
        current=self.table.item(row,1)
        color=QColorDialog.getColor(QColor(current.text()),self,'选择映射颜色',QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if color.isValid():current.setText(color.name());current.setBackground(color);current.setForeground(QColor('white') if color.lightness()<128 else QColor('black'))
    def points(self):
        return validate_points([[float(self.table.item(i,0).text()),self.table.item(i,1).text()] for i in range(self.table.rowCount())])
    def preview_colors(self):
        try:
            points=self.points();pix=QPixmap(520,30);p=QPainter(pix);g=QLinearGradient(0,0,520,0);lo,hi=points[0][0],points[-1][0]
            for value,color in points:g.setColorAt((value-lo)/(hi-lo),QColor(color))
            p.fillRect(pix.rect(),g);p.end();self.gradient.setPixmap(pix)
        except Exception:self.gradient.setText('请填写不同的灰度值及有效颜色')
    def accept(self):
        try:self.points()
        except Exception as e:QMessageBox.warning(self,'控制点无效',str(e));return
        super().accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('Starfield Studio 2.5 | 工业黑白相机直播');self.resize(1430,920)
        self.worker=CaptureWorker();self.controls={};self.connected=False;self.recording=False;self.seen=0
        self.crop=None;self.latest=None;self.last_stamp=0;self.fps=0;self.master_active=False
        self.advanced={k:copy.deepcopy(DEFAULTS[k]) for k in ('ae_roi','math_roi','custom_points','curve_points')}
        self.output=OutputWindow();self.output.hide()
        self._slider_timers={};self.build();self.set_values(DEFAULTS);self.wire_controls();self.worker.start();self.timer=QTimer(self);self.timer.timeout.connect(self.poll);self.timer.start(50)
    def wire_controls(self):
        for key,c in self.controls.items():
            if isinstance(c,QCheckBox):c.clicked.connect(lambda checked=False,k=key:self.commit_control(k))
            elif isinstance(c,QComboBox):c.activated.connect(lambda index,k=key:self.commit_control(k))
            elif isinstance(c,SliderControl):c.valueChanged.connect(lambda value,k=key:self.queue_slider(k))
            else:c.editingFinished.connect(lambda k=key:self.commit_control(k))
    def queue_slider(self,key):
        if key.startswith('curve_'):self.refresh_curve()
        timer=self._slider_timers.get(key)
        if timer is None:
            timer=QTimer(self);timer.setSingleShot(True);timer.setInterval(45);timer.timeout.connect(lambda k=key:self.commit_control(k));self._slider_timers[key]=timer
        timer.start()
    def commit_control(self,key):
        v=self.values([key])
        if v[key] is None:return
        if key in ('exposure','gain'):
            v.update(auto=False,ae_mode='手动');self.set_values({'ae_mode':'手动'})
        if key in ('black','white'):
            v['stretch']=False;self.set_values({'stretch':False})
        if key in ('ae_low','ae_high') and self.controls['ae_low'].value()>self.controls['ae_high'].value():
            self.show_error('最短曝光不能大于最长曝光');return
        if key in ('black','white') and self.controls['white'].value()<=self.controls['black'].value():
            self.show_error('亮点必须大于暗点');return
        self.worker.send('settings',values=v)
        if key in ('ae_show','math_show'):self.refresh_regions()
        if key=='curve_mode' or key.startswith('curve_'):self.refresh_curve()
        self.statusBar().showMessage('正在应用设置；右侧显示实际输出状态')
    def spin(self,key,low,high,value,decimals=2):
        c=QDoubleSpinBox();c.setRange(low,high);c.setDecimals(decimals);c.setValue(value);c.setKeyboardTracking(False)
        self.controls[key]=c;return c
    def slider(self,key,low,high,value,decimals=1,suffix=''):
        c=SliderControl(low,high,value,decimals,suffix);self.controls[key]=c;return c
    def check(self,key,text,value=False):
        c=QCheckBox(text);c.setChecked(value);self.controls[key]=c;return c
    def combo(self,key,values):
        c=QComboBox()
        for text,data in values:c.addItem(text,data)
        self.controls[key]=c;return c
    def button(self,text,fn,primary=False):
        b=QPushButton(text);b.clicked.connect(fn)
        if primary:b.setObjectName('primary')
        return b
    def group(self,title,parent):
        g=QGroupBox(title);f=QFormLayout(g);f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow);parent.addWidget(g);return f
    def page(self,tabs,title):
        w=QWidget();lay=QVBoxLayout(w);lay.setContentsMargins(8,6,8,8)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setWidget(w);scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff);tabs.addTab(scroll,title);return lay
    def build(self):
        bar=QToolBar();self.addToolBar(bar)
        title=QLabel('  STARFIELD  /  星野直播  ');title.setStyleSheet('font-size:17px;font-weight:700;color:#89e4d7;');bar.addWidget(title)
        self.pause_btn=self.button('暂停采集',lambda:self.worker.send('pause'));bar.addWidget(self.pause_btn)
        bar.addWidget(self.button('清空滚动窗口',lambda:self.worker.send('reset')))
        bar.addWidget(self.button('保存图像',self.save_image))
        self.rec_btn=self.button('录制原始 SER',self.record);bar.addWidget(self.rec_btn)
        bar.addWidget(self.button('OBS 纯画面',self.output.show,True))
        bar.addWidget(self.button('保存配置',self.save_config));bar.addWidget(self.button('加载配置',self.load_config))
        split=QSplitter();self.setCentralWidget(split)
        tabs=QTabWidget();tabs.setMinimumWidth(320);tabs.setMaximumWidth(420);split.addWidget(tabs)
        lay=self.page(tabs,'相机 / 叠加')
        f=self.group('连接设备',lay)
        self.source=QComboBox();self.source.addItems(['工业黑白相机','模拟星野 · 用于试用'])
        f.addRow('图像来源',self.source)
        self.backend=QComboBox()
        for key,(label,_) in BACKENDS.items():self.backend.addItem(label,key)
        self.sdk_paths={key:sdk_path(key) for key in BACKENDS};self.previous_backend='tucam'
        f.addRow('相机接口类型',self.backend)
        self.dll=QLineEdit(default_sdk());self.dll.setToolTip('官方 x64 TUCam.dll 所在路径');f.addRow('相机接口',self.dll)
        f.addRow(self.button('选择接口文件…',self.choose_dll))
        self.backend.currentIndexChanged.connect(self.backend_changed)
        self.backend_note=QLabel('面向部分工业黑白相机；按接口读取实际能力。');self.backend_note.setWordWrap(True);f.addRow(self.backend_note)
        self.index=QSpinBox();self.index.setRange(0,15);f.addRow('设备序号',self.index)
        self.connect_btn=self.button('连接相机',self.connect_camera,True);f.addRow(self.connect_btn)
        self.device_label=QLabel('尚未连接');self.device_label.setWordWrap(True);f.addRow(self.device_label)
        f=self.group('采集参数',lay)
        f.addRow('曝光 / ms',self.spin('exposure',.001,15000,100,3))
        f.addRow('增益 / 驱动单位',self.spin('gain',0,1000,1,3))
        f.addRow('输入位深',self.combo('input_bits',[('默认 Mono16',16)]))
        f.addRow('分辨率 / 相机合并',self.combo('resolution',[('连接后读取',None)]))
        f.addRow('相机 Binning',self.combo('native_bin',[('连接后读取',None)]))
        f.addRow('软件 Binning',self.combo('software_bin',[(f'{n} × {n} · 平均',n) for n in (1,2,3,4)]))
        f.addRow(self.button('应用采集参数',self.apply_capture))
        f=self.group('目标亮度自动曝光',lay)
        f.addRow('调整策略',self.combo('ae_mode',[(s,s) for s in ['手动','优先曝光时长','优先增益']]))
        f.addRow('目标亮度 / %',self.spin('target',.1,95,20,1))
        f.addRow('最短曝光 / ms',self.spin('ae_low',.001,15000,5,3))
        f.addRow('最长曝光 / ms',self.spin('ae_high',.001,15000,5000,3))
        f.addRow('最低增益',self.spin('ae_gain_low',0,1000,1,3))
        f.addRow('最高增益',self.spin('ae_gain_high',0,1000,257,3))
        f.addRow(self.button('在预览图框选测光区域',lambda:self.begin_region('ae')))
        f.addRow(self.button('恢复全画面测光',lambda:self.set_region('ae',None)))
        f.addRow(self.check('ae_show','显示测光选框',True))
        self.ae_roi_label=QLabel('测光：全画面');self.ae_roi_label.setWordWrap(True);f.addRow(self.ae_roi_label)
        note=QLabel('先调优先项，到边界后再调另一项。测量选区原始灰度的90%分位；隐藏选框不改变测光。启用校正时只使用匹配档位。');note.setWordWrap(True);f.addRow(note)
        f=self.group('最近几秒 · 滚动处理',lay)
        f.addRow('窗口单位',self.combo('window_unit',[('按时间','时间'),('按帧数','帧数')]))
        f.addRow('窗口时长 / 秒',self.spin('seconds',.05,120,3,2))
        f.addRow('窗口帧数',self.spin('window_frames',1,100000,30,0))
        f.addRow('运算方式',self.combo('mode',[('关闭叠加 · 单帧','关闭'),('滚动帧平均','平均'),('滚动帧积分','积分'),('窗口最大值 · 保留星点/流星','最大值')]))
        compute_combo=self.combo('compute_device',[(s,s) for s in DEVICE_CHOICES])
        compute_combo.setToolTip('自动优先使用 OpenCL GPU；GPU（OpenCL）适用于安装了驱动运行时的 NVIDIA 与 Intel 显卡。不可用时会自动回退 CPU。')
        f.addRow('计算设备',compute_combo)
        f.addRow('亮度触发条件',self.combo('trigger_condition',[(s,s) for s in ['关闭','平均亮度低于','平均亮度高于']]))
        f.addRow('触发阈值 / %',self.spin('trigger_threshold',0,100,20,1))
        f.addRow('触发后切换为',self.combo('trigger_mode',[('滚动帧平均','平均'),('滚动帧积分','积分'),('窗口最大值 · 保留星点/流星','最大值'),('关闭叠加 · 单帧','关闭')]))
        budget=self.spin('memory',0,65536,0,0);budget.setSpecialValueText('自动 · 按可用内存扩展')
        budget.setToolTip('0 为自动扩展，并给系统保留可用内存；大于0为手动硬上限。不会丢弃窗口内的帧。')
        f.addRow('缓存 / MB（0=自动）',budget)
        f.addRow(self.button('应用自动曝光 / 叠加',self.apply_processing,True))
        note=QLabel('窗口可按时间或帧数限制；改变单位、时长或帧数后从下一帧重新计数。最大值模式逐像素保留窗口内最亮值，适合星点和流星；触发器使用当前窗口全部帧的平均亮度，达到条件时切换到指定模式。');note.setWordWrap(True);f.addRow(note)
        lay.addStretch()
        lay=self.page(tabs,'校正 / 显示')
        f=self.group('校正主帧',lay)
        for key,label in [('bias','启用偏置校正'),('dark','启用暗场校正'),('flat','启用平场校正')]:f.addRow(self.check(key,label))
        self.master_count=QSpinBox();self.master_count.setRange(1,256);self.master_count.setValue(32);f.addRow('平均帧数',self.master_count)
        for kind,label in [('bias','拍摄偏置'),('dark','拍摄当前曝光暗场'),('flat','拍摄平场')]:
            f.addRow(self.button(label,lambda checked=False,k=kind:self.capture_master(k)))
        f.addRow(self.button('取消校正帧采集',lambda:self.worker.send('cancel_master')))
        f.addRow(self.button('应用校正开关',self.apply_processing,True))
        f.addRow(self.button('保存校正库…',lambda:self.library_file(True)))
        f.addRow(self.button('加载校正库…',lambda:self.library_file(False)))
        self.master_label=QLabel('未拍摄校正帧');self.master_label.setWordWrap(True);f.addRow(self.master_label)
        note=QLabel('暗场已含偏置，不会重复扣除。平场需先拍摄偏置或同曝光暗场。改变镜头、光圈或滤镜后请重拍平场。');note.setWordWrap(True);f.addRow(note)
        f=self.group('亮暗平衡 · 仅影响显示',lay)
        f.addRow(self.check('stretch','持续自动拉伸（抵消亮度变化）',False))
        f.addRow(self.button('拉伸一次，然后固定亮暗点',lambda:self.worker.send('auto_levels')))
        f.addRow('暗点 / 黑场电平',self.spin('black',-1e7,1e9,0,1))
        f.addRow('亮点 / 白场电平',self.spin('white',-1e7,1e9,8000,1))
        f.addRow('伽马',self.spin('gamma',.05,10,1,2))
        f.addRow('伪彩',self.combo('palette',[(s,s) for s in ['灰度','火焰','青蓝','科学色','自定义']]))
        f.addRow(self.button('编辑自定义灰度—颜色…',self.edit_palette))
        f=self.group('实时图像调整 · 仅预览 / OBS',lay)
        f.addRow('对比度',self.slider('contrast',-100,100,0,1,'%'))
        f.addRow('锐化强度',self.slider('sharpen',0,300,0,1,'%'))
        f.addRow('锐化半径',self.slider('sharpen_radius',.1,20,1,2,' px'))
        f.addRow('曲线模式',self.combo('curve_mode',[(s,s) for s in CURVE_MODES]))
        f.addRow('阴影',self.slider('curve_shadows',-100,100,0,1,'%'))
        f.addRow('暗部',self.slider('curve_darks',-100,100,0,1,'%'))
        f.addRow('亮部',self.slider('curve_lights',-100,100,0,1,'%'))
        f.addRow('高光',self.slider('curve_highlights',-100,100,0,1,'%'))
        f.addRow(self.button('编辑点曲线…',self.edit_curve))
        self.curve_widget=CurveWidget();f.addRow(self.curve_widget)
        f.addRow(self.button('应用显示设置',self.apply_display))
        note=QLabel('调整只作用于预览、OBS 和预览 PNG，不改动原始 TIFF／SER／FITS。Camera Raw 参数曲线按阴影、暗部、亮部、高光分区调整；点曲线可编辑输入—输出控制点。锐化采用实时反遮罩，半径越大影响范围越宽。');note.setWordWrap(True);f.addRow(note)
        f=self.group('即时降噪 · 仅预览 / OBS',lay)
        f.addRow('降噪模式',self.combo('denoise_mode',[(s,s) for s in DENOISE_MODES]))
        f.addRow('降噪强度',self.slider('denoise_amount',0,100,50,1,'%'))
        note=QLabel('采用低开销的局部滤波，不增加滚动缓存。中值 3×3 更适合去除孤立热像素；高斯 3×3 更适合轻度压低随机噪声。强度为0等同关闭。单像素星点可能被中值滤波削弱，拍摄星点或流星时建议关闭或使用较低强度。');note.setWordWrap(True);f.addRow(note)
        f=self.group('弱光增强预设 · 仅预览 / OBS',lay)
        f.addRow('增强模式',self.combo('lowlight_mode',[(s,s) for s in LOWLIGHT_MODES]))
        f.addRow('增强强度',self.slider('lowlight_strength',0,100,50,1,'%'))
        note=QLabel('参考公开的弱光 ISP 思路：自适应提亮暗部，或加一层低开销局部降噪；“星点/流星保护”只在暗部提亮，尽量保持亮点和超出统计白点的流星。它不复制任何厂商闭源模型，也不改变原始帧、校正帧或滚动缓存。需要时间降噪时请配合上方滚动平均/积分；追踪流星建议使用窗口最大值并降低降噪强度。');note.setWordWrap(True);f.addRow(note)
        lay.addStretch()
        lay=self.page(tabs,'像素运算')
        f=self.group('局部像素运算 · 默认关闭',lay)
        f.addRow('运算',self.combo('math_op',[(s,s) for s in MATH_OPS]))
        f.addRow(self.button('在预览图框选运算区域',lambda:self.begin_region('math')))
        f.addRow(self.button('恢复全画面运算',lambda:self.set_region('math',None)))
        f.addRow(self.check('math_show','显示运算选框',True))
        self.math_roi_label=QLabel('运算区域：全画面');self.math_roi_label.setWordWrap(True);f.addRow(self.math_roi_label)
        f.addRow('灰度筛选下限',self.spin('math_low',-1e9,1e9,-1e9,2))
        f.addRow('灰度筛选上限',self.spin('math_high',-1e9,1e9,1e9,2))
        coef=self.spin('math_k',-1e6,1e6,1,3);coef.setToolTip('仅用于加常数、乘系数、幂律；幂律指数范围0.05至8。')
        f.addRow('常数 / 系数 / 指数',coef)
        f.addRow('非线性灰度尺度',self.spin('math_scale',.001,1e9,65535,3))
        f.addRow('限幅下限',self.spin('math_clip_low',-1e9,1e9,0,2))
        f.addRow('限幅上限',self.spin('math_clip_high',-1e9,1e9,65535,2))
        f.addRow(self.button('记录当前画面为参考帧',lambda:self.worker.send('reference')))
        note=QLabel('运算在校正／叠加之后、伪彩之前执行。只修改同时符合画面选区与灰度范围的像素。窗口运算使用最近几秒；参考帧固定不更新。加常数可填负值。幂律、对数、平方根保留负值符号。一次选择一种运算。');note.setWordWrap(True);f.addRow(note)
        note=QLabel('关闭叠加且未使用窗口运算时不缓存多帧。关闭叠加但局部使用窗口运算时，只缓存该选区。');note.setWordWrap(True);f.addRow(note)
        lay.addStretch()
        center=QWidget();cl=QVBoxLayout(center);cl.setContentsMargins(8,8,8,8)
        self.live_label=QLabel('准备采集');self.live_label.setStyleSheet('color:#91e4d4;font-size:14px;');cl.addWidget(self.live_label)
        self.capture_state=QLabel('');cl.addWidget(self.capture_state)
        self.preview=Preview();self.preview.cropChanged.connect(self.set_crop);self.preview.regionSelected.connect(self.set_region);cl.addWidget(self.preview,1)
        row=QHBoxLayout();row.addWidget(self.button('适合窗口',self.preview.fit_image))
        row.addWidget(self.button('1:1 预览像素',self.one_to_one));row.addWidget(self.button('框选直播裁切',self.begin_crop))
        row.addWidget(self.button('取消裁切',self.clear_crop));cl.addLayout(row)
        self.caption=QLabel('滚轮缩放 · 拖动平移 · 裁切作用于预览和 OBS，不改变原始采集');self.caption.setWordWrap(True);cl.addWidget(self.caption)
        split.addWidget(center)
        right=QWidget();right.setMinimumWidth(235);right.setMaximumWidth(290);rl=QVBoxLayout(right)
        rl.addWidget(QLabel('直方图来源'))
        rl.addWidget(self.combo('hist_source',[(s,s) for s in ['相机原始灰度','处理后','显示灰度']]))
        self.hist=Histogram();rl.addWidget(self.hist)
        self.stats=QLabel('等待有效图像');self.stats.setWordWrap(True);self.stats.setStyleSheet('line-height:1.5;');rl.addWidget(self.stats)
        note=QLabel('原始：按接口位深显示量程。处理后：包含负值及积分超量程，横轴自动扩展。显示灰度：伪彩前0–255。纵轴均为对数。');note.setWordWrap(True);rl.addWidget(note)
        rl.addWidget(QLabel('运行记录'));self.log=QPlainTextEdit();self.log.setReadOnly(True);self.log.setMaximumBlockCount(150);rl.addWidget(self.log,1)
        split.addWidget(right);split.setSizes([345,800,255]);self.statusBar().showMessage('相机原始数据 → 校正 → 滚动处理 → 预览 / OBS')
    def values(self,keys):
        out={}
        for k in keys:
            if k in self.advanced:out[k]=copy.deepcopy(self.advanced[k]);continue
            if k not in self.controls:continue
            c=self.controls[k]
            out[k]=c.isChecked() if isinstance(c,QCheckBox) else c.currentData() if isinstance(c,QComboBox) else c.value()
        return out
    def set_values(self,vals):
        for k,v in vals.items():
            if k in self.advanced:self.advanced[k]=copy.deepcopy(v);continue
            if k not in self.controls:continue
            c=self.controls[k];c.blockSignals(True)
            if isinstance(c,QCheckBox):c.setChecked(bool(v))
            elif isinstance(c,QComboBox):
                i=c.findData(v)
                if i>=0:c.setCurrentIndex(i)
            else:c.setValue(v)
            c.blockSignals(False)
        if hasattr(self,'preview'):self.refresh_regions()
        if hasattr(self,'curve_widget'):self.refresh_curve()
        if hasattr(self,'controls') and 'window_unit' in self.controls:self.refresh_window_controls()
    def choose_dll(self):
        filename=BACKENDS[self.backend.currentData()][1]
        p,_=QFileDialog.getOpenFileName(self,'选择官方 x64 相机接口','',f'接口文件 ({filename})')
        if p:self.dll.setText(p)
    def backend_changed(self):
        self.sdk_paths[self.previous_backend]=self.dll.text();key=self.backend.currentData();self.previous_backend=key
        self.dll.setText(self.sdk_paths[key]);self.dll.setToolTip(BACKENDS[key][1])
        self.backend_note.setText('请安装完整 MVS 开发包，选择 MvImport 中的 Python 接口文件。USB3/GigE 黑白相机；画幅切换为相机 ROI。' if key=='hik' else '选择对应厂商的官方 x64 动态库。只接受支持的黑白原始格式。')
    def connect_camera(self):
        if self.connected:self.worker.send('disconnect')
        else:self.worker.send('connect',sim=self.source.currentIndex()==1,backend=self.backend.currentData(),dll=self.dll.text(),index=self.index.value())
    def apply_capture(self):
        v=self.values(['exposure','gain','input_bits','resolution','native_bin','software_bin']);v={k:x for k,x in v.items() if x is not None}
        self.worker.send('settings',values=v)
    def apply_processing(self):
        v=self.values(['ae_mode','target','ae_low','ae_high','ae_gain_low','ae_gain_high','seconds','window_unit','window_frames','mode','compute_device','memory','trigger_condition','trigger_threshold','trigger_mode','dark','bias','flat'])
        if v['ae_low']>v['ae_high']:self.show_error('最短曝光不能大于最长曝光');return
        self.worker.send('settings',values=v)
    def apply_display(self):
        v=self.values(['stretch','black','white','gamma','palette','contrast','sharpen','sharpen_radius','curve_mode',
                       'curve_shadows','curve_darks','curve_lights','curve_highlights','curve_points','denoise_mode','denoise_amount',
                       'lowlight_mode','lowlight_strength'])
        if not v['stretch'] and v['white']<=v['black']:self.show_error('亮点必须大于暗点');return
        self.worker.send('settings',values=v)
    def edit_palette(self):
        d=PaletteDialog(self.advanced['custom_points'],self)
        if d.exec()==QDialog.DialogCode.Accepted:
            v={'custom_points':d.points(),'palette':'自定义'};self.set_values(v);self.worker.send('settings',values=v)
    def edit_curve(self):
        d=CurveDialog(self.advanced['curve_points'],self)
        if d.exec()==QDialog.DialogCode.Accepted:
            v={'curve_points':d.points(),'curve_mode':'点曲线'};self.set_values(v);self.worker.send('settings',values=v)
    def refresh_curve(self):
        if not hasattr(self,'curve_widget'):return
        p=tuple(self.controls[k].value() for k in ('curve_shadows','curve_darks','curve_lights','curve_highlights'))
        mode=self.controls['curve_mode'].currentData();self.curve_widget.set_curve(mode,p,self.advanced['curve_points'])
        active=mode=='Camera Raw 参数曲线'
        for k in ('curve_shadows','curve_darks','curve_lights','curve_highlights'):self.controls[k].setEnabled(active)
    def refresh_window_controls(self):
        if 'window_unit' not in self.controls:return
        frame_mode=self.controls['window_unit'].currentData()=='帧数'
        self.controls['window_frames'].setEnabled(frame_mode)
        self.controls['seconds'].setEnabled(not frame_mode)
        active=self.controls['trigger_condition'].currentData()!='关闭'
        for k in ('trigger_threshold','trigger_mode'):self.controls[k].setEnabled(active)
    def capture_master(self,kind):
        if not self.connected:self.show_error('请先连接相机');return
        if self.source.currentIndex()==0:
            msg='请盖好镜头，完全遮光。' if kind!='flat' else '请对准均匀光源，保持镜头、光圈和滤镜不变，并避免过曝。'
            if QMessageBox.question(self,'准备校正帧',msg+'\n准备完成后开始采集？')!=QMessageBox.StandardButton.Yes:return
        self.worker.send('master',master_kind=kind,count=self.master_count.value())
    def library_file(self,save):
        fn=QFileDialog.getSaveFileName if save else QFileDialog.getOpenFileName
        p,_=fn(self,'校正库','','校正库 (*.npz)')
        if p:self.worker.send('library_save' if save else 'library_load',path=p)
    def save_image(self):
        options={'原始 TIFF（16位容器） (*.tif)':'raw','处理后浮点 TIFF (*.tif)':'float','处理后 FITS (*.fits)':'fits','预览 PNG (*.png)':'png'}
        p,f=QFileDialog.getSaveFileName(self,'保存当前完整图像','',';;'.join(options))
        if p:self.worker.send('save',path=p,format=options[f])
    def record(self):
        if self.recording:self.worker.send('record');return
        p,_=QFileDialog.getSaveFileName(self,'保存原始视频（16位容器）','','SER (*.ser)')
        if p:self.worker.send('record',path=p)
    def save_config(self):
        p,_=QFileDialog.getSaveFileName(self,'保存配置','','JSON (*.json)')
        if p:
            Path(p).write_text(json.dumps(dict(version=1,settings=self.values(list(DEFAULTS)+['resolution','native_bin']),sdk=self.dll.text(),backend=self.backend.currentData()),ensure_ascii=False,indent=2),encoding='utf-8')
            self.log.appendPlainText('配置已保存；校正库请另行保存。')
    def load_config(self):
        p,_=QFileDialog.getOpenFileName(self,'加载配置','','JSON (*.json)')
        if not p:return
        try:
            d=json.loads(Path(p).read_text(encoding='utf-8'))
            if d.get('version')!=1:raise ValueError('配置版本不兼容')
            backend=d.get('backend','tucam')
            if backend not in BACKENDS:raise ValueError('配置中的相机接口无效')
            if self.connected and backend!=self.backend.currentData():raise ValueError('请断开相机后加载其他接口的配置')
            if not self.connected:self.backend.setCurrentIndex(self.backend.findData(backend))
            vals={k:v for k,v in d['settings'].items() if k in DEFAULTS or k in ('resolution','native_bin')}
            if vals.get('hist_source')=='原始16位':vals['hist_source']='相机原始灰度'
            if 'ae_mode' not in vals and 'auto' in vals:vals['ae_mode']='优先曝光时长' if vals['auto'] else '手动'
            for k,v in vals.items():
                if k in ('ae_roi','math_roi'):validate_roi(v);continue
                if k=='custom_points':validate_points(v);continue
                if k=='curve_points':validate_curve_points(v);continue
                if k in ('resolution','native_bin'):
                    if v is not None and not isinstance(v,int):raise ValueError('分辨率配置无效')
                    continue
                if isinstance(DEFAULTS[k],bool):
                    if not isinstance(v,bool):raise ValueError('配置开关无效')
                elif isinstance(DEFAULTS[k],(float,int)):
                    if not isinstance(v,(int,float)) or not np.isfinite(v):raise ValueError('配置数值无效')
                elif not isinstance(v,str):raise ValueError('配置文本无效')
            self.set_values(vals);self.dll.setText(d.get('sdk',self.dll.text()))
            restored=self.values(DEFAULTS.keys());restored.update(self.values(['resolution','native_bin']))
            restored={k:v for k,v in restored.items() if k not in ('resolution','native_bin') or v is not None}
            self.worker.send('settings',values=restored)
            self.log.appendPlainText('已加载配置，设备连接和校正库需单独选择。')
        except Exception as e:self.show_error(str(e))
    def begin_crop(self):
        self.preview.selection_kind='crop';self.preview.cropping=True;self.statusBar().showMessage('在图像上按住左键框选直播区域')
    def begin_region(self,kind):
        self.preview.selection_kind=kind;self.preview.cropping=True
        self.statusBar().showMessage('在图像上拖动框选'+('测光区域' if kind=='ae' else '像素运算区域'))
    def set_region(self,kind,roi):
        if roi is not None and self.crop:
            x,y,w,h=self.crop;rx,ry,rw,rh=roi;roi=(x+rx*w,y+ry*h,rw*w,rh*h)
        key=kind+'_roi';roi=validate_roi(roi);self.advanced[key]=roi;self.worker.send('settings',values={key:roi});self.refresh_regions()
    def refresh_regions(self):
        self.preview.image_region=self.crop or (0,0,1,1)
        for key,label in [('ae',self.ae_roi_label),('math',self.math_roi_label)]:
            roi=self.advanced[key+'_roi'];self.preview.regions[key]=(roi,self.controls[key+'_show'].isChecked())
            label.setText(('测光' if key=='ae' else '运算区域')+'：'+('全画面' if roi is None else f'左 {roi[0]*100:.1f}% / 上 {roi[1]*100:.1f}% / 宽 {roi[2]*100:.1f}% / 高 {roi[3]*100:.1f}%'))
        self.preview.draw_regions()
    def set_crop(self,crop):
        if self.crop:
            x,y,w,h=self.crop;cx,cy,cw,ch=crop;crop=(x+cx*w,y+cy*h,cw*w,ch*h)
        self.crop=crop;self.worker.send('settings',values={'preview_crop':crop});self.preview.box.setRect(QRectF());self.preview.fit_image()
        self.refresh_regions()
        self.statusBar().showMessage('已从处理后的完整图像裁切预览；原始数据保持完整')
    def clear_crop(self):
        self.crop=None;self.worker.send('settings',values={'preview_crop':None});self.preview.box.setRect(QRectF());self.preview.fit_image()
        self.refresh_regions()
    def one_to_one(self):self.preview.fit=False;self.preview.resetTransform()
    def show_error(self,text):
        self.log.appendPlainText('⚠ '+text);self.statusBar().showMessage(text);self.output.message='采集提示：'+text;self.output.update()
    def poll(self):
        while True:
            try:e=self.worker.events.get_nowait()
            except queue.Empty:break
            kind=e['kind']
            if kind=='connected':
                self.connected=True;self.connect_btn.setText('断开连接');self.device_label.setText(e['name'])
                compute=e.get('compute',{});device_text=compute.get('label','CPU')
                self.device_label.setText(e['name']+'\n计算：'+device_text)
                self.source.setEnabled(False);self.index.setEnabled(False);self.backend.setEnabled(False);self.dll.setEnabled(False)
                for k,r in [('exposure',e['exp_range']),('gain',e['gain_range'])]:
                    self.controls[k].setRange(r[0],r[1]);self.controls[k].setSingleStep(max(.001,r[2]))
                for k in ('ae_low','ae_high'):self.controls[k].setRange(e['exp_range'][0],e['exp_range'][1])
                for k in ('ae_gain_low','ae_gain_high'):self.controls[k].setRange(e['gain_range'][0],e['gain_range'][1])
                for k,opts in [('resolution',e['resolutions']),('native_bin',e['bins']),('input_bits',e.get('bit_options',[]))]:
                    c=self.controls[k];c.blockSignals(True);c.clear()
                    for v,t in opts:c.addItem(t,v)
                    if not opts:c.addItem('驱动未提供',None)
                    c.setEnabled(bool(opts) and (k!='input_bits' or len(opts)>1))
                    c.blockSignals(False)
                self.set_values(e)
                self.log.appendPlainText('已连接：'+e['name']);self.log.appendPlainText('计算设备：'+device_text);self.output.message=''
            elif kind=='disconnected':
                self.connected=False;self.connect_btn.setText('连接相机');self.live_label.setText('未连接');self.output.message='未连接相机';self.output.update()
                self.source.setEnabled(True);self.index.setEnabled(True);self.backend.setEnabled(True);self.dll.setEnabled(True);self.controls['input_bits'].setEnabled(True);self.output.pix=QPixmap()
            elif kind=='settings':self.set_values(e['values'])
            elif kind=='telemetry':
                self.set_values({k:v for k,v in e['values'].items() if k in self.controls and not self.controls[k].hasFocus()})
            elif kind=='error':self.show_error(e['text'])
            elif kind=='log':self.log.appendPlainText(e['text'])
            elif kind=='paused':
                self.pause_btn.setText('恢复采集' if e['value'] else '暂停采集');self.live_label.setText('采集已暂停' if e['value'] else '采集中')
                self.output.message='采集已暂停' if e['value'] else '';self.output.update()
            elif kind=='record':self.recording=e['active'];self.rec_btn.setText('停止录制' if self.recording else '录制原始 SER')
            elif kind=='master':self.master_active=e['active'];self.master_label.setText(e['text']);self.output.message='正在采集校正帧' if e['active'] else '';self.output.update()
        with self.worker.lock:d=self.worker.latest
        if d and d['serial']!=self.seen:
            self.seen=d['serial'];self.latest=d
            dt=d['stamp']-self.last_stamp;self.last_stamp=d['stamp']
            if 0<dt<20:self.fps=.7*self.fps+.3/dt
            rgb=d['rgb'];h,w=rgb.shape[:2];q=QImage(rgb.data,w,h,rgb.strides[0],QImage.Format.Format_RGB888).copy();pix=QPixmap.fromImage(q)
            self.preview.set_image(pix)
            self.output.pix=pix
            if not self.worker.paused and not self.master_active:self.output.message='模拟画面 · 非相机数据' if d['sim'] else ''
            self.output.update()
            self.hist.counts=d['counts'];self.hist.low=d['hist_low'];self.hist.high=d['hist_high'];self.hist.update();s=d['stats']
            mode_text='单帧' if d.get('effective_mode',d['mode'])=='关闭' else '滚动'+d.get('effective_mode',d['mode'])
            configured_text='单帧' if d['mode']=='关闭' else '滚动'+d['mode']
            self.live_label.setText(('模拟预览' if d['sim'] else '实际采集')+f"   ·   {d['shape'][1]} × {d['shape'][0]}   ·   {self.fps:.1f} 帧/秒   ·   {mode_text}")
            ae=d['ae_mode'];stretch='持续自动' if d['stretch'] else '固定'
            denoise=d.get('denoise_mode','关闭')
            if denoise!='关闭':denoise+=f" {d.get('denoise_amount',0):g}%"
            lowlight=d.get('lowlight_mode','关闭')
            if lowlight!='关闭':lowlight+=f" {d.get('lowlight_strength',0):g}%"
            self.capture_state.setText('测光控制：'+d['ae_status']+'  ·  像素运算：'+d['math_op']+'  ·  亮度触发：'+d.get('trigger_state','未启用')+'  ·  即时降噪：'+denoise+'  ·  弱光增强：'+lowlight+'  ·  '+d.get('compute_device','CPU'))
            trigger_line='' if configured_text==mode_text else f"\n设置模式  {configured_text}"
            overflow='；滚动积分超过原始范围属于浮点累加' if d.get('processed_overflow') and d.get('effective_mode')=='积分' else ''
            self.stats.setText(f"原始帧 Mono{d['bits']} · 有效范围 0—{d['raw_limit']} · 16 位容器\n原始均值  {d['raw_mean']:.1f} · 原始峰值  {d['raw_max']:.1f}\n曝光读回  {d['exposure']:.3f} ms\n增益读回  {d['gain']:g}\n自动曝光  {ae}\n显示拉伸  {stretch}\n\n生效模式  {mode_text}{trigger_line}\n窗口缓存  {d['frames']} 帧（{('时间' if d['window_unit']=='时间' else '帧数')}）\n首末跨度  {d['span']:.2f} 秒\n缓存合计  {d['memory']:.0f} MB\n处理尺寸  {d['processed_shape'][1]}×{d['processed_shape'][0]}\n处理类型  {d['processed_dtype']}{overflow}\n\n处理均值  {s['mean']:.1f}\n超过亮点  {s['white_clip']:.3f}%\n对焦参考  {s['focus']:.1f}\n\n当前暗点  {d['black']:.1f}\n当前亮点  {d['white']:.1f}")
        if self.connected and self.last_stamp and not self.worker.paused and not self.master_active:
            elapsed=time.monotonic()-self.last_stamp
            expected=max(2.,self.controls['exposure'].value()/1000*3+1)
            if elapsed>expected:self.capture_state.setText(f'等待新帧 {elapsed:.1f} 秒；OBS 保留最近完成画面')
    def closeEvent(self,e):
        self.worker.send('quit');self.worker.join(timeout=.1)
        if self.worker.is_alive():
            self.setEnabled(False);self.statusBar().showMessage('正在完成当前曝光并保存录制，请稍候…')
            QTimer.singleShot(300,self.close);e.ignore();return
        self.output.close();e.accept()

def main():
    app=QApplication(sys.argv)
    for name in ('msyh.ttc','msyhbd.ttc'):
        p=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/name
        if p.exists():QFontDatabase.addApplicationFont(str(p))
    app.setFont(QFont('Microsoft YaHei UI',9));app.setStyle('Fusion');app.setStyleSheet(STYLE)
    w=MainWindow();w.show()
    if '--demo' in sys.argv:
        w.source.setCurrentIndex(1);w.connect_camera()
    elif '--camera' in sys.argv:w.connect_camera()
    if '--screenshot' in sys.argv:
        p=sys.argv[sys.argv.index('--screenshot')+1]
        def grab():w.grab().save(p);w.close()
        QTimer.singleShot(35000 if '--camera' in sys.argv else 3500,grab)
    return app.exec()
if __name__=='__main__':sys.exit(main())
