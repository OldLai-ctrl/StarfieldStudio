# 相机接口接入

软件面向部分工业黑白相机。先安装厂商驱动和开发包，再从左侧选择接口。

## TUCam

选择官方 x64 `TUCam.dll`，必须与设备及驱动匹配，并保留配套文件。只接受接口明确报告为单通道且提供 RAW16 的设备。曝光、增益、分辨率与可用相机 Binning 由接口读取。

官方入口：[Tucsen 下载中心](https://www.tucsen.com/download/)。

## 海康 MVS

从[海康机器视觉下载中心](https://www.hikrobotics.com/en/machinevision/service/download/)安装完整 MVS 开发包，包括 x64 运行库、驱动及 Python 示例。仅安装运行库可能没有 Python 接口文件。

在程序内选择 `Development/Samples/Python/MvImport/MvCameraControl_class.py`，保留同目录其他接口文件。程序会尝试常见安装路径，也支持手动指定。加载官方 Python 接口不要求另装 Python。

支持部分 USB3/GigE 黑白设备，按设备序号连接。GigE 需预先在 MVS 中完成网段设置。本版使用 ExposureTime、Gain、PixelFormat、Width、Height 等标准节点；设备必须允许关闭硬件自动曝光、自动增益和触发。

支持非压缩 Mono8/10/12/14/16。仅提供 Packed、压缩或彩色格式的设备会被拒绝。画幅选项为相机 ROI 裁切，原始保存也使用此裁切范围；需要缩小完整画面时用软件 Binning。更换 MVS 开发包版本后需重启程序。

## 图谱 ToupCam

从[图谱官方 SDK 下载中心](https://www.touptekphotonics.com/download/?category=SDK)下载 ToupCamSDK，选择其 Windows x64 `toupcam.dll`。程序使用 `Toupcam_WaitImageV3` 等原生函数，过旧或函数名称不同的贴牌库不保证兼容。

确认黑白传感器后启用 RAW，优先使用设备高位深。支持接口报告的 8/10/11/12/14/16 位，分辨率由设备提供。增益为接口单位，不能与海康 dB 或其他厂商增益数值直接比较。

## 共同行为

- 连接后读取真实范围，软件自动调节按设备能力工作。硬件自动曝光/增益被关闭并检查。
- 参数变更后检查读回；错误或格式不一致时暂停，不显示假成功。
- 增益或画幅变化清空历史帧；启动及参数切换可能丢弃短暂过渡帧。
- 原始灰度值不拉伸。较低位深仍以 16 位容器存储，在 TIFF 元数据或 SER 配套 JSON 中保留位深。
- 三类接口共用校正、有限窗口叠加、测光框、伪彩、局部运算及 OBS。
- 不同相机、增益、位深、画幅的校正帧不能混用。

当前接口契约测试使用模拟设备，不代表三品牌实机认证。
