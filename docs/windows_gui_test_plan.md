# Windows GUI Manual Test Plan

本计划用于在 Windows 上手动验证 RespAnno 的完整 GUI 功能。
Linux 上因缺少 PortAudio / sounddevice / xcb 等桌面组件，无法运行完整 GUI。

---

## 1. Windows 环境准备

- Windows 10 或 11 (64-bit)
- 16 GB RAM 推荐
- 支持音频播放的声卡（或虚拟音频设备）

---

## 2. conda 环境创建

```powershell
# 在项目根目录下
conda env create -f environment.yml
conda activate respanno
```

如果 environment.yml 中 sounddevice 安装失败，可改用 pip：

```powershell
conda activate respanno
pip install sounddevice
```

验证导入：

```powershell
python -c "import PyQt5; import pyqtgraph; import librosa; import sklearn; import lightgbm; import sounddevice; print('OK')"
```

---

## 3. 启动命令

```powershell
# 方式一：直接启动 legacy 程序
python 1.6.6.py

# 方式二：通过包入口启动
python -m respanno.main
```

预期：弹出主窗口，标题包含 "Time-Frequency Analysis and Annotation"。

---

## 4. 导入 WAV 测试

| 步骤 | 操作 | 预期结果 |
|------|------|---------|
| 4.1 | 点击 "Import WAV File" 按钮 | 弹出文件选择对话框 |
| 4.2 | 选择一个 .wav 文件（任意采样率、时长） | 状态栏显示 "Loading audio..." → "Drawing waveform..." → "Computing STFT display..." |
| 4.3 | 等待加载完成 | 标题栏显示文件名；波形图有曲线；STFT 图有彩色频谱；进度条在 0 |
| 4.4 | 加载 44100 Hz 长文件 | 默认重采样到 4000 Hz，标题栏显示 `(resample=4000 Hz)` |
| 4.5 | 点击 Play | 听到声音播放；进度条移动；时间标签更新 |
| 4.6 | 点击 Pause | 停止播放；按钮变回 "Play" |
| 4.7 | 拖动进度条后松手 | Seek 到新位置继续播放 |

---

## 5. 波形/STFT 显示测试

| 步骤 | 操作 | 预期结果 |
|------|------|---------|
| 5.1 | 加载 WAV 后查看波形区域 | 白色曲线，信号幅度正确 |
| 5.2 | 查看 STFT 区域 | 彩色频谱图（默认 Heatmap 配色），纵轴为频率 (0～f_max Hz)，横轴为时间 |
| 5.3 | 鼠标在 STFT 图上移动 | 标题显示当前鼠标位置的 t 和 f 坐标 |
| 5.4 | 使用 combo 切换到 Grayscale | STFT 变为黑白配色 |
| 5.5 | 切回 Heatmap | 恢复彩色 |

---

## 6. Settings / Preprocessing 测试

| 步骤 | 操作 | 预期结果 |
|------|------|---------|
| 6.1 | 点击 "Settings" 按钮 | 弹出多标签设置对话框 |
| 6.2 | 查看 STFT 标签页 | 显示 n_fft、hop_length、f_max 输入框 + 直方图 + colorbar |
| 6.3 | 查看 Display 标签页 | 显示缩放滑条 + Y轴上下限 |
| 6.4 | 查看 Preprocessing 标签页 | 显示重采样开关 (4000 Hz)、滤波开关、滤波类型/截止频率/阶数 |
| 6.5 | 查看 Auto Label Import 标签页 | 显示格式、后缀、分隔符、列号映射设置 |
| 6.6 | 修改 n_fft=1024, hop_length=512，OK | STFT 图刷新 |
| 6.7 | 修改 f_max=1000，OK | STFT 频率范围缩小 |
| 6.8 | 切换滤波类型 (bandpass/lowpass/highpass/bandstop) | 无崩溃；下次加载音频时生效 |
| 6.9 | 勾选/取消 Enable resampling | 无崩溃；下次加载音频时生效 |
| 6.10 | 点击 Reset Defaults | 色阶恢复 1%–99% 分位 |
| 6.11 | Cancel 退出 | 参数不变 |
| 6.12 | 再次打开 + OK | 参数已保存 |

---

## 7. 手动标注测试

| 步骤 | 操作 | 预期结果 |
|------|------|---------|
| 7.1 | 在标注区（第三栏）按住左键拖选一段 | 出现半透明拖选区域 |
| 7.2 | 松手 | 弹出 "Add Annotation" 对话框 |
| 7.3 | 选择预设类型 "哮鸣音 (wheeze)" | 文本框自动填入 "wheeze" |
| 7.4 | 点 OK | 标注条出现（彩色填充块）；STFT 对应区域出现同色高亮 |
| 7.5 | 不选预设，手动输入 "custom_type"，OK | 标注条出现 |
| 7.6 | 在标注条上右键 → "Play" | 循环播放该段（如声卡可用） |
| 7.7 | 在标注条上右键 → "Delete" | 标注条和 STFT 高亮移除 |
| 7.8 | 标注条上双击 | 进入编辑模式：边框变虚线，左右把手出现 |
| 7.9 | 编辑模式下拖动左右把手 | 标注条边界改变；状态栏实时显示起止时间 |
| 7.10 | 编辑模式下拖动整条 | 标注条平移 |
| 7.11 | 编辑模式下按 Enter | 提交修改；退出编辑模式 |
| 7.12 | 编辑模式下按 Esc | 取消修改；恢复原来位置 |
| 7.13 | Ctrl+Z | 撤销上一次删除/编辑操作 |

---

## 8. 标签导入导出测试

| 步骤 | 操作 | 预期结果 |
|------|------|---------|
| 8.1 | 手动标注几条，点 "Export Annotations" | 弹出保存对话框，默认文件名 `<wav_name>_events.csv` |
| 8.2 | 保存到某目录，用记事本打开 | CSV 格式：`start,end,label,source`，数据正确 |
| 8.3 | 点 "Import Annotations"，选择刚才导出的 CSV | 导入成功，标注条重新画出 |
| 8.4 | 用 TXT (Tab 分隔) 测试导入 | 正确导入 |
| 8.5 | 在 Settings→Auto Label Import 中勾选 Enable | 下次加载 WAV 时自动导入同名 `_events.csv` |

---

## 9. 上一首/下一首测试

| 步骤 | 操作 | 预期结果 |
|------|------|---------|
| 9.1 | 在包含多个 WAV 的目录加载一个文件 | 建立同目录 WAV 列表 |
| 9.2 | 点 "Next" 或按 Down 键 | 切换到下一个 WAV；标题栏更新；波形/STFT 刷新 |
| 9.3 | 点 "Previous" 或按 Up 键 | 切换到上一个 WAV |
| 9.4 | 在第一个文件点 Previous | 弹出 "This is already the first file" |
| 9.5 | 在最后一个文件点 Next | 弹出 "This is already the last file" |

---

## 10. FFT / Short-Time Features 测试

| 步骤 | 操作 | 预期结果 |
|------|------|---------|
| 10.1 | 加载 WAV 后点 "Switch Spectrum" | 第 1 次：STFT → FFT 图，显示青色频谱曲线 |
| 10.2 | 再次点 "Switch Spectrum" | FFT → Short-Time Features 图，显示所选特征曲线（有图例） |
| 10.3 | 再次点 "Switch Spectrum" | Features → STFT，回到初始视图 |
| 10.4 | Settings→Short-Time Features 标签页中勾选不同特征 | 特征页显示对应曲线（最多 5 条），颜色不同 |
| 10.5 | 勾选超过 5 个特征 | 自动限制为 5 个 |

---

## 11. ML / HSMM 按钮存在性测试

> 注：此项只验证按钮存在且可点击，不验证 ML 训练结果正确性。

| 步骤 | 操作 | 预期结果 |
|------|------|---------|
| 11.1 | 查看工具栏 | 存在 "ML Label:" 下拉框（含 wheeze/Crackles 等标签） |
| 11.2 | 存在 "Train Model" 按钮 | 可点击 |
| 11.3 | 存在 "Auto-label Unreviewed" 按钮 | 可点击 |
| 11.4 | 存在 "Annotation Legend" 按钮 | 点击弹出颜色图例对话框 |
| 11.5 | 无手动标注时点 Train Model | 弹出提示 "No manual annotations yet" 或类似信息（不崩溃） |
| 11.6 | 标注几条后点 Train Model | 弹出训练结果对话框（不崩溃） |
| 11.7 | 训练后点 Auto-label Unreviewed | 在未审阅区域生成机器标注（虚线框） |
| 11.8 | 右键机器标注 → "Accept" | 标注变为实线（已认可状态） |

---

## 12. 退出软件测试

| 步骤 | 操作 | 预期结果 |
|------|------|---------|
| 12.1 | 关闭主窗口 (X) | 程序正常退出，无崩溃 |
| 12.2 | Ctrl+Q | 程序正常退出 |
| 12.3 | 再次启动 → 加载文件 → 关闭 | 无残留进程 |

---

## 13. 常见问题记录

| 问题 | 可能原因 | 排查方法 |
|------|---------|---------|
| sounddevice 导入失败 | PortAudio DLL 缺失 | `pip install sounddevice` 或安装 PortAudio |
| lightgbm 导入失败 | lib_lightgbm.dll 缺失 | `pip install lightgbm` 或安装 VC++ Redistributable |
| STFT 图为空白 | f_max 设置错误 | 检查 Settings→STFT→f_max |
| 标注条不显示 | Y 轴范围不对 | 尝试添加一条标注看是否刷新 |
| 播放无声 | Windows 音频设备问题 | 检查系统音量、默认播放设备 |
| 导出 CSV 乱码 | 编码问题 | 用 UTF-8 编码打开 CSV |
