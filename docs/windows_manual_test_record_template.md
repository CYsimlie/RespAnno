# Windows Manual Test Record

> 填表人：__________  
> 测试日期：__________  
> Windows 版本：__________  
> Python 版本：__________  
> 当前 git commit：__________  

---

## 1. 基础环境

- [ ] conda env create -f environment.yml 成功
- [ ] python -c "import PyQt5, pyqtgraph, librosa, sklearn, lightgbm, sounddevice; print('OK')" 输出 OK
- [ ] 系统有声卡 / 虚拟音频设备

## 2. 启动

- [ ] `python 1.0.0.py` 启动无报错
- [ ] 主窗口标题包含 "Time-Frequency Analysis and Annotation"
- [ ] 菜单栏存在 (File / Settings / Help)
- [ ] ML 工具栏可见

## 3. WAV 加载

- [ ] 点击 "Import WAV File" 弹出文件选择框
- [ ] 选择一个 .wav 文件后加载成功
- [ ] 标题栏显示文件名
- [ ] 状态栏显示加载过程 (Loading audio... / Drawing waveform... / Computing STFT...)

## 4. 波形显示

- [ ] 波形区域有白色曲线
- [ ] 纵轴范围与音频幅度匹配
- [ ] 横轴为时间 (s)

## 5. STFT 显示

- [ ] STFT 区域有彩色频谱图
- [ ] 鼠标在 STFT 上移动时标题显示 t 和 f 坐标
- [ ] Combo 切换 Heatmap / Grayscale 正常

## 6. 进度条与播放

- [ ] 进度条可以拖动
- [ ] 时间标签跟随更新
- [ ] 红色竖线在波形和 STFT 图上同步移动

## 7. 手动标注

- [ ] 在第三栏拖选后弹出 "Add Annotation" 对话框
- [ ] 选择预设类型后文本框自动填入英文名
- [ ] 手动输入自定义标签创建成功
- [ ] 标注条颜色与预设颜色一致
- [ ] STFT 对应区域出现同色高亮
- [ ] 多条标注自动分轨（最多 3 行）

## 8. 标注编辑

- [ ] 双击标注条进入编辑模式（虚线）
- [ ] 拖动左右把手改变边界
- [ ] 拖动整条平移
- [ ] Enter 提交修改
- [ ] Esc 取消修改恢复原位

## 9. 标注删除与撤销

- [ ] 右键 → Delete 删除标注
- [ ] Ctrl+Z 恢复已删除标注
- [ ] 编辑后 Ctrl+Z 恢复原区间

## 10. 标签导入

- [ ] Import Annotations 弹出文件选择
- [ ] 选择 CSV 导入成功
- [ ] 导入的标注在界面显示
- [ ] 导入的 source 正确 (manual / ml 等)

## 11. 标签导出

- [ ] Export Annotations 弹出保存对话框
- [ ] 默认文件名为 `<wav>_events.csv`
- [ ] 保存后在文件系统可以找到
- [ ] CSV 格式：start, end, label, source
- [ ] 用记事本打开内容正确

## 12. Settings 对话框

- [ ] STFT 标签页：n_fft / hop_length / f_max 输入框 + 直方图 + colorbar
- [ ] Display 标签页：缩放滑条 + Y轴上下限
- [ ] Preprocessing 标签页：重采样开关 + 滤波设置
- [ ] Auto Label Import 标签页：格式/后缀/分隔符/列号
- [ ] Short-Time Features 标签页：56 个特征可选（最多 5 个）
- [ ] OK 后参数保存生效
- [ ] Cancel 后参数不变
- [ ] Reset Defaults 不崩溃

## 13. Preprocessing 设置

- [ ] 重采样开关正常
- [ ] 滤波类型切换正常 (bandpass/lowpass/highpass/bandstop)
- [ ] 截止频率输入框范围正确
- [ ] Butterworth order 输入框范围正确 (1–12)
- [ ] Zero-phase 复选框可用

## 14. 上一首 / 下一首

- [ ] Previous 按钮可用
- [ ] Next 按钮可用
- [ ] Up/Down 快捷键可用
- [ ] 切换后标题栏更新
- [ ] 切换后波形和 STFT 正确
- [ ] 第一首不能继续 Previous
- [ ] 最后一首不能继续 Next

## 15. FFT 页面

- [ ] Switch Spectrum → FFT 页面正常
- [ ] 显示青色频谱曲线
- [ ] 横轴 Frequency (Hz)，纵轴 Amplitude
- [ ] 鼠标可缩放/平移

## 16. Short-Time Features 页面

- [ ] Switch Spectrum → Short-Time Features 页面正常
- [ ] 显示所选特征曲线
- [ ] 图例显示特征名
- [ ] 最多 5 条曲线
- [ ] 特征颜色不重复

## 17. ML 工具栏

- [ ] ML Label 下拉框包含预设标签
- [ ] Train Model 按钮存在
- [ ] Auto-label Unreviewed 按钮存在
- [ ] Annotation Legend 按钮弹出图例对话框
- [ ] 无标注时训练不崩溃（弹出提示信息）
- [ ] 训练后 Auto-label 可执行

## 18. 自动导入标签

- [ ] Settings → Auto Label Import → Enable 勾选
- [ ] 加载 WAV 后自动找到同名 _events 文件
- [ ] 自动导入的标注在界面显示
- [ ] 状态栏显示 "Auto-imported events annotations: xxx (N items)"

## 19. 帮助

- [ ] F1 弹出 About 对话框
- [ ] Help → About 显示作者信息

## 20. 退出

- [ ] 关闭窗口无崩溃
- [ ] Ctrl+Q 退出无崩溃
- [ ] 再次启动正常

---

## 异常记录

| # | 操作步骤 | 预期结果 | 实际结果 | 截图 |
|---|---------|---------|---------|------|
| 1 |         |         |         |      |
| 2 |         |         |         |      |
| 3 |         |         |         |      |

---

## 结论

- [ ] 全部通过 — 可以进入下一阶段
- [ ] 部分失败（见异常记录）— 需要修复
- [ ] 严重崩溃 — 停止推进，修复后再测
