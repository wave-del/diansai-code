# Q6A QNN HTP 桥接

`q6a_bridge.c` 通过 QAIRT 2.37.1 官方接口加载 HTP Context，一次创建后重复
`graphExecute`，没有逐帧子进程、临时输入输出文件或 CPU 后端。
读取 System 元数据校验单图、输入输出名称、形状、UINT8 类型和量化编码，
原生层完成四输出反量化；Python 层在现有视觉接口内完成预处理和后处理。

部署库与 Context 位于项目 `model-lib/q6a/`。构造推理器时校验全部必需文件
SHA256；单纯导入模块或启动默认离线界面不会加载原生库。推理器要求在
创建线程推理和关闭，可使用 `with` 管理生命周期。返回数组独立于下次推理缓冲区。

仅重新编译桥接库时需要 SDK 头文件：

```bash
bash vision/native/qnn_htp/build.sh /path/to/qairt/2.37.1.250807 /tmp/libq6a_qnn.so
```

编译仅链接系统 `dl` 和 `m`；运行不需要 SDK、编译器或 NPU 测试目录。
替换部署库后必须重新验证并更新资源清单，不能直接绕过哈希检查。
原 `native/yolo11_seg_npu/` 及 `model-lib/` 根目录的 A733 文件保留作历史资料，
Q6A 不加载这些库。

当前状态为本地图片验证；相机、连续帧稳定和运动执行未验收。
