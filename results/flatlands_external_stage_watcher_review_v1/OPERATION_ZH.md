# 阶段后处理运行说明

源码与 17 项 CPU 模拟测试已完成；本回执生成时未启动实际 watcher、审计、渲染或 GPU 任务。由 root 审查并提交源码后启动一次即可：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /home/hairo/miniconda3/bin/python3.13 scripts/watch_flatlands_formal_stages.py --watch
```

每 30 秒检查已完成阶段。复用协议、数据、指标和完成回执一致的已有成功审计；没有新输入时复用完整且文件哈希一致的报告。首次运行如旧报告缺少新阶段或独立审计标注，会生成一个新报告快照。队列释放锁后再作最终补扫并退出。不要同时运行 --once 与 --watch。

运行状态在正式结果目录的 postprocessing/status.json、state.json 和 active.json；独立进程快照、源文件、协议、日志及结束回执在 postprocessing/sessions/<id>/。使用独立 .watcher.lock；读取共享 STOP，但不创建、删除 STOP，不向训练队列发信号。

失败或中断的审计、报告保存原始尝试并标为 blocked_manual_review，不自动重试。人工解决后，在新的审计 attempt 中提供匹配的成功回执，或完成匹配的报告快照，后续扫描可以复用并解除阻塞。源码或协议运行时变化会停止本 watcher 自己的 CPU 子进程，等待人工审查；不会继续混用源码。

此脚本不运行训练、推理、完整训练调度，不批准视觉门槛，也不发布网站。阶段报告仍是 validation-only 工程诊断。
