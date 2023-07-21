## Server 模型

Gunicorn 是基于 pre-fork 的进程模型。意味着有一个集中的 master 进程，管理了多个 worker 进程。master 不需要知道 clients。而所有请求和响应都是**完全**被 worker 处理的

这种 Pre-Fork 模式和线程模型有很大区别：线程模式下，master 需要创建轻量级的线程来分发请求。而当其中某个线程有问题，会导致master进程有问题

Master: 感觉只是用来管理这些 worker 的，有点像 Daemon 进程一样

## Sync Worker
默认的 worker 类型，每次只会处理一个请求。所以任意的错误，只会影响到最多一个请求。

sync worker 不支持持久化 connections - 每个 connection 都会在响应被发送后关闭掉（即使在程序 header 里设置了 'Keep-Alive' 或者 'Connection: keep-alive'

## Async Workers

一些需要 asynchronous workers 的场景：

* Applications 使用了长的阻塞的调用 (le, external web services)
* 直接面向互联网暴露服务
* Long polling
* Web sockets
* Comet
