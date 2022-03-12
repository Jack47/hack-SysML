conan 里可以有不同的配置：比如架构、编译器版本等。

## Decentralized package manager
Conan 是 C/S 架构。客户端拉取 package，同时上传 packages到不同 server，类似 git 一样。

server 只是存储包，并不会 build 或者 创建包。包是 client 上传的。

## Binary management
它可以管理预编译好的binary，有很多平台和配置。这样避免从源码重新编译，节省宝贵时间，同时提高可复现性和跟踪性。

![](https://docs.conan.io/en/latest/_images/conan-binary_mgmt.png)
主要是如下的概念：

package: 由 conanfile.py 来定义的。它定义了package的依赖，源码，如何从源码编译二进制等。一个 “conanfile.py” recipe 
可以产生任意数量的二进制，给不同数量平台和配置一人一个，包括：操作系统，架构，编译器，构建类型等。

从 server 上安装包时，只会拉取当前平台和配置，而不是所有的。

## All platforms，all build systems and compilers
Windows, Linux(Ubuntu, Debian, RedHat,ArchLinux, Raspbian), OSX，FreeBSD 等。它可以在任何支持 Python 的平台上运行。

它支持已有的平台：裸机到桌面，移动设备，嵌入式，服务器，跨平台。


