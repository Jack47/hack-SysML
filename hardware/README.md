## nvidia-smi 
1. 也提供 PCI Express Bandwidth Utilization in KiB/s ? 这个是 GPU 对PCIe 的利用率，还是 PCIe 的利用率？
2. GPU Utilization

## PCIe
[pcicrawler](https://github.com/facebook/pcicrawler): a CLI tool to display/filter/export information about PCI or PCI Express devices and their topology.
问题是需要 root 权限，要读 sysfs 里的东西

## NVLink
dcgm-exporter: requires Golang >= 1.14, DCGM installed.


### [prometheus nvlink exporter](https://github.com/Beuth-Erdelt/prometheus_nvlink_exporter)
上述连接里有详细的结果图

`nvidia-smi nvlink -sc 0bz` , `nvidia-smi nvlink -sc 1pz`

NVLink informations:

```
nvidia-smi nvlink -g 0
```

PCI Informations

```
nvidia-smi dmon -s t -c 1

```


