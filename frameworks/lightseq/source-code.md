其中 pytorch 层面的 encoder layer，会申请总体大小的 prameter，然后会绑定给 layer，跟他共享内存。这样是一个大片的内存里，包含了很多个小 layer 的 parameter
