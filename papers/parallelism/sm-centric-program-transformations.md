1. 在解决的是什么问题？GPU上调度是硬件控制，不对程序员开放的问题
2. 为何成功，标志/准是什么？
3. 在前人基础上的关键创新是什么？程序层面能控制调度，可以提高数据局部性。
4. 关键结果有哪些？
5. 有哪些局限性？如何优化？
6. 这个工作可能有什么深远的影响？


## 1  思路
1. SM-based task selection. 传统的是根据 thread id 来让thread 干活，而本方法里是让 thread 根据它所在的 SM id 来决定干什么活
2. filling-retreating scheme，可以灵活控制一个SM上活跃的threads 数量
