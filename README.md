# chinese_clip


- 1.本项目代码基于open_clip project建设，并针对中文领域数据以及在中文数据上实现更好的效果做了优化。

- 2.chinese-clip预训练模型下载， 关注微信公众号paperClub， 回复关键词“中文clip预训练模型”即可获得下载链接。


- 3.开始本项目前，需先检查是否满足下列环境配置要求:

    python >= 3.6.4  
    pytorch >= 1.7.1 (with torchvision)  
    CUDA Version >= 10.1  
    运行下列命令即可安装本项目所需的三方库。


- 4.训练：配置参数在utils/params.py当中，可以根据情况修改
    python ./clip_finetune.py

- 5.测试结果:最近太忙，等有空了梳理一遍。

![](./out/loss.png)
![](./out/search1.png)
![](./out/search2.png)
![](./out/search3.png)
![](./out/search4.png)


> 预训练模型太大，下载请微信公众号paperClub回复关键词【中文clip预训练模型】， 或者联系我即可（paperclub@163.com）。
