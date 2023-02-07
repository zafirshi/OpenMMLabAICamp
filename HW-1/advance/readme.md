# 进阶项目：使用MMclassification在cifar10数据集上取得超过90%的分类识别精度


## 数据处理
由于计算开发平台无法访问外网下载数据集，需要先将数据集下载好并存放在/data/cifar10/文件目录下，
下载文件是tar.gz格式的打包文件，无需其他操作，库底层会自动解压加载


## 配置文件
直接的方法是使用算法库中自带的配置文件（在/config/resnet中），但这是因为cifar10是基本数据集，
最好的实践方法是加载在Imagenet上训练的预训练参数，自己写配置文件，在cifar10上作微调的迁移学习方法，
这样更符合实际


配置文件的书写方法可以参照MMClassifation的官方文档，这里简单粘贴一下：

```python
_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar10_bs16.py', 
    '../_base_/default_runtime.py'
]
 
model = dict(
    head=dict(
        num_classes=10,
    loss=dict(
        type='LabelSmoothLoss',
        label_smooth_val=0.1,
        num_classes=10,  # 更改输出头
        reduction='mean',
        loss_weight=1.0),
    )
)

img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    to_rgb=False,
)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Resize', size=224),  # 将cifar中的数据形式直接resize到imageNet输入大小
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
]
test_pipeline = [
    dict(type='Resize', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]
data = dict(
    samples_per_gpu=32, # 修改单张显卡bs
    train=dict(
        type='CIFAR10',
        data_prefix='data/cifar10', 
        pipeline=train_pipeline),
    val=dict(
        type='CIFAR10',
        data_prefix='data/cifar10',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type='CIFAR10',
        data_prefix='data/cifar10',
        pipeline=test_pipeline,
        test_mode=True))

optimizer = dict(type='SGD', lr=0.01/4, momentum=0.9, weight_decay=0.0001)  # 微调学习率
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='step', step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=200)
log_config = dict(interval=100)

# 加载预先训练好的参数
load_from = '/HOME/scz4242/run/mmclassification/mmclassification-master/checkpoints/resnet50_8xb32_in1k_20210831-ea4938fc.pth'
```

## 运行结果

在重写配置文件和微调学习率之后，最优的在测试集上的结果是：【resnet50_1xb32_cifar10】
```python
{"mode": "val", "epoch": 20, "iter": 313, "lr": 0.00025, "accuracy_top-1": 97.23, "accuracy_top-5": 99.92}
```

如果直接使用config文件夹下的配置文件，会收敛比较慢，最优结果是：【resnet50_8xb16_cifar10】
```python
{"mode": "val", "epoch": 189, "iter": 625, "lr": 0.001, "accuracy_top-1": 95.11, "accuracy_top-5": 99.84}
```

上面比较之后可以发现：自己微调之后的model精度是比直接使用原始配置文件的效果好的！：）

