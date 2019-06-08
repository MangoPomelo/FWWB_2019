1. 先启动Preprocessor.py生成字典和矢量
2. 参数在Configuration.py里面改，模型可以间断训练，会自动保存
3. 之后启动train.py训练，继续训练的时候也是运行这个，不用再运行Preprocessor了否则模型数据和文字不对应要删除重新来
4. loss越小模型越好
5. 当前文件夹还应包含子文件夹"checkpoints","raw_data","npz_data","dicts","result"
