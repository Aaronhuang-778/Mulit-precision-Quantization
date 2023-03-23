
### 设计思路
创建一个字典graph_dict
创建一个计算图graph

遍历原始计算图的每一行
对于原始计算图中的信息
- graph标题: 以{id: None}形式存入graph_dict
- 输入张量: 以{id: shape}形式存入graph_dict
- Constant: 以{id: value}形式存入graph_dict。后面其他节点调用的时候可以直接根据id取出值
- 算子声明: 以{id: father_name.this_name}形式存入graph_dict。这样可以与named_parameters中的名字相匹配
- Sequential声明: 以{id: father_name.this_name}形式存入graph_dict。这样可以与named_parameters中的名字相匹配
- ListConstruct: 以{id: value}形式存入graph_dict。后面其他节点调用的时候可以直接根据id取出值
- 权重Tensor: 以{id: father_name.this_name}形式存入graph_dict。注:
    - 对于weight和bias。可以根据名字从named_parameters中找到对应的Tensor。进而能够获得其他所有参数
    - 对于running_var和running_mean。由于named_parameters中没有相应的Tensor，所以无法获得相应Tensor。但生成graph.txt时也用不到
- 算子计算: 以{id: shape}形式存入graph_dict。同时声明一个Op对象，存入graph。注:
    - 这样子在计算过程中可以直接推算shape
- return: 以{id: shape}形式存入graph_dict。同时存入graph


### 使用方法
python3 get_model.py \
    -m model_path \
    -s input_shape_like_1,3,224,224 \
    -o output_dir

### 存储设定
将graph.txt存在output_dir里
将权重bin文件也存在output_dir里
在graph.txt中各个算子里面记录权重文件路径。此路径为权重文件相对于graph.txt的相对路径