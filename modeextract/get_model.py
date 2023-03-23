
# -*- coding: utf-8 -*
import argparse
import torch
import op
import time
from util import *



def get_graph_id(line):
    '''
    从整个计算图的描述行中获取编号
    '''
    i = 0
    while(line[i] != '%'):
        i += 1
    i += 1
    id = ""
    while(line[i] != ':'):
        id += line[i]
        i += 1
    return id


def split_line(line):
    '''
    切分计算图普通行。开头百分号不要。第一个冒号前面一部分。第一个等号前面一部分
    '''
    i = 0
    while(line[i] != "%"):
        i += 1
    i += 1
    id = ""
    while(line[i] != ":"):
        id += line[i]
        i += 1
    i += 1
    line_type = ""
    while(line[i] != "="):
        line_type += line[i]
        i += 1
    i += 1
    line_value = line[i : len(line)]
    return id, line_type, line_value


def broadcast_shape(shape1, shape2):
    if(len(shape1) < len(shape2)):
        return shape2
    elif(len(shape1) > len(shape2)):
        return shape1
    else:
        for i in range(len(shape1)):
            if(shape1[i] > shape2[i]):
                # 如果shape1更大，则广播到shape1
                return shape1
        # 如果shape1不是更大，则广播到shape2
        return shape2


def is_in_graph(graph, id):
    # 检查该id是否存在于graph中
    for node in graph:
        if(node.id == id):
            return True
    return False


def read_model(model_path, input_shape, output_dir):
    # 加载jit模型

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = torch.jit.load(model_path, device)

    # 判断模型类型
    if(type(model) != torch.jit._script.RecursiveScriptModule):
        xxlog("Model must be torch Recursive Script Module", XXError())
        raise("Model must be torch Recursive Script Module")

    # 提取权重数据
    parameter_dict = {}
    for name,parameter in model.named_parameters():
        parameter_dict[name] = parameter.detach().numpy()
    state_dict = model.state_dict()
    xxlog("Extract parameters finished")

    # 获取模型计算图字符串
    # model_graph = model.inlined_graph.__str__()
    model_graph = model.inlined_graph
    graph_head = model_graph.__str__()[:model_graph.__str__().index(')')+2]
    node_list = graph_head.split("\n") + [node.__str__().replace("\n", "") for node in model_graph.nodes()]
    for i in node_list:
        print(i)
    # with open("graph.txt", "w") as f:
    #     f.write(model_graph)
    #     exit()
    print("=========================================")
    xxlog("Extract calculation graph finished")

    # 按行切分字符串
    # model_graph_lines = model_graph.split("\n")

    # 创建计算图存储字典
    graph_dict = {}

    # 创建生成计算图存储列表
    graph = []

    # 返回编号
    ret_count = 0

    # 预留编号
    reserve_id = 2147483647
    
    # for line in model_graph_lines:
    for line in node_list:
        line = line.split("#")[0]       # 去除注释
        line = line.replace(" ", "")    # 去除空格
        if(line == ""):                 # 跳过空行
            continue
        if(line[0:5] == "graph"):                                       # graph
            # 对于整个graph，提取id，以{id:None}加入graph_dict
            id = get_graph_id(line)
            graph_dict[id] = None
            xxlog("Graph \"%s\" detected"%(id))
        elif(not "=" in line):
            # 输入Tensor 或 return
            if(line[0:6] == "return"):                                  # return
                # 对于return，为其分配id。提取输入节点。以{id:input_node}存入graph_dict
                id = "return" + str(ret_count)
                ret_count += 1
                i = 0
                while(line[i] != '%'):
                    i += 1
                i += 1
                input_id = ""
                while(line[i] != ")"):
                    input_id += line[i]
                    i += 1
                input_shape = graph_dict[input_id]
                output_shape = input_shape
                graph_dict[id] = output_shape
                graph.append(op.Output(id, input_id))
                xxlog("return \"%s\" detected"%(id))
            else:                                                       # input
                # 对于input，提取id。以{id:input_shape}存入graph_dict
                id = ""
                i = 0
                while(line[i] != "%"):
                    i += 1
                i += 1
                while(line[i] != ":"):
                    id += line[i]
                    i += 1
                shape = [int(i) for i in input_shape.split(",")]
                graph_dict[id] = shape
                # 创建对象，加入graph
                graph.append(op.Input(
                    id, shape
                ))
                xxlog("input \"%s\" detected"%(id))
        else:
            # 其他内容
            id, line_type, line_value = split_line(line)
            if("Constant" in line_value):                               # Constant
                # 对于Constant，提取值。以{id:value}或{name:value}存入graph_dict
                try:
                    value_index = line.index("value")
                except:
                    value_index = None
                try:
                    name_index = line.index("name")
                except:
                    name_index = None
                if(value_index):
                    i = value_index + 6
                    value_string = ""
                    while(line[i] != "]"):
                        value_string += line[i]
                        i += 1
                if(name_index):
                    i = name_index + 5
                    name_string = ""
                    while(line[i] != "]"):
                        name_string += line[i]
                        i += 1
                if(line_type == "int"):
                    graph_dict[id] = int(value_string)
                elif(line_type == "float"):
                    graph_dict[id] = float(value_string)
                elif(line_type == "None"):
                    graph_dict[id] = None
                elif(line_type == "bool"):
                    graph_dict[id] = bool(value_string)
                elif(line_type == "Function"): # Function类型，目前只记录Function名字
                    # Nothing to do now
                    graph_dict[id] = name_string
                else:
                    xxlog("Unknown type in Constant: %s"%(line_type), XXError())
                    raise("Unknown type in Constant")
                if value_index:
                    xxlog("Constant \"%s\": type \"%s\", value: %s detected"%(id, line_type, graph_dict[id]))
                elif name_index:
                    xxlog("Constant \"%s\": type \"%s\", name: %s detected"%(id, line_type, graph_dict[id]))
            elif("ListConstruct" in line_value):                        # ListConstruct
                # 对于ListConstruct，提取父节点id，然后从id读出值。以{id:[value, value]}存入graph_dict
                line_value = line_value.replace("%", "")
                i = line_value.index("(") + 1
                id_string = ""
                while(line_value[i] != ")"):
                    id_string += line_value[i]
                    i += 1
                if type(graph_dict[id_string.split(",")[0]]) == int or type(id_string.split(",")[0]) == float:
                    value = [graph_dict[i] for i in id_string.split(",")]
                else:
                    value = [i for i in id_string.split(",")]
                graph_dict[id] = value
                xxlog("ListConstruct \"%s\": %s detected"%(id, value))
            elif("__torch__" in line_type):                 # op or sequential
                # 对于op或sequential。提取父节点id。提取本节点名称。以{id:父节点名称.名称}存入
                father_id = ""
                i = line_value.index("(") + 2
                while(line_value[i] != ")"):
                    father_id += line_value[i]
                    i += 1
                name = ""
                i = line_value.index("name") + 6
                while(line_value[i] != "\""):
                    name += line_value[i]
                    i += 1
                if(graph_dict[father_id] == None):  # 如果父节点是根节点，则直接存入
                    graph_dict[id] = name
                else:
                    graph_dict[id] = graph_dict[father_id] + "." + name
                xxlog("Node: \"%s\": %s detected"%(id, graph_dict[id]))
            elif("Tensor" in line_type):                    # weight or calculate
                # 切分指令
                i = 0
                instr = ""
                while(line_value[i] != "(" and line_value[i] != "["):
                    instr += line_value[i]
                    i += 1
                # 切分参数列表
                line_value_temp = line_value.replace("%", "")
                i = line_value_temp.index("(") + 1
                args = ""
                while(line_value_temp[i] != ")"):
                    args += line_value_temp[i]
                    i += 1
                args = args.split(",")

                if(instr == "prim::GetAttr" and "name" in line_value):          # weight bias ...
                    name = ""
                    i = line_value.index("name") + 6
                    while(line_value[i] != "\""):
                        name += line_value[i]
                        i += 1
                    father_id = ""
                    i = line_value.index("(") + 2
                    while(line_value[i] != ")"):
                        father_id += line_value[i]
                        i += 1
                    graph_dict[id] = graph_dict[father_id] + "." + name
                    xxlog("Parameter \"%s\": %s detected"%(id, graph_dict[id]))
                elif(instr == "aten::conv2d" or instr == "aten::_convolution"):# elif(instr == "aten::_convolution"):                            # conv2d
                    # convolution中参数的前6位为input, weight, bias, stride, padding, dilation，后面的没发现有什么用
                    input_id = args[0]
                    # 根据参数从graph_dict里提取数据
                    input_shape = graph_dict[args[0]]
                    weight_name = graph_dict[args[1]]
                    bias_name = graph_dict[args[2]]
                    stride = graph_dict[args[3]]
                    padding = graph_dict[args[4]]
                    dilation = graph_dict[args[5]]
                    # 根据weight和bias名称从named_parameters中取Tensor
                    if(weight_name != None):
                        weight = parameter_dict[weight_name].copy()
                    else:
                        weight = None
                    if(bias_name != None):
                        bias = parameter_dict[bias_name].copy()
                    else:
                        bias = None
                    # 创建Conolution对象加入graph
                    graph.append(op.Conv2d(
                        id, input_id, weight, bias, weight.shape[0], weight.shape[1], 
                        [weight.shape[2], weight.shape[3]], stride, padding ,dilation
                    ))
                    # 推算输出shape，加入graph_dict
                    output_shape = []
                    output_shape.append(input_shape[0])
                    output_shape.append(weight.shape[0])
                    padded_size = [
                        input_shape[2] + 2 * padding[0],
                        input_shape[3] + 2 * padding[1]
                    ]
                    output_shape.append((padded_size[0] - (dilation[0] * (weight.shape[2]-1) + 1)) // stride[0] + 1)
                    output_shape.append((padded_size[1] - (dilation[1] * (weight.shape[3]-1) + 1)) // stride[1] + 1)
                    graph_dict[id] = output_shape
                    xxlog("Conv2d \"%s\": input:%s, output_channel:%s, input_channel:%s, kernel_size:%s, "\
                        "stride:%s, padding:%s, dilation:%s detected"%(
                        id, input_id, weight.shape[0], weight.shape[1], 
                        [weight.shape[2], weight.shape[3]], stride, padding, dilation
                    ))
                elif(instr == "aten::batch_norm"):                              # bn
                    # bn参数的1-5位为input, weight, bias, running_mean, running_var。7,8位为momentum, eps
                    input_id = args[0]
                    # 根据参数从graph_dict中提取数据
                    input_shape = graph_dict[args[0]]
                    weight_name = graph_dict[args[1]]
                    bias_name = graph_dict[args[2]]
                    running_mean_name = graph_dict[args[3]]
                    running_var_name = graph_dict[args[4]]
                    momentum = graph_dict[args[6]]
                    eps = graph_dict[args[7]]
                    # 根据weight bias名称从named_parameters中提取Tensor
                    if(weight_name != None):
                        weight = parameter_dict[weight_name].copy()
                    else:
                        weight = None
                    if(bias_name != None):
                        bias = parameter_dict[bias_name].copy()
                    else:
                        bias = None
                    # 根据running mean var名称从state_dict中提取Tensor
                    running_mean = state_dict[running_mean_name].detach().numpy()
                    running_var = state_dict[running_var_name].detach().numpy()
                    # 创建bn对象加入graph
                    graph.append(op.Batch_Norm2d(
                        id, input_id, eps, momentum, weight, bias, running_mean, running_var, input_shape[1]
                    ))
                    # 推算输出shape，加入graph_dict
                    output_shape = input_shape
                    graph_dict[id] = output_shape
                    xxlog("BatchNorm2d \"%s\": input:%s, num_features:%s, eps:%s, momentum:%s detected"%(
                        id, input_id, weight.shape[0], eps, momentum
                    ))
                elif(instr == "aten::relu_" or 
                     instr == "aten::relu"):                                   # relu
                    # relu参数，1位为input
                    input_id = args[0]
                    input_shape = graph_dict[args[0]]
                    # 创建relu对象加入graph
                    graph.append(op.Relu(
                        id, input_id
                    ))
                    # 推算output_shape，加入graph_dict
                    output_shape = input_shape
                    graph_dict[id] = output_shape
                    xxlog("Relu \"%s\": input:%s detected"%(id, input_id))
                elif(instr == "aten::max_pool2d"):                              # maxpool2d
                    # maxpool2d，参数1-5分别为input, kernel_size, stride, padding, dilation
                    input_id = args[0]
                    input_shape = graph_dict[args[0]]
                    kernel_size = graph_dict[args[1]]
                    stride = graph_dict[args[2]]
                    padding = graph_dict[args[3]]
                    dilation = graph_dict[args[4]]
                    # 创建maxpool2d对象加入graph
                    graph.append(op.Maxpool2d(
                        id, input_id, kernel_size, stride, padding, dilation
                    ))
                    # 推算output_shape，加入graph_dict
                    padded_size = [
                        input_shape[2] + 2 * padding[0],
                        input_shape[3] + 2 * padding[1]
                    ]
                    output_shape = []
                    output_shape.append(input_shape[0])
                    output_shape.append(input_shape[1])
                    output_shape.append((padded_size[0] - (dilation[0]*(kernel_size[0]-1)+1)) // stride[0] + 1)
                    output_shape.append((padded_size[1] - (dilation[1]*(kernel_size[1]-1)+1)) // stride[1] + 1)
                    graph_dict[id] = output_shape
                    xxlog("Maxpool2d \"%s\": input:%s, kernel_size:%s, stride=%s, padding=%s, dilation=%s detected"%(
                        id, input_id, kernel_size, stride, padding, dilation
                    ))
                elif(instr == "aten::avg_pool2d"):                              # avgpool2d
                    # avgpool2d，参数1-4分别为input, kernel_size, stride, padding
                    input_id = args[0]
                    input_shape = graph_dict[args[0]]
                    kernel_size = graph_dict[args[1]]
                    stride = graph_dict[args[2]]
                    padding = graph_dict[args[3]]
                    # 创建avgpool2d对象加入graph
                    graph.append(op.Avgpool2d(
                        id, input_id, kernel_size, stride, padding
                    ))
                    # 推算output_shape，加入graph_dict
                    padded_size = [
                        input_shape[2] + 2 * padding[0],
                        input_shape[3] + 2 * padding[1]
                    ]
                    output_shape = []
                    output_shape.append(input_shape[0])
                    output_shape.append(input_shape[1])
                    output_shape.append((padded_size[0] - kernel_size[0]) // stride[0] + 1)
                    output_shape.append((padded_size[1] - kernel_size[1]) // stride[1] + 1)
                    graph_dict[id] = output_shape
                    xxlog("Avgpool2d \"%s\": input:%s, kernel_size:%s, stride=%s, padding=%s detected"%(
                        id, input_id, kernel_size, stride, padding
                    ))
                elif(instr == "aten::add_" or 
                     instr == "aten::add" or instr == "aten::addmm"):                                     # add
                    # add参数1-2分别为input1, input2
                    input1_id = args[0]
                    input2_id = args[1]
                    # 提取参数
                    input1_shape = graph_dict[args[0]]
                    input2_shape = graph_dict[args[1]]
                    # 创建对象加入graph
                    graph.append(op.Add(
                        id, input1_id, input2_id
                    ))
                    # 推算output_shape。既然已经出现在模型里了，那么shape一定是可加的，不需要再判断
                    output_shape = broadcast_shape(input1_shape, input2_shape)
                    graph_dict[id] = output_shape
                    xxlog("Add \"%s\": input1:%s, input2:%s detected"%(id, input1_id, input2_id))
                elif(instr == "aten::adaptive_avg_pool2d"):                     # adaptive avgpool2d
                    # adaptive avgpool2d参数1-2为input, output_shape
                    input_id = args[0]
                    input_shape = graph_dict[args[0]]
                    output_size = graph_dict[args[1]]
                    # 转换为avgpool2d，推算kernel_size, stride, padding, dilation
                    kernel_size = [input_shape[2] // output_size[0], input_shape[3] // output_size[1]]
                    stride = kernel_size
                    padding = [0, 0]
                    # 创建对象，存入graph
                    graph.append(op.Avgpool2d(
                        id, input_id, kernel_size, stride, padding
                    ))
                    # 推断output shape，存入graph_dict
                    output_shape = []
                    output_shape.append(input_shape[0])
                    output_shape.append(input_shape[1])
                    output_shape.append(output_size[0])
                    output_shape.append(output_size[1])
                    graph_dict[id] = output_shape
                    xxlog("AdaptiveAvgpool2d \"%s\": input:%s detected. Converted to Avgpool2d: kernel_size=%s, " \
                        "stride=%s, padding=%s"%(id, input_id, kernel_size, stride, padding))
                elif(instr == "aten::flatten"):                                 # flatten
                    # flatten参数1-3为input, start_dim, end_dim
                    input_id = args[0]
                    start_dim = graph_dict[args[1]]
                    end_dim = graph_dict[args[2]]
                    if(start_dim != 1 or end_dim != -1):
                        raise ValueError("Only support flatten with start_dim=1 and end_dim=-1")
                    # 从graph_dict提取参数
                    input_shape = graph_dict[args[0]]
                    # 创建对象存入graph
                    graph.append(op.Flatten(
                        id, input_id
                    ))
                    # 推算output_shape存入graph_dict
                    output_shape = []
                    output_shape.append(input_shape[0])
                    output_shape.append(input_shape[1] * input_shape[2] * input_shape[3])
                    graph_dict[id] = output_shape
                    xxlog("Flatten \"%s\": input:%s detected"%(id, input_id))
                elif(instr == "aten::t"):                                       # transpose
                    # 在计算图里见到的这个是在对fc.weight转置，因为fc.weight是以[output,input]存储的，
                    # 但计算时应为hidden*weight，所以需要转置才能使shape匹配
                    # 但并不知道它会不会用来对中间Tensor转置，所以分类讨论
                    input_id = args[0]
                    if(is_in_graph(graph, input_id)):           # 如果该input_id存在于graph中，则说明是对中间结果转置
                        raise TypeError("Tranpose a tensor not supported yet")
                    else:                                       # 否则是对张量转置
                        # 将parameter_dict中的权重转置。不需要对已加入graph的权重转置，因为它们是在转置之前加入的
                        input_name = graph_dict[input_id]
                        parameter_dict[input_name] = parameter_dict[input_name].T
                        # 直接继承转置之前的名字，加入graph_dict
                        graph_dict[id] = input_name
                    xxlog("Transpose \"%s\": input:%s detected"%(id, input_id))
                elif(instr == "aten::linear"):                                   # linear
                    # 发现pytorch使用addmm表示fc，torch1.9.1已经改成用linear表示
                    # 参数按addmm为input, mat1, mat2, beta, alpha
                    # 所做计算为out = beta*input + alpha(mat1*mat2)，即gemm
                    # 参数按fc为bias, input, weight, beta, alpha，此时应限定beta=1, alpha=1

                    # 参数按linear为input, weight, bias
                    input_id = args[0]
                    weight_name = graph_dict[args[1]]
                    bias_name = graph_dict[args[2]]
                    input_shape = graph_dict[args[0]]
                    # if(beta != 1 or alpha != 1):
                    #     xxlog("Only support beta=1 and alpha=1 in addmm", XXError())
                    #     raise ValueError("Only support beta=1 and alpha=1 in addmm")
                    if(weight_name != None):
                        weight = parameter_dict[weight_name]
                    else:
                        weight = None
                    if(bias_name != None):
                        bias = parameter_dict[bias_name]
                    else:
                        bias = None
                    # 创建对象加入graph
                    graph.append(op.Dense(
                        id, input_id, weight, bias, weight.shape[0], weight.shape[1]
                    ))
                    # 推算output_shape存入graph_dict
                    output_shape = []
                    output_shape.append(input_shape[0])
                    output_shape.append(weight.shape[0])
                    graph_dict[id] = output_shape
                    xxlog("Fc \"%s\": input:%s, output_channel:%s, input_channel:%s detected"%(
                        id, input_id, weight.shape[0], weight.shape[1]
                    ))
                elif(instr == "aten::dropout"):                                 # dropout
                    # 参数1-3为input, p, inplace
                    input_id = args[0]
                    input_shape = graph_dict[args[0]]
                    p = graph_dict[args[1]]
                    # 创建对象加入graph
                    graph.append(op.Dropout(
                        id, input_id, p
                    ))
                    # 推算output_shape存入graph_dict
                    output_shape = []
                    output_shape = input_shape
                    graph_dict[id] = output_shape
                    xxlog("Dropout \"%s\": input:%s, p:%s detected"%(id, input_id, p))
                elif(instr == "aten::cat"):                                 # concat
                    # 参数1-2为input, dim
                    input_id = args[0]
                    inputs = graph_dict[input_id]
                    dim = graph_dict[args[1]]
                    cur_input1_id = inputs[0]
                    for i in range(1, len(inputs)):
                        cur_id = reserve_id if i < len(inputs) - 1 else id
                        # 创建对象加入graph
                        graph.append(op.Concat(
                            cur_id, 
                            cur_input1_id, 
                            inputs[i],
                            dim
                        ))
                        # 推算output_shape存入graph_dict
                        output_shape = graph_dict[cur_input1_id].copy()
                        output_shape[dim] += graph_dict[inputs[i]][dim]
                        graph_dict[cur_id] = output_shape

                        xxlog("Concat \"%s\": input1:%s, input2:%s, dim:%d detected"%(cur_id, cur_input1_id, inputs[i], dim))
                        cur_input1_id = cur_id
                        if i < len(inputs) - 1:
                            reserve_id -= 1
                else:
                    print(instr)
                    xxlog("Unknown op: %s"%(line), XXError())
                    raise TypeError("Unknown op")

    return graph, graph_dict, parameter_dict


def rearrange_graph(graph):
    '''
    对模型进行编号排序。由于原始模型中算子的编号是多样的，且不按顺序，所以给它们重新编号
    '''
    number = 0
    graph_len = len(graph)
    # 遍历每个节点
    for n,node in enumerate(graph):
        node_id = node.id
        node.id = str(number)
        number += 1
        # 遍历后续节点
        for s in range(n+1, graph_len):
            # 对于以当前节点的输出作为输入的后续节点，修改其输入节点的id
            try:
                if(graph[s].input_id == node_id):
                    graph[s].input_id = node.id
            except:
                pass
            try:
                if(graph[s].input1_id == node_id):
                    graph[s].input1_id = node.id
            except:
                pass
            try:
                if(graph[s].input2_id == node_id):
                    graph[s].input2_id = node.id
            except:
                pass


def save_graph(graph, output_dir):
    '''
    存储模型
    '''
    string = ""
    for node in graph:
        print(node)
        string += node.output(output_dir) + "\n"
    with open(output_dir+"/graph.txt", "w") as f:
        f.write(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get PyTorch JIT Model")

    parser.add_argument("-m", "--model", required=True ,help="pytorch jit model path")
    parser.add_argument("-s", "--input_shape", required=True, help="input Tensor shape, e.g.: 1,3,224,224")
    parser.add_argument("-o", "--output_dir", default="output", help="output directory")

    args = parser.parse_args()

    clear_log()

    model_path = args.model
    input_shape = args.input_shape
    output_dir = args.output_dir

    xxlog("Read model_path: %s"%(model_path), XXInfo())
    xxlog("Read input_shape: %s"%(input_shape))
    xxlog("Read output_dir: %s"%(output_dir))

    if(model_path == None):
        xxlog("Model path connt be None", XXError())
        raise ValueError("model cannot be None")

    graph, graph_dict, parameter_dict = read_model(model_path, input_shape, output_dir)
    xxlog("Read model finished")

    rearrange_graph(graph)
    xxlog("Graph rearrange finished")

    save_graph(graph, output_dir)
    xxlog("Extract result save at %s"%(output_dir))