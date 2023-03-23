

import struct


class Dense:
    def __init__(
        self,
        id,
        input_id,
        weight,
        bias,
        output_channel,
        input_channel
    ):
        self.id = id
        self.input_id = input_id
        self.weight = weight
        self.bias = bias
        self.output_channel = output_channel
        self.input_channel = input_channel

    def __str__(self):
        string = "%%%s=nn.dense(input=%%%s, weight=%s, bias=%s, output_channel=%d, input_channel=%d);"%(
            self.id, self.input_id, self.weight.shape, 
            None if (self.bias is None) else self.bias.shape,
            self.output_channel, self.input_channel
        )
        return string
    
    def output(self, output_dir):
        weight_path_in_graph = "layer_%s_weight.bin"%(self.id)
        bias_path_in_graph = "layer_%s_bias.bin"%(self.id)
        weight_path = "%s/layer_%s_weight.bin"%(output_dir, self.id)
        bias_path = "%s/layer_%s_bias.bin"%(output_dir, self.id)
        # weight bias写入文件
        self.weight.tofile(weight_path)
        if(not self.bias is None):
            self.bias.tofile(bias_path)
        else:
            bias_path_in_graph = "None"
            bias_path = "None"
        string = "%%%s=nn.dense(input=%%%s, weight=%s, bias=%s, output_channel=%d, input_channel=%d);"%(
            self.id, self.input_id, weight_path_in_graph, bias_path_in_graph,
            self.output_channel, self.input_channel
        )
        return string


class Input:
    def __init__(
        self,
        id,
        input_shape
    ):
        self.id = id
        self.input_shape = input_shape
    
    def __str__(self):
        string = "%%%s=input(shape=%s);"%(
            self.id, self.input_shape
        )
        return string
    
    def output(self, output_dir):
        string = "%%%s=input(shape=(%d,%d,%d,%d));"%(
            self.id, self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]
        )
        return string


class Output:
    def __init__(
        self, 
        id, 
        input_id
    ):
        self.input_id = input_id
        self.id = id
    def __str__(self):
        string = "%%%s=output(input=%%%s);"%(
            self.id, self.input_id
        )
        return string
    
    def output(self, output_dir):
        string = "%%%s=output(input=%%%s);"%(
            self.id, self.input_id
        )
        return string


class Conv2d:
    def __init__(
        self, 
        id, 
        input_id, 
        weight, 
        bias, 
        output_channel, 
        input_channel, 
        kernel_size, 
        stride, 
        padding, 
        dilation
    ):
        self.id = id
        self.input_id = input_id
        self.weight = weight
        self.bias = bias
        self.output_channel = output_channel
        self.input_channel = input_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
    
    def __str__(self):
        string = "%%%s=nn.conv2d(input=%%%s, weight=%s, bias=%s, output_channel=%d, input_channel=%d, " \
        "kernel_size=%s, stride=%s, padding=%s, dilation=%s);"%(
            self.id, self.input_id, self.weight.shape, 
            None if (self.bias is None) else self.bias.shape, 
            self.output_channel, self.input_channel,
            self.kernel_size, self.stride, self.padding,self.dilation
        )
        return string
    
    def output(self, output_dir):
        weight_path_in_graph = "layer_%s_weight.bin"%(self.id)
        bias_path_in_graph = "layer_%s_bias.bin"%(self.id)
        weight_path = "%s/layer_%s_weight.bin"%(output_dir, self.id)
        bias_path = "%s/layer_%s_bias.bin"%(output_dir, self.id)
        self.weight.tofile(weight_path)
        if(not self.bias is None):
            self.bias.tofile(bias_path)
        else:
            bias_path_in_graph = "None"
            bias_path = "None"
        string = "%%%s=nn.conv2d(input=%%%s, weight=%s, bias=%s, output_channel=%d, input_channel=%d, " \
        "kernel_size=(%d,%d), stride=(%d,%d), padding=(%d,%d), dilation=(%d,%d));"%(
            self.id, self.input_id, weight_path_in_graph, bias_path_in_graph,
            self.output_channel, self.input_channel,
            self.kernel_size[0], self.kernel_size[1], self.stride[0], self.stride[1], 
            self.padding[0], self.padding[1], self.dilation[0], self.dilation[1]
        )
        return string


class Batch_Norm2d:
    def __init__(
        self,
        id, 
        input_id,
        eps,
        momentum,
        weight,
        bias,
        running_mean,
        running_var,
        num_features
    ):  
        self.id = id
        self.input_id = input_id
        self.eps = eps
        self.momentum = momentum
        self.weight = weight
        self.bias = bias
        self.running_mean = running_mean
        self.running_var = running_var
        self.num_features = num_features
    
    def __str__(self):
        string = "%%%s=nn.batch_norm2d(input=%%%s, weight=%s, bias=%s, running_mean=%s, running_var=%s, " \
        "num_features=%d, eps=%f, momentum=%f);"%(
            self.id, self.input_id, self.weight.shape, 
            None if (self.bias is None) else self.bias.shape,
            self.running_mean.shape, self.running_var.shape,
            self.num_features, self.eps, self.momentum
        )
        return string
    
    def output(self, output_dir):
        weight_path_in_graph = "layer_%s_weight.bin"%(self.id)
        bias_path_in_graph = "layer_%s_bias.bin"%(self.id)
        running_mean_path_in_graph = "layer_%s_running_mean.bin"%(self.id)
        running_var_path_in_graph = "layer_%s_running_var.bin"%(self.id)
        weight_path = "%s/layer_%s_weight.bin"%(output_dir, self.id)
        bias_path = "%s/layer_%s_bias.bin"%(output_dir, self.id)
        running_mean_path = "%s/layer_%s_running_mean.bin"%(output_dir, self.id)
        running_var_path = "%s/layer_%s_running_var.bin"%(output_dir, self.id)
        self.weight.tofile(weight_path)
        if(not self.bias is None):
            self.bias.tofile(bias_path)
        else:
            bias_path_in_graph = "None"
            bias_path = "None"
        self.running_mean.tofile(running_mean_path)
        self.running_var.tofile(running_var_path)
        string = "%%%s=nn.batch_norm2d(input=%%%s, weight=%s, bias=%s, running_mean=%s, running_var=%s, " \
        "num_features=%d, eps=%f, momentum=%f);"%(
            self.id, self.input_id, weight_path_in_graph, bias_path_in_graph,
            running_mean_path_in_graph, running_var_path_in_graph,
            self.num_features, self.eps, self.momentum
        )
        return string


class Relu:
    def __init__(
        self, 
        id,
        input_id
    ):
        self.id = id
        self.input_id = input_id

    def __str__(self):
        string = "%%%s=nn.relu(input=%%%s);"%(
            self.id, self.input_id
        )
        return string
    
    def output(self, output_dir):
        string = "%%%s=nn.relu(input=%%%s);"%(
            self.id, self.input_id
        )
        return string


class Maxpool2d:
    def __init__(
        self,
        id,
        input_id,
        kernel_size,
        stride,
        padding,
        dilation
    ):
        self.id = id
        self.input_id = input_id
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
    
    def __str__(self):
        string = "%%%s=nn.maxpool2d(input=%%%s, kernel_size=%s, stride=%s, padding=%s, dilation=%s);"%(
            self.id, self.input_id, self.kernel_size, self.stride, self.padding, self.dilation
        )
        return string
    
    def output(self, output_dir):
        string = "%%%s=nn.maxpool2d(input=%%%s, kernel_size=(%d,%d), stride=(%d,%d), " \
        "padding=(%d,%d), dilation=(%d,%d));"%(
            self.id, self.input_id, self.kernel_size[0], self.kernel_size[1], 
            self.stride[0], self.stride[1], self.padding[0], self.padding[1], 
            self.dilation[0], self.dilation[1]
        )
        return string


class Add:
    def __init__(
        self,
        id,
        input1_id,
        input2_id
    ):
        self.id = id
        self.input1_id = input1_id
        self.input2_id = input2_id
    
    def __str__(self):
        string = "%%%s=add(input1=%%%s, input2=%%%s);"%(
            self.id, self.input1_id, self.input2_id
        )
        return string
    
    def output(self, output_dir):
        string = "%%%s=add(input1=%%%s, input2=%%%s);"%(
            self.id, self.input1_id, self.input2_id
        )
        return string


class Avgpool2d:
    def __init__(
        self,
        id,
        input_id,
        kernel_size,
        stride,
        padding
    ):
        self.id = id
        self.input_id = input_id
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def __str__(self):
        string = "%%%s=nn.avgpool2d(input=%%%s, kernel_size=%s, stride=%s, padding=%s);"%(
            self.id, self.input_id, self.kernel_size, self.stride, self.padding
        )
        return string
    
    def output(self, output_dir):
        string = "%%%s=nn.avgpool2d(input=%%%s, kernel_size=(%d,%d), stride=(%d,%d), "\
        "padding=(%d,%d));"%(
            self.id, self.input_id, self.kernel_size[0], self.kernel_size[1], 
            self.stride[0], self.stride[1], self.padding[0], self.padding[1]
        )
        return string


class Flatten:
    def __init__(
        self,
        id,
        input_id
    ):
        self.id = id
        self.input_id = input_id

    def __str__(self):
        string = "%%%s=nn.flatten(input=%%%s);"%(
            self.id, self.input_id
        )
        return string
    
    def output(self, output_dir):
        string = "%%%s=nn.flatten(input=%%%s);"%(
            self.id, self.input_id
        )
        return string


class Dropout:
    def __init__(
        self,
        id,
        input_id,
        p
    ):
        self.id = id
        self.input_id = input_id
        self.p = p
    
    def __str__(self):
        string = "%%%s=nn.dropout(input=%%%s, p=%f);"%(
            self.id, self.input_id, self.p
        )
        return string
    
    def output(self, output_dir):
        string = "%%%s=nn.dropout(input=%%%s, p=%f);"%(
            self.id, self.input_id, self.p
        )
        return string


class Concat:
    def __init__(
        self,
        id,
        input1_id,
        input2_id,
        dim
    ):
        self.id = id
        self.input1_id = input1_id
        self.input2_id = input2_id
        self.dim = dim
    
    def __str__(self):
        string = "%%%s=concat(input1=%%%s, input2=%%%s, dim=%d);"%(
            self.id, self.input1_id, self.input2_id, self.dim
        )
        return string
    
    def output(self, output_dir):
        string = "%%%s=concat(input1=%%%s, input2=%%%s, dim=%d);"%(
            self.id, self.input1_id, self.input2_id, self.dim
        )
        return string