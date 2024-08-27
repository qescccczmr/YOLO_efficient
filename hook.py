import torch.nn as nn
import torch
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule,self).__init__()
        self.fc1=nn.Linear(10,5)
        self.fc2=nn.Linear(5,2)
    def forward(self,x):
        x=self.fc1(x)
        x=self.fc2(x)
        return x
    
#创建模型实例
model=MyModule()
def forward_hook(module,input,output):
    print(f"inside forward hook for{module.__class__.__name__}")
    print(f"input shape:{input[0].shape}")
    print(f"output shape:{output.shape}")
#注册前项传播hook
hook_handle=model.fc2.register_forward_hook(forward_hook)
#准备数据进行前项传播
input_data=torch.randn(1,10)
output=model(input_data)
#注销前项传播hook 
hook_handle.remove()
ptq_origin_layer_pairs = [[],[],[],[]]
pqt_output=[]
origin_output=[]
def make_layer_forward_hook(module_output_list):
    def forward_hook(module,input,output):
        module_output_list.append(output)
    return forward_hook
remove_handle=[]
for ptq_m,orig_m in ptq_origin_layer_pairs:
    remove_handle.append(ptq_m.register_forward_hook(make_layer_forward_hook(pqt_output)))
    remove_handle.append(orig_m.register_forward_hook(make_layer_forward_hook(origin_output)))
for i in remove_handle:
    i.remove()
ptq_model(images)
ori_model(images)
loss=0
for index,(pqt_output,origin_output) in enumerate(zip(pqt_output,origin_output)):
    pass