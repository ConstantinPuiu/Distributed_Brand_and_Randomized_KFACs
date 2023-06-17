import torch

class class_object_to_replicate_nn_module_class_name_structure:
    def __init__(self, layer_type):
        self.__name__ = layer_type

class lightweight_module:
    def __init__(self, torch_nn_module_to_turn_lightweight, current_counter):
        
        if torch_nn_module_to_turn_lightweight.__class__.__name__ == 'Linear':
            self.name = 'L_' + str(current_counter)
            self.__class__ = class_object_to_replicate_nn_module_class_name_structure('Linear')
        elif torch_nn_module_to_turn_lightweight.__class__.__name__ == 'Conv2d':
            self.name = 'C_' + str(current_counter)
            self.__class__ = class_object_to_replicate_nn_module_class_name_structure('Conv2d')
            self.kernel_size = torch_nn_module_to_turn_lightweight.kernel_size
            self.stride = torch_nn_module_to_turn_lightweight.stride
            self.padding = torch_nn_module_to_turn_lightweight.padding
            
        if torch_nn_module_to_turn_lightweight.bias is not None:
            self.bias = not None
        else:
            self.bias = None
        
        