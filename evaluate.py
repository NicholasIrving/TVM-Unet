import tvm, tvm.testing
import time
from tvm import autotvm, relay
from tvm.contrib import graph_executor
import numpy as np
import argparse
import torch
from modelList import get_model
import os

def evaluate(mod, params, target, device, log_file, input_shape, dtype):
    print('Evaluating ......')
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build_module.build(mod, target=target, params=params)
    module = graph_executor.GraphModule(lib['default'](device))
    data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
    module.set_input('input', data_tvm)

    ftime = module.module.time_evaluator('run', device, number=1, repeat=600)
    prof_res = np.array(ftime().results) * 1000

    print(
        "Mean inference time of original model (std dev): %.2f ms (%.2f ms)"
        % (np.mean(prof_res), np.std(prof_res))
    )

    # evaluating tuned model
    with autotvm.apply_history_best(log_file):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)
        module = graph_executor.GraphModule(lib['default'](device))
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('input', data_tvm)

        ftime = module.module.time_evaluator('run', device, number=1, repeat=600)
        prof_res = np.array(ftime().results) * 1000

        print(
            "Mean inference time of tuned model (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )


if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('model', help='Choose the model to evaluate')
        parser.add_argument('log_file', help='Choose the tuned log file')
        parser.add_argument('--device',  choices=['cpu', 'cuda'], default='cuda', help='Choose a device to evaluate')
        parser.add_argument('--input_shape', default="1, 3, 224, 224", help='Input shape for the model')
        parser.add_argument('--dtype', default='float32', help='Input data type')

        args = parser.parse_args()
        return args

    args = parse_args()
    cuda_path = os.path.dirname(os.path.dirname(os.popen('which nvcc').read()))
    os.environ['PATH'].join(cuda_path + '/bin')
    if 'LD_LIBRARY_PATH' not in os.environ:
        os.environ['LD_LIBRARY_PATH'] = cuda_path + '/lib64'

    if args.device == 'cuda':
        target = tvm.target.cuda()
        device = tvm.cuda()
    else:
        target = tvm.target.Target('llvm')
        device = tvm.cpu()

    log_name = args.log_file
    dtype = args.dtype

    input_shape = tuple([int(i) for i in args.input_shape.split(',')])
    input_data = torch.randn(input_shape)
    input_name = 'input'
    shape_list = [(input_name, input_shape)]

    model = get_model(args.model)
    script_model = torch.jit.trace(model, input_data).eval()
    mod, params = relay.frontend.from_pytorch(script_model, shape_list)

    evaluate(mod, params, target, device, log_name, input_shape, dtype)







