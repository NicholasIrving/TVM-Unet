import tvm, tvm.testing
from tvm import autotvm, relay
import torch, torchvision
import numpy as np
from utils import *
from modelList import get_model
import argparse
import multiprocessing
from evaluate import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description='Auto-tune a pytorch model')
    parser.add_argument('model', default='resnet18', help='Choose a model to tune')
    parser.add_argument('--eval', action='store_true', help='Evaluate and compared')
    parser.add_argument('--tuner', choices=['xgb', 'ga', 'random', 'grid'], default='xgb', help='Choose a tuner to tune')
    parser.add_argument('--device',  choices=['cpu', 'cuda'], default='cuda', help='Choose a device to tune')
    parser.add_argument('--input_shape', default=(1, 3, 224, 224), help='Input shape for the model')
    parser.add_argument('--dtype', default='float32', help='Data type of model')
    parser.add_argument('--optimize', default='nn.conv2d', help='Relay ops to be tuned. If not specified, all tunable ops will be extracted')
    parser.add_argument('--n_trial', default=2000, help='Maximum number of configs to try (measure on real hardware)')
    parser.add_argument('--early_stopping', default=600, help='Early stop the tuning when not finding better configs in this number of trials')

    args = parser.parse_args()
    cuda_path = os.path.dirname(os.path.dirname(os.popen('which nvcc').read()))
    os.environ['PATH'].join(cuda_path + '/bin')
    if 'LD_LIBRARY_PATH' not in os.environ:
        os.environ['LD_LIBRARY_PATH'] = cuda_path + '/lib64'

    return args


args = parse_args()
print('Loading model.........')
model = get_model(args.model)


if args.device == 'cuda':
    target = tvm.target.cuda()
    n_parallel = target.max_num_threads
    device = tvm.cuda()
else:
    target = tvm.target.Target('llvm')
    n_parallel = multiprocessing.cpu_count()
    device = tvm.cpu()

input_shape = args.input_shape
input_data = torch.randn(input_shape)
input_name = 'input'
shape_list = [(input_name, input_shape)]
scripted_model = torch.jit.trace(model, input_data).eval()

print('Getting relay.........')
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list, default_dtype=args.dtype)

print('Extracting tasks.........')
tasks = autotvm.task.extract_from_program(mod['main'], params, target, ops=(relay.op.get(args.optimize),))

log_name = gen_log(args.model)
tmp_log = gen_tmp_log(log_name)
tune_option = {
    'tmp_log': tmp_log,
    'tuner': args.tuner,
    'n_trial': args.n_trial,
    'early_stopping': args.early_stopping,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(n_parallel=n_parallel),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150)
    )
}

print('Tuning.........')
for i, task in enumerate(tasks):

    if args.tuner == "xgb":
        tuner = tvm.autotvm.tuner.XGBTuner(task, loss_type="rank")
    elif args.tuner == "ga":
        tuner = autotvm.tuner.GATuner(task, pop_size=100)
    elif args.tuner == "random":
        tuner = autotvm.tuner.RandomTuner(task)
    elif args.tuner == "grid":
        tuner = autotvm.tuner.GridSearchTuner(task)

    prefix = '[Task %2d / %2d]' % (i + 1, len(tasks))
    n_trial = min(tune_option['n_trial'], len(task.config_space))
    tuner.tune(
        n_trial=n_trial,
        early_stopping=tune_option['early_stopping'],
        measure_option=tune_option['measure_option'],
        callbacks=[
            autotvm.callback.progress_bar(n_trial, prefix=prefix),
            autotvm.callback.log_to_file(tune_option['tmp_log'])
        ]
    )

autotvm.record.pick_best(tmp_log, log_name)

if args.eval:
    evaluate(mod, params, target, device, log_name, input_shape, args.dtype)


