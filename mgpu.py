import torch
import tvm.exec.rpc_tracker
import tvm.exec.rpc_server
from tvm.rpc.tracker import Tracker
from tvm import rpc
import os, nvgpu

class mgpu():

    def __init__(self, uuid):
        # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        self.gpu_num = torch.cuda.device_count()
        self.gpu_name = [torch.cuda.get_device_name()]
        self.aaaa = self.aaaaa()
        self.tracker_host = '0.0.0.0'
        self.tracker_port = 9190
        self.tracker_port_end = 9199
        self.tracker_silent = True

        self.server_host = '0.0.0.0'
        self.server_port = 9090
        self.server_port_end=9199
        self.server_tracker_url = self.tracker_host
        self.server_tracker_port = self.tracker_port
        self.server_tracker_addr = (self.server_tracker_url, self.server_tracker_port)
        self.server_key = self.gpu_name
        self.server_silent = True

    def aaaaa(self):
        return torch.cuda.get_device_properties(0)

    def add_rpc_server(self, key):
        server = rpc.Server(
            self.server_host,
            self.server_port,
            self.server_port_end,
            key=key,
            tracker_addr=self.server_tracker_addr,
            silent=self.server_silent,
            no_fork=False,
        )
        return server

    def add_rpc_tracker(self):
        tracker = Tracker(host='0.0.0.0', port=9190, port_end=9199, silent=True)
        return tracker

    def use_mgpu(self):
        tracker = self.add_rpc_tracker()
        tracker.proc.join()

        servers = []
        for i in range(self.num_gpu):
            servers.append(self.add_rpc_server(key=self.gpu_name[i]))
        for i in servers:
            i.proc.join()



if __name__ == '__main__':
    print("eeeee")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    gpu_num = torch.cuda.device_count()
    os.environ.pop('CUDA_VISIBLE_DEVICES')
    os.environ.clear()
    os.system('CUDA_VISIBLE_DEVICES=0,1')
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # aaa = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_num1 = torch.cuda.device_count()
    a = []
    uuids = nvgpu.gpu_info()
    gpu_name = torch.cuda.get_device_name(0)
    for i in uuids:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        a.append(mgpu(str(i['uuid'])))