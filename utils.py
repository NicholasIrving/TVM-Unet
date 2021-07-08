import time
import os
def gen_log(model_name):
    folder = './tuningLog/'
    if os.path.exists(folder):
        pass
    else:
        os.makedirs(folder)

    now = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log_name = folder + model_name + now + '.log'

    return log_name


def gen_tmp_log(log_name):
    tmp_log = log_name + '.tmp'
    if os.path.exists(tmp_log):
        os.remove(tmp_log)

    return tmp_log