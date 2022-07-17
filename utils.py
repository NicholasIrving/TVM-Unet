import time
import os
def gen_log(model_name, tuner):
    folder = './tuningLog/'
    if os.path.exists(folder):
        pass
    else:
        os.makedirs(folder)

    now = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log_name = folder + model_name + '-' + tuner + '-' + now + '.log'

    return log_name


def gen_tmp_log(log_name):
    tmp_log = log_name + '.tmp'
    if os.path.exists(tmp_log):
        os.remove(tmp_log)

    return tmp_log

def analysis_config_space(tasks):
    import os
    for i in range(len(tasks)):
        for j in tasks[i].config_space:
            obj = tasks[i].config_space.space_map[j]
            folder = '/disk1/NiCholas/aa/task[%d]/' % (i)
            if not os.path.exists(folder):
                os.makedirs(folder)
            file = '/disk1/NiCholas/aa/task[%d]/%s.txt' % (i, j.replace('\'', ''))
            with open(file, 'a') as f:
                f.write(j + '\n')
                if j != 'auto_unroll_max_step' and j != 'unroll_explicit':
                    f.write('product:' + str(obj.product) + '\n')
                    f.write('factors:' + str(obj.factors) + '\n')
                    f.write('num_output:' + str(obj.num_output) + '\n')
                    f.write('entities:' + '\n')
                    for k in obj.entities:
                        f.write(str(k) + '\n')
                    f.close()
                else:
                    f.write('entities:' + '\n')
                    for k in obj.entities:
                        f.write(str(k) + '\n')
                    f.close()