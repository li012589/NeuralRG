from multiprocessing import Pool,Queue,Process,SimpleQueue
import sys
import subprocess
import setting
import numpy
import copy

q = Queue()

commands=[]
for name,content in setting.parameters.items():
    if len(commands) == 0:
        commands = [setting.command+[name]+[i] for i in content]
    else:
        step = len(commands)
        commands = [x for _ in range(len(content)) for x in copy.deepcopy(commands)]
        for n,i in enumerate(content):
            for j in range(step):
                commands[n*step+j] += [name]+[i]
for c in commands:
    q.put(c)


qRev = SimpleQueue()

def worker(settings):
    while not q.empty():
        setting.before()
        command = q.get()
        command += settings
        print("[Core] Working on:",''.join(i+' ' for i in command))
        output = subprocess.check_output(command)
        save = setting.process(output.decode('utf-8').split('\n'))
        qRev.put([''.join(i+' ' for i in command),save])
        setting.after()
        print("[Core] Work finish:",''.join(i+' ' for i in command))
        sys.stdout.flush()
    return 0

processes = []
for i in range(setting.maximumJobs):
    print("[Core] Initing work",str(i),"with setting:",''.join(i+' ' for i in setting.settings[i]))
    p = Process(target = worker,args=(setting.settings[i],))
    p.start()
    processes.append(p)

for p in processes:
    p.join()

print("[Core] Workers all finished")
RES = {}
while not qRev.empty():
    res = qRev.get()
    RES[res[0]] = res[1]
setting.finish(RES)

