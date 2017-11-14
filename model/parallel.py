import threading

def parallelize(model,deviceID,fnName,inputs,args):

    models = []
    for device in deviceID:
        tmp = model.cuda(device)
        models.append(tmp)

    nGroup = len(deviceID)
    nBatch = len(inputs)//nGroup
    #nLast = nGroup - nBatch*nGroup

    group = {}
    for i in range(nGroup):
        end = (i+1)*nBatch
        if i == nGroup-1:
            group[i]=(inputs[end-nBatch:-1].cuda(deviceID[i]))
        else:
            group[i]=(inputs[end-nBatch:end].cuda(deviceID[i]))

    lock = threading.Lock()

    results = {}
    def _work(i,model,inputs):
        result = getattr(model,fnName)(inputs,*args)
        with lock:
            results[i] = result

    threads = [threading.Thread(target=_work,args = (i,models[i],group[i])) for i in range(nGroup)]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    output = []
    for i in range(nGroup):
        output.append(results[i].cuda(deviceID[0]))

    return output
