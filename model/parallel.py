import threading

def parallelize(model,deviceID,inputs,dims = 0):
    models = []
    for device in deviceID:
        tmp = model.cuda(device)
        models.append(tmp)

    nGroup = len(deviceID)
    nBatch = len(device)//nGroup
    nLast = nGroup - nBatch*nGroup

    group = []
    for i in range(nGroup):
        end = (i+1)*nBatch
        if i == nGroup-1:
            group.append(inputs[end-nBatch,:])
        else:
            group.append(inputs[end-nBatch,end])

    result = []
    def _work(model,inputs):
        pass
