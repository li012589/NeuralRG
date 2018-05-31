import torch
import numpy as np


def latentU(z,flow,source):
    x,_ = flow.generate(z)
    return -(flow.prior.logProbability(z)+source.logProbability(x)-flow.logProbability(x))


def learn(source, flow, batchSize, epochs, lr=1e-3, save = True, saveSteps = 10, weight_decay = 0.001, adaptivelr = True, measureFn = None):
    params = list(flow.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    nparams = sum([np.prod(p.size()) for p in params])
    print ('total nubmer of trainable parameters:', nparams)
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    if adaptivelr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.7)

    LOSS = []
    ETC = []

    for epoch in range(epochs):
        x,sampleLogProbability = flow.sample(batchSize)
        loss = sampleLogProbability.mean() - source.logProbability(x).mean()
        flow.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch:",epoch, "L:",loss.item())

        LOSS.append([loss.item()])

        if save and epoch%saveSteps == 0:
            save_dict = flow.save({})
            torch.save(save_dict, flow.name+'_epoch'+str(epoch)+".saving")

    return flow