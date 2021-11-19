from evaluator import evaluate, sample_from_prior
import torch
import numpy as np
import json
import sys
from plots import plots



def get_IS_sample(exp):
    #init calc:
    # output = lambda _, x: x
    # sigma = {'logW':0}
    # res =  evaluate(exp, sigma, env=None)(sigma, 'addr_start', output)
    # #TODO : hint, "get_sample_from_prior" as a basis for your solution

    res, logW = sample_from_prior(exp)
    return logW, res

if __name__ == '__main__':

    for j in range(1,5):
        with open('C:/Users/jlovr/CS532-HW6/SMC/programs/{}.daphne'.format(j),'r') as f:
            exp = json.load(f)
        print('\n\n\nSample of prior of program {}:'.format(j))
        log_weights = []
        values = []
        for i in range(50):
            # if i%500==0:
            #     print(i)
            logW, sample = get_IS_sample(exp)
            log_weights.append(logW)
            values.append(sample)

        # print(log_weights)
        # print(values)
        
        logWs = torch.tensor(log_weights)

        values = torch.stack(values)
        values = values.reshape((values.shape[0],values.size().numel()//values.shape[0]))
        if torch.count_nonzero(logWs) == 0:
            print('covariance: ', np.cov(values.float().detach().numpy(),rowvar=False))    
            print('posterior mean:', values.float().detach().numpy().mean(axis=0))
            weighted_samples =values
        else:
            log_Z = torch.logsumexp(logWs,0) - torch.log(torch.tensor(logWs.shape[0],dtype=float))
            log_norm_weights = logWs - log_Z
            weights = torch.exp(log_norm_weights).detach().numpy()
            weighted_samples = (torch.exp(log_norm_weights).reshape((-1,1))*values.float()).detach().numpy()
            print('covariance: ', np.cov(values.float().detach().numpy(),rowvar=False, aweights=weights))
            print('posterior mean:', weighted_samples.mean(axis=0))

        plots(values, log_weights, j)

