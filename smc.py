from numpy.core.fromnumeric import argmax, argmin
from evaluator import evaluate
import torch
import numpy as np
import json
import sys
import matplotlib.pyplot as plt

from primitives import log
from plots import plots



def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(sigma, *args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(sigma, *args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res

def resample_particles(particles, log_weights):
    L = len(log_weights)
    log_ws = torch.FloatTensor(log_weights)
    discrete_dist = torch.distributions.categorical.Categorical(logits=log_ws)
    new_particles = []
    for _ in range(L):
        k = discrete_dist.sample()
        new_particles.append(particles[k])

    logZ = torch.logsumexp(log_ws,0) - torch.log(torch.tensor(log_ws.shape[0],dtype=float))

    return logZ, new_particles


def SMC(n_particles, exp):
    particles = []
    weights = []
    logZs = []
    sigma = {'logW':0}
    output = lambda _, x: x

    for i in range(n_particles):
        cont, args, sigma = evaluate(exp, sigma, env=None)(sigma, 'addr_start', output)
        logW = 0.
        weights.append(logW)
        res = cont, args, {'logW':weights[i]}
        particles.append(res)

    done = False
    smc_cnter = 0
    while not done:
        for i in range(n_particles): #Even though this can be parallelized, we run it serially
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res[0]
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''    
                else:
                    if not done:        # is the /first/ particle i=0 done?
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:  # res[2] == 'observe'
                cont, args, sigma = res
                weights[i] = res[2]['logW'].clone().detach()        # get weights
                particles[i] = cont, args, {'logW':weights[i]}      # get continuation

                if i == 0:
                    address = sigma['alpha']
                try:
                    assert(sigma['alpha'] == address)
                except:
                    raise AssertionError('particle address error')

        if not done:
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)
            
        smc_cnter += 1  # number of continuations/observes completed. 

    if logZs == []:
        return 0, particles
    else:
        return logZs[-1], particles


if __name__ == '__main__':

    for i in range(1,5):
        with open('C:/Users/jlovr/CS532-HW6/SMC/programs/{}.daphne'.format(i),'r') as f:
            exp = json.load(f)
        
        logZ_list = []

        for n_particles in [1,10,100,1000,10000,100000]:
            
            logZ, particles = SMC(n_particles, exp)

            values = torch.stack(particles)
            
            #### presentation of the results

            print("Program:", i, "  number of particles:", n_particles)

            print('posterior mean:', values.float().detach().numpy().mean(axis=0))
            if n_particles > 1:
                if i == 3:
                    print('variance: ', np.diag(np.cov(values.float().detach().numpy(),rowvar=False)))  
                else:
                    print('variance: ', np.cov(values.float().detach().numpy(),rowvar=False))    
            weights = np.ones(n_particles)
                
            print("logZ:", np.array(logZ, dtype=float))
            logZ_list.append(logZ)
            
            plots(particles, i, n_particles)
        
        plt.figure(figsize=(8,4))
        plt.xlabel("$\log_{10} (n)$")
        plt.ylabel("logZ")
        plt.title("Marginal log-probability estimate returned by SMC for program " + str(i))

        plt.plot(logZ_list)
        figstr = "logZ_estimates/program_"+str(i)
        plt.savefig(figstr)

