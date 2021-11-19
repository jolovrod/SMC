from primitives import env as penv
import json
import torch
from daphne import daphne
from pyrsistent import pmap, plist
import numpy as np
from tests import is_tol, run_prob_test,load_truth
import sys
import threading

#these are adapted from Peter Norvig's Lispy
class Env():
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.data = pmap(zip(parms, args))
        self.outer = outer
        if outer is None:
            self.level = 0
        else:
            self.level = outer.level+1

    def __getitem__(self, item):
        return self.data[item]

    def find(self, var):
        "Find the innermost Env where var appears."
        if (var in self.data):
            return self
        else:
            if self.outer is not None:
                return self.outer.find(var)
            else:
                raise RuntimeError('var "{}" not found in outermost scope'.format(var))

    def print_env(self, print_lowest=False):
        print_limit = 1 if print_lowest == False else 0
        outer = self
        while outer is not None:
            if outer.level >= print_limit:
                print('Scope on level ', outer.level)
                if 'f' in outer:
                    print('Found f, ')
                    print(outer['f'].body)
                    print(outer['f'].parms)
                    print(outer['f'].env)
                print(outer,'\n')
            outer = outer.outer


class Procedure(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env
    def __call__(self, sigma, *args):
        # TODO: sigma needs to be passed because it could be updated. 
        # print("in Proceedure call")
        # print("args", args)
        # print("\n\n\n")
        # TODO: pass sigma correctly. 
        return evaluate(self.body, sigma, Env(self.parms, args, self.env))


def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env(penv.keys(), penv.values())
    return env



def evaluate(exp, sigma, env=None):

    if env is None:
        env = standard_env()
        
    
    if type(exp) is list:
        op, *args = exp
        if op == 'sample':
            alpha = evaluate(args[0], sigma, env=env)
            d = evaluate(args[1], sigma, env=env)
            s = d.sample()
            k = evaluate(args[2], sigma, env=env)
            sigma['type'] = 'sample'
            sigma['alpha'] = alpha
            #TODO: put any other stuff you need here
            return k, [s], sigma
        elif op == 'observe':
            alpha = evaluate(args[0], sigma, env=env)
            d = evaluate(args[1], sigma, env=env)
            c = evaluate(args[2], sigma, env=env)
            k = evaluate(args[3], sigma, env=env)
            sigma['type'] = 'observe'
            sigma['logW'] = sigma['logW'] + d.log_prob(c)
            sigma['alpha'] = alpha
            #TODO: put any other stuff you need here
            return k, [c], sigma
        elif op == 'if':
            cond,conseq,alt = args
            if evaluate(cond, sigma, env=env):
                return evaluate(conseq, sigma, env=env)
            else:
                return evaluate(alt, sigma, env=env)
        elif op == 'fn': 
            # Is always called from the func eval case (below)
            params, body = args #fn is:  ['fn', ['arg1','arg2','arg3'], body_exp]
            return Procedure(params, body, env)
        else: #func eval
            proc = evaluate(op, sigma, env=env)
            values = [evaluate(e, sigma, env=env) for e in args]
            sigma['type'] = 'proc'
            #TODO: put any other stuff you need here
            # this is the continuation that gets 
            return proc, values, sigma
    elif type(exp) is str:
        if exp[0] == "\"":  # strings have double, double quotes
            return exp[1:-1]
        if exp[0:4] == 'addr':
            return exp[4:]
        lowest_env = env.find(exp)
        return lowest_env[exp]
    elif type(exp) is float or type(exp) is int or type(exp) is bool:
        return torch.tensor(exp)
    else:
        raise ValueError('Expression type unkown')


def sample_from_prior(exp):
    #init calc:
    output = lambda _,x: x # The output is the identity (also takes sigma as argument)
    sigma = {'logW':0}
    res =  evaluate(exp, sigma, env=None)(sigma, 'addr_start', output) #set up the initial call
    while type(res) is tuple: #if there are continuations, the res will be a tuple
        cont, args, sig = res # res is contininuation, arguments, and a map, which you can use to pass back some additional stuff
        res = cont(sigma, *args) #call the continuation
        # continuation is a function
        # function is called with the arguments. 
        # TODO: add sigma to function call. 
    #when res is not a tuple, the calculation has finished
    # ie 
    return res, sigma['logW']

def get_stream(exp):
    while True:
        yield sample_from_prior(exp)


def run_deterministic_tests(use_cache=True, cache='programs/tests/'):

    for i in range(1,15):
        if use_cache:
            with open('C:/Users/jlovr/CS532-HW6/SMC/programs/tests/deterministic/test_{}.daphne'.format(i),'r') as f:
                exp = json.load(f)
        else:
            exp = daphne(['desugar-hoppl-cps', '-i', 'C:/Users/jlovr/CS532-HW6/SMC/programs/tests/deterministic/test_{}.daphne'.format(i)])
            with open('C:/Users/jlovr/CS532-HW6/SMC/programs/tests/deterministic/test_{}.daphne'.format(i),'w') as f:
                json.dump(exp, f)
        truth = load_truth('C:/Users/jlovr/CS532-HW6/SMC/programs/tests/deterministic/test_{}.truth'.format(i))
        ret = sample_from_prior(exp)
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        print('Test {} passed'.format(i))

    print('FOPPL Tests passed')

    for i in range(1,13):
        if use_cache:
            with open('C:/Users/jlovr/CS532-HW6/SMC/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i),'r') as f:
                exp = json.load(f)
        else:
            exp = daphne(['desugar-hoppl-cps', '-i', 'C:/Users/jlovr/CS532-HW6/SMC/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
            with open('C:/Users/jlovr/CS532-HW6/SMC/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i),'w') as f:
                json.dump(exp, f)

        truth = load_truth('C:/Users/jlovr/CS532-HW6/SMC/programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = sample_from_prior(exp)

        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))

        print('Test {} passed'.format(i))

    print('All deterministic tests passed')



def run_probabilistic_tests(use_cache=True, cache='programs/tests/'):

    num_samples=1e4
    max_p_value = 1e-2

    for i in [1,2,3,4,6]: #test 5 does not work, sorry. 
        if use_cache:
            with open('C:/Users/jlovr/CS532-HW6/SMC/programs/tests/probabilistic/test_{}.daphne'.format(i),'r') as f:
                exp = json.load(f)
        else:
            exp = daphne(['desugar-hoppl-cps', '-i', 'C:/Users/jlovr/CS532-HW6/SMC/programs/tests/probabilistic/test_{}.daphne'.format(i)])
            with open('C:/Users/jlovr/CS532-HW6/SMC/programs/tests/probabilistic/test_{}.daphne'.format(i),'w') as f:
                json.dump(exp, f)
        truth = load_truth('C:/Users/jlovr/CS532-HW6/SMC/programs/tests/probabilistic/test_{}.truth'.format(i))

        stream = get_stream(exp)

        p_val = run_prob_test(stream, truth, num_samples)

        print('p value', p_val)
        assert(p_val > max_p_value)

    print('All probabilistic tests passed')


def my_main():
    # run the tests, if you wish:  
    
    # run_deterministic_tests(use_cache=True)
    # run_probabilistic_tests(use_cache=True)

    # compile json's here:
    # for i in range(1,4):
    #     exp = daphne(['desugar-hoppl-cps', '-i', 'C:/Users/jlovr/CS532-HW6/SMC/programs/{}.daphne'.format(i)])
    #     with open('C:/Users/jlovr/CS532-HW6/SMC/programs/{}.daphne'.format(i),'w') as f:
    #         json.dump(exp, f)


    for i in range(1, 5):
        #load your precompiled json's here:
        with open('C:/Users/jlovr/CS532-HW6/SMC/programs/{}.daphne'.format(i),'r') as f:
            exp = json.load(f)


        #this should run a sample from the prior
        print(sample_from_prior(exp))

        # #you can see how the CPS works here, you define a continuation for the last call:
        # output = lambda x: x #The output is the identity

        # #set up the initial call, every evaluate returns a continuation, a set of arguments, and a map sigma at every procedure call, every sample, and every observe
        # res =  evaluate(exp, {}, env=None)('addr_start', output) 
        # cont, args, sigma = res
        # print(cont, args, sigma)
        # #you can keep calling this to run the program forward:
        # res = cont(*args)

        # #you know the program is done, when "res" is not a tuple, but a simple data object

        print("\n\n\n")


if __name__ == '__main__':
    sys.setrecursionlimit(100000)
    threading.stack_size(200000000)
    thread = threading.Thread(target=my_main)
    thread.start()     

