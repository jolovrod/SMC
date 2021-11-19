import torch
import torch.distributions as tdist
import distributions as dist

def Normal(sigma, alpha, loc, scale, k):
    return k(sigma, dist.Normal(loc.float(), scale.float()))

def Bernoulli(sigma, alpha, probs, k):
    return k(sigma, dist.Bernoulli(probs))

def Categorical(sigma, alpha, probs, k):
    return k(sigma, dist.Categorical(probs=probs))

def Dirichlet(sigma, alpha, concentration, k):
    return k(sigma, dist.Dirichlet(concentration))

def Gamma(sigma, alpha, concentration, rate, k):
    return k(sigma, dist.Gamma(concentration, rate))

def Beta(sigma, alpha, arg0, arg1, k):
    return k(sigma, tdist.Beta(arg0, arg1))

def Exponential(sigma, alpha, rate, k):
    return k(sigma, tdist.Exponential(rate))

def Uniform(sigma, alpha, hi, lo, k):
    return k(sigma, tdist.Uniform(hi, lo))



def sqrt(sigma, alpha, arg, k):
    return k(sigma, torch.sqrt(arg.float()))

def exp(sigma, alpha, arg, k):
    return k(sigma, torch.exp(arg.float()))

def log(sigma, alpha, arg, k):
    return k(sigma, torch.log(arg.float()))

def tanh(sigma, alpha, arg, k):
    return k(sigma, torch.tanh(arg.float()))

def add(sigma, alpha, a, b, k):
    return k(sigma, torch.add(a, b))

def mul(sigma, alpha, a, b, k):
    return k(sigma, torch.mul(a,b))

def div(sigma, alpha, a, b, k):
    return k(sigma, torch.div(a,b))

def sub(sigma, alpha, a, b, k):
    return k(sigma, torch.sub(a,b))

def gt(sigma, alpha, a, b, k):
    return k(sigma, torch.gt(a, b))

def lt(sigma, alpha, a, b, k):
    return k(sigma, torch.lt(a,b))


def vector(sigma, alpha, *args):
    k = args[-1]
    args = args[:-1]
    if len(args) == 0:
        return k(sigma, torch.tensor([]))
    elif type(args[0]) is torch.Tensor:
        try:
            output = torch.stack(args) #stack works for 1D, but also ND
        except Exception:
            output = list(args) #NOTE:  that these are NOT persistent
        return k(sigma, output)
    else:
        return k(sigma, list(args)) #this is for probability distributions


def hashmap(sigma, alpha, *args):
    k = args[-1]
    args = args[:-1]
    new_map = {} #NOTE: also not persistent
    for i in range(len(args)//2):
        if type(args[2*i]) is torch.Tensor:
            key = args[2*i].item()
        elif type(args[2*i]) is str:
            key = args[2*i]
        else:
            raise ValueError('Unkown key type, ', args[2*i])
        new_map[key] = args[2*i+1]
    return k(sigma, new_map)

def first(sigma, alpha, sequence, k):
    return k(sigma, sequence[0])

def second(sigma, alpha, sequence, k):
    return k(sigma, sequence[1])

def rest(sigma, alpha, sequence, k):
    return k(sigma, sequence[1:])


def last(sigma, alpha, sequence, k):
    return k(sigma, sequence[-1])

def get(sigma, alpha, data, element, k):
    if type(data) is dict:
        if type(element) is torch.Tensor:
            key = element.item()
        elif type(element) is str:
            key = element
        return k(sigma, data[key])
    else:
        return k(sigma, data[int(element)])

def put(sigma, alpha, data, element, value, k): #vector, index, value
    if type(data) is dict:
        newhashmap = data.copy() #NOTE: right now we're copying
        if type(element) is torch.Tensor:
            key = element.item()
        elif type(element) is str:
            key = element
        newhashmap[key] = value
        return k(sigma, newhashmap)
    else:
        newvector = data.clone() 
        newvector[int(element)] = value
        return k(sigma, newvector)

def remove(sigma, alpha, data, element, k):
    if type(data) is dict:
        newhashmap = data.copy()
        if type(element) is torch.Tensor:
            key = element.item()
        elif type(element) is str:
            key = element
        _ = newhashmap.pop(key)        
        return k(sigma, newhashmap)
    else:
        idx = int(element)
        newvector = torch.cat([data[0:idx],data[idx+1:]],dim=0)
        return k(sigma, newvector)
    
def append(sigma, alpha, data, value, k):
    return k(sigma, torch.cat([data,torch.tensor([value])], dim=0))

def is_empty(sigma, alpha, arg, k):
    return k(sigma, len(arg) == 0)

def peek(sigma, alpha, sequence, k): #NOTE: only defined for vector
    return k(sigma, sequence[0])

def conj(sigma, alpha, sequence, element, k):
    if type(sequence) is torch.Tensor:
        return k(sigma, torch.cat((element.reshape(1), sequence)))
    elif type(sequence) is list:
        return k(sigma, [element] + sequence)


def mat_transpose(sigma, alpha, arg, k):
    return k(sigma, torch.transpose(arg, 1, 0))

def mat_mul(sigma, alpha, arg0, arg1, k):
    return k(sigma, torch.matmul(arg0,arg1))
    
def mat_repmat(sigma, alpha, mat, dim, n, k):
    shape = [1,1]
    shape[int(dim)] = int(n)
    return k(sigma, mat*torch.ones(tuple(shape)))


def push_addr(sigma, alpha, value, k):
    # print("in push_addr", alpha, value, k)
    # print("args", alpha, value, k)
    # print('pushing ', value, ' onto ', alpha)
    return k(sigma, alpha + '_' + value)


env = {
       #distr
           'normal': Normal,
           'beta': Beta,
           'discrete': Categorical,
           'dirichlet': Dirichlet,
           'exponential': Exponential,
           'uniform-continuous': Uniform,
           'gamma': Gamma,
           'flip': Bernoulli,
           
           # #math
           'sqrt': sqrt,
           'exp': exp,
           'log': log,
           'mat-tanh' : tanh,
           'mat-add' : add,
           'mat-mul' : mat_mul,
           'mat-transpose' : mat_transpose,
           'mat-repmat' : mat_repmat,
           '+': add,
           '-': sub,
           '*': mul,
           '/': div,
           
           # #
           '<' : lt,
           '>' : gt,
           # '<=' : torch.le,
           # '>=' : torch.ge,
           # '=' : torch.eq,
           # '!=' : torch.ne,
           # 'and' : torch.logical_and,
           # 'or' : torch.logical_or,

           'vector': vector,
           'hash-map' : hashmap,
           'get': get,
           'put': put,
           'append': append,
           'first': first,
           'second': second,
           'rest': rest,
           'last': last,
           'empty?': is_empty,
           'conj' : conj,
           'peek' : peek,

           'push-address' : push_addr,
           }


