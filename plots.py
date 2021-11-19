import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plots(samples, i, n):
    
    num_samples = len(samples)

    if i in [1,2,4]:

        samples = np.array(samples, dtype=float)
        
        plt.figure(figsize=(5,4))

        if i == 1:
            plt.xlabel("n")
        else:
            plt.xlabel("mu")

        plt.ylabel("Frequency")
        plt.title("Distributions obtained program " + str(i) + " with " + str(num_samples) + " particles")

        sns.histplot(samples,kde=False, bins=50)

        figstr = "histograms/n_"+str(num_samples)+"_program_"+str(i)
        plt.savefig(figstr)

    elif i==3:


        # this is ugly. Fix it. 
        for n in range(num_samples):
            samples[n] = np.array(samples[n], dtype=int)

        variables = np.array(samples,dtype=object).T.tolist()

        for d in range(len(variables)):
            counts = [0,0,0]
            for element in variables[d]:
                counts[element] += 1
            plt.figure(figsize=(5,4))
            plt.bar([0,1,2],counts)
            plt.xlabel("observations["+str(d)+"]")
            plt.ylabel("frequency")
            figstr = "histograms/n_"+str(num_samples)+"_program_"+str(i)+"_var_"+str(d)
            state_dist = [x/num_samples for x in counts]
            print("state distribution dim", d,":", state_dist)
            plt.savefig(figstr)

        print("\n")

    # plt.show()
    print("\n\n\n")
    plt.close('all')
