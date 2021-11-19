import json
import subprocess

def daphne(args, cwd='C:/Users/jlovr/CS532-HW3/Inference-algorithms/daphne'):
    proc = subprocess.run(['lein','run','-f','json'] + args,
                          capture_output=True, cwd=cwd, shell=True)
    if(proc.returncode != 0):
        raise Exception(proc.stdout.decode() + proc.stderr.decode())
    return json.loads(proc.stdout)

    # lein run -f json desugar -i C:/Users/jlovr/CS532-HW2/programs/1.daphne