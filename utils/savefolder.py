import subprocess

def createWorkSpace(path):
    savingPath = path+"savings/"
    recordPath = path+"records/"
    picPatch = path+"pic/"
    cmd = ['mkdir', '-p', savingPath]
    subprocess.check_call(cmd)
    cmd = ['mkdir', '-p', recordPath]
    subprocess.check_call(cmd)
    cmd = ['mkdir', '-p', picPatch]
    subprocess.check_call(cmd)