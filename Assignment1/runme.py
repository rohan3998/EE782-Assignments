import subprocess
lr=["0.0001","0.001","0.01","0.1"]
split=["1","0.8","0.6","0.4","0.2"]
for j in split:
    for i in lr:

        subprocess.call(["python","code_for_submission.py","0.0","0.0",i,j,"adam"])
