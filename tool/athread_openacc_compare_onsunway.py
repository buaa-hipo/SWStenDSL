import subprocess
from subprocess import Popen as cmd
import os
import shutil
import time

stencils_2d = ["2d9pt_star", "2d81pt_box", "2d121pt_box", "2d5pt_arbitrary_shape", "2d5pt_nested"]
stencils_3d = ["3d13pt_star", "3d27pt_box", "3d125pt_box", "3d7pt_arbitrary_shape", "3d7pt9pt_nested"]

dst_dir_prefix = "examples"
utils_path = "../../utils"
makefileSerial_path = "makefileSerial"
makefileOpenACC_path = "makefileOpenACC"

def run_athread_compare(stencils):
    f = open("athread_result", 'a')
    counter = 0
    i = 0
    while i < len(stencils):
        print("athread case: "+ stencils[i])
        src_dir = "athread_compare/stencil_"+stencils[i]+"/"
        dst_dir = dst_dir_prefix+stencils[i]+"/"+"athread_compare/"
        if os.path.exists(dst_dir) is True:
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        shutil.copy(makefileSerial_path, dst_dir)
        os.symlink(utils_path, dst_dir+"utils")
        cmd_string = "cd "+dst_dir+";"
        cmd_string += "make -f makefileSerial"
        cmd_string += "make -f makefileSerial run"
        print("running athread compare: "+stencils[i])
        print("start at: " + time.asctime(time.localtime(time.time())))
        process = cmd(cmd_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, err = process.communicate()
        output_string = output.decode(encoding='utf-8', errors='replace')
        if (process.returncode != 0):
            print(output_string)
            print("someting error, resubmit the job")
            counter += 1
            if (counter <= 3):
                continue
            else:
                output_string = "retry failed for 3 times"
                counter = 0
        
        output_string_list = output_string.split("\n")
        for line in output_string_list:
            print(line)
            if (line.find("Time:") != -1 or line == "retry failed for 3 times"):
                f.write(stencils[i]+" "+line+"\n")
                f.flush()
        i+=1
    f.close()

def run_openacc_compare(stencils):
    f = open("openacc_result", 'a')
    counter = 0
    i = 0
    while i < len(stencils):
        print("openacc case: "+ stencils[i])
        src_dir = "openacc_compare/stencil_"+stencils[i]+"/"
        dst_dir = dst_dir_prefix+stencils[i]+"/"+"openacc_compare/"
        if os.path.exists(dst_dir) is True:
            shutil.rmtree(dst_dir)
        shutil.copytree(src_dir, dst_dir)
        shutil.copy(makefileSerial_path, dst_dir)
        cmd_string = "cd "+dst_dir+";"
        cmd_string += "make -f makefileOpenACC"
        cmd_string += "make -f makefileOpenACC run"
        print("running openacc compare: "+stencils[i])
        print("start at: " + time.asctime(time.localtime(time.time())))
        process = cmd(cmd_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, err = process.communicate()
        output_string = output.decode(encoding='utf-8', errors='replace')
        if (process.returncode != 0):
            print(output_string)
            print("someting error, resubmit the job")
            counter += 1
            if (counter <= 3):
                continue
            else:
                output_string = "retry failed for 3 times"
                counter = 0
        
        output_string_list = output_string.split("\n")
        for line in output_string_list:
            print(line)
            if (line.find("Time:") != -1 or line == "retry failed for 3 times"):
                f.write(stencils[i]+" "+line+"\n")
                f.flush()
        i+=1
    f.close()

def main():
    run_athread_compare(stencils_2d)
    run_athread_compare(stencils_3d)
    run_openacc_compare(stencils_2d)
    run_openacc_compare(stencils_3d)

if __name__ == '__main__':
    main()