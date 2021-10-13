import subprocess
from subprocess import Popen as cmd
import os
import shutil
import time

stencils_2d = ["2d9pt_star", "2d81pt_box", "2d121pt_box", "2d5pt_arbitrary_shape", "2d5pt_nested"]
stencils_3d = ["3d13pt_star", "3d27pt_box", "3d125pt_box", "3d7pt_arbitrary_shape", "3d7pt9pt_nested"]

mpi_tile_size_2d = [[16, 8], [16, 16], [32, 16], [32, 32]]
mpi_tile_size_3d = [[8, 4, 4], [8, 4, 8], [8, 8, 8], [16, 8, 8]]

dst_dir_prefix = "examples/"
utils_path = "../../utils"
makefileMPI_path = "makefileMPI"

def run_scalability(stencils, mpi_tile_size, scala):
    f = open(scala+"_result", 'a')
    counter = 0
    for case in stencils:
        print(scala+" case:" + case)
        i = 0
        while i < len(mpi_tile_size):
            mpi_tile_string = ''
            node_num = 1
            for j in range(len(mpi_tile_size[i])):
                mpi_tile_string += "_"+str(mpi_tile_size[i][j])
                node_num *= mpi_tile_size[i][j]
            src_dir = scala+"_scalability/c/stencil_"+case+mpi_tile_string+"/"
            dst_dir = dst_dir_prefix+case+"/"+scala+mpi_tile_string+"/"
            if os.path.exists(dst_dir) is True:
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            shutil.copy(makefileMPI_path, dst_dir)
            os.symlink(utils_path, dst_dir+"/utils")
            cmd_string = "cd "+dst_dir+";"
            cmd_string += "make -f makefileMPI;"
            cmd_string += "bsub -o output.log -I -b -q q_sw_share -n " + str(node_num) + " -cgsp 64 -share_size 2048 ./MPIExe"
            print("running "+case+"/"+scala+mpi_tile_string+"...")
            print("start at: " +time.asctime(time.localtime(time.time())))
            process = cmd(cmd_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            output, err = process.communicate()
            output_string = output.decode(encoding='utf-8')
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
                    f.write(case+"_"+scala+mpi_tile_string+" "+line+"\n")
                    f.flush()
            i+=1

    f.close()

def main():
    run_scalability(stencils_2d, mpi_tile_size_2d, "weak")
    run_scalability(stencils_3d, mpi_tile_size_3d, "weak")
    run_scalability(stencils_2d, mpi_tile_size_2d, "strong")
    run_scalability(stencils_3d, mpi_tile_size_3d, "strong")

if __name__ == "__main__":
    main()
