import subprocess
from subprocess import Popen as cmd
import os
import shutil
import time

stencils_2d = ["2d9pt_star", "2d81pt_box", "2d121pt_box", "2d5pt_arbitrary_shape"]
# stencils_3d = ["3d13pt_star", "3d27pt_box", "3d125pt_box", "3d7pt_arbitrary_shape"]
stencils_3d = ["3d27pt_box", "3d125pt_box", "3d7pt_arbitrary_shape"]

tile_size_2d = {0:[[4], [8], [16], [32]], 1:[8, 16, 32, 64]}
tile_size_3d = {0:[[2, 2], [2, 4], [2, 8], [4, 8]], 1:[8, 16, 32, 64]}

dst_dir_prefix = "examples/"
utils_path = "../../utils"
makefileSerial_path = "makefileSerial"

def run_heatmap(stencils, tile_size):
    f = open("heatmap_result", 'a')
    counter = 0
    for case in stencils:
        print("heat_map: "+case)
        for i in range(len(tile_size[0])):
            j = 0
            while j < len(tile_size[1]):
                tile_size_string = ""
                for k in range((len(tile_size[0][i]))):
                    tile_size_string += "_"+str(tile_size[0][i][k])
                tile_size_string += "_"+str(tile_size[1][j])
                src_dir = "heat_map/c/stencil_"+case+tile_size_string+"/"
                dst_dir = dst_dir_prefix+case+"/heat_map"+tile_size_string+"/"
                if os.path.exists(dst_dir) is True:
                    shutil.rmtree(dst_dir)    
                shutil.copytree(src_dir, dst_dir)
                shutil.copy(makefileSerial_path, dst_dir)
                os.symlink(utils_path, dst_dir+"/utils")
                cmd_string = "cd "+dst_dir+";"
                cmd_string += "make -f makefileSerial;"
                cmd_string += "make -f makefileSerial run"
                print("running "+case+"/heat_map"+tile_size_string+"...")
                print("start at: "+time.asctime(time.localtime(time.time())))
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
                        output_string = "retry failed for three times"
                        counter = 0
                    
                output_string_list = output_string.split("\n")
                for line in output_string_list:
                    print(line)
                    if (line.find("Time:") != -1 or line == "retry failed for three times"):
                        f.write(case+"_heatmap"+tile_size_string+" "+line+"\n")
                        f.flush()
                j+=1
    f.close()

def main():
    # run_heatmap(stencils_2d, tile_size_2d)
    run_heatmap(stencils_3d, tile_size_3d)

if __name__ == "__main__":
    main()
