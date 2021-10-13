from os import system as cmd
import os
import shutil
import time

stencils_2d = ["2d9pt_star", "2d81pt_box", "2d121pt_box", "2d5pt_arbitrary_shape"]
stencils_3d = ["3d13pt_star", "3d27pt_box", "3d125pt_box", "3d7pt_arbitrary_shape"]

tile_size_2d = {0:[[4], [8], [16], [32]], 1:[8, 16, 32, 64]}
tile_size_3d = {0:[[2, 2], [2, 4], [2, 8], [4, 8]], 1:[8, 16, 32, 64]}

heat_map_dir_prefix = "heat_map/"
if os.path.exists(heat_map_dir_prefix) is True:
    shutil.rmtree(heat_map_dir_prefix)
os.mkdir(heat_map_dir_prefix)
os.mkdir(heat_map_dir_prefix+"dsl")
os.mkdir(heat_map_dir_prefix+"sw")
os.mkdir(heat_map_dir_prefix+"c")

def heat_map(stencils, tile_size):
    cmd("rm -f *.c *.sw")
    for case in stencils:
        print("heat_map: "+str(case))
        path_src = "../examples/"+str(case)+"/stencil_"+str(case)+".dsl"
        for i in range(len(tile_size[0])):
            for j in range(len(tile_size[1])):
                tile_size_string = ""
                for k in range((len(tile_size[0][i]))):
                    tile_size_string += "_"+str(tile_size[0][i][k])
                tile_size_string += "_"+str(tile_size[1][j])
                dsl_path_out = heat_map_dir_prefix+"dsl/stencil_"+str(case)+tile_size_string+".dsl"
                with open(path_src, 'r', encoding='utf-8') as f:
                    output_file = open(dsl_path_out, 'w')
                    for line in f.readlines():
                        output_string = ''
                        if (line.find("tile") != -1):
                            output_string = "\t\ttile("
                            for k in range((len(tile_size[0][i]))):
                                output_string += str(tile_size[0][i][k]) + ", "
                            output_string += str(tile_size[1][j]) + ")\n"
                        elif (line.find("mpiTile") == -1 and line.find("mpiHalo") == -1):
                            output_string = line
                        output_file.write(output_string)
                    output_file.close()
                # 使用stenCC编译器编译对应的dsl
                sw_path_out = heat_map_dir_prefix+"sw/"
                c_path_out_dir = heat_map_dir_prefix + "c/stencil_"+str(case)+tile_size_string
                os.mkdir(c_path_out_dir)
                cmd("./bin/stenCC "+dsl_path_out+" --emit=sw > stencil_"+str(case)+tile_size_string+".sw 2>&1")
                cmd("python3 translate.py stencil_"+str(case)+tile_size_string+".sw")
                shutil.move("stencil_"+str(case)+tile_size_string+".sw", sw_path_out)
                cmd("mv *.c " + c_path_out_dir)
                # 复制对应的driver, 并去掉结果验证部分
                driver_path_src = "../examples/"+str(case)+"/stencil_"+str(case)+"_driver.serial.c"
                driver_path_out = c_path_out_dir+"/stencil_"+str(case)+"_driver.serial.c"
                with open(driver_path_src, 'r', encoding='utf-8') as f:
                    output_file = open(driver_path_out, 'w')
                    for line in f.readlines():
                        output_string = ''
                        if (line.find("mpe_verify();") == -1):
                            output_string = line
                        output_file.write(output_string)
                    output_file.close()

def main():
    heat_map(stencils_2d, tile_size_2d)
    heat_map(stencils_3d, tile_size_3d)

    
if __name__ == "__main__":
    main()