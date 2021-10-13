from os import system as cmd
import os
import shutil
import time

stencils_2d = ["2d9pt_star", "2d81pt_box", "2d121pt_box", "2d5pt_arbitrary_shape", "2d5pt_nested"]
stencils_3d = ["3d13pt_star", "3d27pt_box", "3d125pt_box", "3d7pt_arbitrary_shape", "3d7pt9pt_nested"]

halo_size_2d = {"2d9pt_star":{0:[[2, 2], [2, 2]]}, 
                "2d81pt_box":{0:[[4, 4], [4, 4]]}, 
                "2d121pt_box":{0:[[5, 5], [5, 5]]}, 
                "2d5pt_arbitrary_shape":{0:[[2, 0], [0, 2]]},
                "2d5pt_nested":{0:[[1, 1], [1, 1]], 1:[[17, 17], [17, 17]]}
                }
halo_size_3d = {"3d13pt_star":{0:[[2, 2], [2, 2], [2, 2]]}, 
                "3d27pt_box":{0:[[1, 1], [1, 1], [1, 1]]}, 
                "3d125pt_box":{0:[[2, 2], [2, 2], [2, 2]]}, 
                "3d7pt_arbitrary_shape":{0:[[0, 2], [0, 2], [0, 2]]},
                "3d7pt9pt_nested":{0:[[1, 1], [1, 1], [1, 1]], 1:[[5, 5], [5, 5], [5, 5]]}
                }

mpi_tile_size_2d = [[16, 8], [16, 16], [32, 16], [32, 32]]
mpi_tile_size_3d = [[8, 4, 4], [8, 4, 8], [8, 8, 8], [16, 8, 8]]

# for strong scalability
sub_grid_2d = [[4096, 4096], [4096, 2048], [2048, 2048], [2048, 1024]]
sub_grid_3d = [[256, 256, 256], [256, 256, 128], [256, 128, 128], [128, 128, 128]]


# 弱拓展性
weak_target_dir_path_prefix = "weak_scalability/"
if os.path.exists(weak_target_dir_path_prefix) is True:
    shutil.rmtree(weak_target_dir_path_prefix)
os.mkdir(weak_target_dir_path_prefix)
os.mkdir(weak_target_dir_path_prefix+"dsl")
os.mkdir(weak_target_dir_path_prefix+"sw")
os.mkdir(weak_target_dir_path_prefix+"c")

def test_weak_scalability(stencils, mpi_tile_size):
    cmd("rm -f *.c *.sw")
    for case in stencils:
        print("weak cases: "+str(case))
        path_src = "../examples/"+str(case)+"/stencil_"+str(case)+".dsl"
        for i in range(len(mpi_tile_size)):
            # 生成指定stencil指定mpiTile的dsl
            mpi_tile_string = ""
            for j in range(len(mpi_tile_size[i])):
                mpi_tile_string += "_"+str(mpi_tile_size[i][j])
            dsl_path_out = weak_target_dir_path_prefix+"dsl/stencil_"+str(case)+mpi_tile_string+".dsl"

            with open(path_src, 'r', encoding='utf-8') as f:
                output_file = open(dsl_path_out, 'w')
                for line in f.readlines():
                    output_string = ''
                    if (line.find("mpiTile") != -1):
                        output_string = "\tmpiTile("
                        for j in range(len(mpi_tile_size[i])):
                            output_string += str(mpi_tile_size[i][j])
                            if (j != len(mpi_tile_size[i])-1):
                                output_string += ", "
                        output_string += ")\n"
                    else:
                        output_string = line
                    output_file.write(output_string)
                output_file.close()
            # 使用stenCC编译器编译对应的dsl
            sw_path_out = weak_target_dir_path_prefix+"sw/"
            c_path_out_dir = weak_target_dir_path_prefix+"c/stencil_"+str(case)+mpi_tile_string
            os.mkdir(c_path_out_dir)
            cmd("./bin/stenCC "+dsl_path_out+" --emit=sw > stencil_"+str(case)+mpi_tile_string+".sw 2>&1")
            cmd("python3 translate.py stencil_"+str(case)+mpi_tile_string+".sw")
            shutil.move("stencil_"+str(case)+mpi_tile_string+".sw", sw_path_out)
            cmd("mv *.c " + c_path_out_dir)
            # 复制对应的driver, 并去除结果输出
            driver_path_src = "../examples/"+str(case)+"/stencil_"+str(case)+"_driver.mpi.c"
            driver_path_out = c_path_out_dir+"/stencil_"+str(case)+"_driver.mpi.c"
            with open(driver_path_src, 'r', encoding='utf-8') as f:
                output_file = open(driver_path_out, 'w')
                for line in f.readlines():
                    output_string = ''
                    if(line.find("swsten_store_data_to_file") == -1):
                        output_string = line
                    output_file.write(output_string)
                output_file.close()


# 强拓展性
strong_target_dir_path_prefix = "strong_scalability/"
if os.path.exists(strong_target_dir_path_prefix) is True:
    shutil.rmtree(strong_target_dir_path_prefix)
os.mkdir(strong_target_dir_path_prefix)
os.mkdir(strong_target_dir_path_prefix+"dsl")
os.mkdir(strong_target_dir_path_prefix+"sw")
os.mkdir(strong_target_dir_path_prefix+"c")


def test_strong_scalability(stencils, mpi_tile_size, sub_grid, halo_size):
    cmd("rm -f *.c *.sw")
    for case in stencils:
        print("strong case: " + str(case))
        path_src = "../examples/"+str(case)+"/stencil_"+str(case)+".dsl"
        for i in range(len(mpi_tile_size)):
            # 生成指定stencil指定mpiTile, 指定domainSize的dsl
            mpi_tile_string = ""
            for j in range(len(mpi_tile_size[i])):
                mpi_tile_string += "_"+str(mpi_tile_size[i][j])
            dsl_path_out = strong_target_dir_path_prefix+"dsl/stencil_"+str(case)+mpi_tile_string+".dsl"
            with open(path_src, 'r', encoding='utf-8') as f:
                output_file = open(dsl_path_out, 'w')
                line_index=0
                kernel_counter = 0
                for line in f.readlines():
                    line_index+=1
                    output_string = ''
                    if (line_index==1):
                        left_squareBr = line.find("[");
                        output_string = line[0:left_squareBr]
                        for k in range(len(sub_grid[i])):
                            dimSize = sub_grid[i][k]
                            dimSize += halo_size[case][0][k][0]
                            dimSize += halo_size[case][0][k][1]
                            output_string += "[" + str(dimSize) + "]"
                        output_string += ") {\n"
                    elif (line.find("mpiTile") != -1):
                        output_string = "\tmpiTile("
                        for j in range(len(mpi_tile_size[i])):
                            output_string += str(mpi_tile_size[i][j])
                            if (j != len(mpi_tile_size[i])-1):
                                output_string += ", "
                        output_string += ")\n"
                    elif (line.find("domain") != -1):
                        output_string = "\t\tdomain("
                        for k in range(len(sub_grid[i])):
                            dimSize = sub_grid[i][k]
                            dimSize += halo_size[case][0][k][0]
                            dimSize += halo_size[case][0][k][1]
                            output_string += "[" + str(halo_size[case][kernel_counter][k][0]) + ", "
                            output_string += str(dimSize-halo_size[case][kernel_counter][k][1]) + ']'
                        output_string += ")\n"
                        kernel_counter += 1
                    else:
                        output_string = line
                    output_file.write(output_string)
                output_file.close()
            # 使用stenCC编译器编译对应的dsl
            sw_path_out = strong_target_dir_path_prefix+"sw/"
            c_path_out_dir = strong_target_dir_path_prefix+"c/stencil_"+str(case)+mpi_tile_string
            os.mkdir(c_path_out_dir)
            cmd("./bin/stenCC "+dsl_path_out+" --emit=sw > stencil_"+str(case)+mpi_tile_string+".sw 2>&1")
            cmd("python3 translate.py stencil_"+str(case)+mpi_tile_string+".sw")
            shutil.move("stencil_"+str(case)+mpi_tile_string+".sw", sw_path_out)
            cmd("mv *.c " + c_path_out_dir)
            # 编辑driver文件
            driver_path_src = "../examples/"+str(case)+"/stencil_"+str(case)+"_driver.mpi.c"
            driver_path_out = c_path_out_dir+"/stencil_"+str(case)+mpi_tile_string+"_driver.mpi.c"
            domain_dim = len(sub_grid[i])
            dim_index = 0
            with open(driver_path_src, 'r', encoding='utf-8') as f:
                output_file = open(driver_path_out, 'w')
                for line in f.readlines():
                    output_string = ''
                    if (dim_index < domain_dim and line.find("DIM_"+str(domain_dim-1-dim_index)) !=-1):
                        output_string = "#define DIM_"+str(dim_index)+" "+str(sub_grid[i][dim_index])+"\n"
                        dim_index+=1
                    elif(line.find("swsten_store_data_to_file") == -1):
                        output_string = line
                    output_file.write(output_string)
                output_file.close()

def main():
    test_weak_scalability(stencils_2d, mpi_tile_size_2d)
    test_weak_scalability(stencils_3d, mpi_tile_size_3d)
    test_strong_scalability(stencils_2d, mpi_tile_size_2d, sub_grid_2d, halo_size_2d)
    test_strong_scalability(stencils_3d, mpi_tile_size_3d, sub_grid_3d, halo_size_3d)

if __name__ == "__main__":
    main()
