from os import system as cmd
import os
import shutil

stencils_2d = ["2d9pt_star", "2d81pt_box", "2d121pt_box", "2d5pt_arbitrary_shape", "2d5pt_nested"]
stencils_3d = ["3d13pt_star", "3d27pt_box", "3d125pt_box", "3d7pt_arbitrary_shape", "3d7pt9pt_nested"]

athread_compare_dir_path_prefix = "athread_compare/"
if os.path.exists(athread_compare_dir_path_prefix) is True:
    shutil.rmtree(athread_compare_dir_path_prefix)
os.mkdir(athread_compare_dir_path_prefix)

def test_athread_compare(stencils):
    for case in stencils:
        print("athread case: "+case)
        common_prefix = "../examples/"+case+"/stencil_"+case
        driver_path_src = common_prefix+"_driver.serial.c"
        master_path_src = common_prefix+"_kernel.serial_master.c"
        slave_path_src = common_prefix+"_kernel.serial_slave.c"
        c_path_out_dir = athread_compare_dir_path_prefix+"/stencil_"+case
        driver_path_out = c_path_out_dir+"/stencil_"+str(case)+"_driver.serial.c"
        os.mkdir(c_path_out_dir)
        # 复制对应的driver, 并去掉结果验证部分
        with open(driver_path_src, 'r', encoding='utf-8') as f:
            output_file = open(driver_path_out, 'w')
            for line in f.readlines():
                output_string = ''
                if (line.find("mpe_verify();") == -1):
                    output_string = line
                output_file.write(output_string)
            output_file.close()
        # 复制master和slave文件
        shutil.copy(master_path_src, c_path_out_dir)
        shutil.copy(slave_path_src, c_path_out_dir)

openACC_compare_dir_path_prefix = "openACC_compare/"
if os.path.exists(openACC_compare_dir_path_prefix) is True:
    shutil.rmtree(openACC_compare_dir_path_prefix)
os.mkdir(openACC_compare_dir_path_prefix)

def test_openACC_compare(stencils):
    for case in stencils:
        print("openACC case: " + case)
        src_path = "../examples/"+case+"/stencil_"+case+"_kernel.openacc.c"
        dst_dir_path = openACC_compare_dir_path_prefix+"/stencil_"+case
        os.mkdir(dst_dir_path)
        # 复制openAcc文件
        shutil.copy(src_path, dst_dir_path)



def main():
    test_athread_compare(stencils_2d)
    test_athread_compare(stencils_3d)
    test_openACC_compare(stencils_2d)
    test_openACC_compare(stencils_3d)

if __name__ == '__main__':
    main()