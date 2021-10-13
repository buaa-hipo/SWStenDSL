import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

stencils =  ["2d9pt_star", "2d81pt_box", "2d121pt_box", "2d5pt_arbitrary_shape", 
             "3d13pt_star", "3d27pt_box", "3d125pt_box", "3d7pt_arbitrary_shape"]

dim1_size = 4
dim0_size = 4
nrows = 2
ncols = 4

plt.rc('font',family='Times New Roman')
fig, ax = plt.subplots(nrows, ncols, figsize=(18, 8))

def draw_stencil_heat(stencil, file_ptr, index):
    print(stencil)
    data = []
    mask = []
    y = 1000000
    col = index % ncols
    row = math.floor(index / ncols)
    for i in range(dim1_size*dim0_size):
        line = file_ptr.readline()
        value = line.split(" ")[4]
        if (value == "three"):
            value = 1000000
        value = float(value)    
        data.append(value)
        if (y > value):
            y = value
        
        if (value == 1000000):
            mask.append(True)
        else:
            mask.append(False)
    
    print(y)
    for i in range(dim1_size*dim0_size):
        data[i] = y / data[i]
    
    a = np.array(data).reshape(dim1_size, dim0_size)
    mask = np.array(mask).reshape(dim1_size, dim0_size)

    df = pd.DataFrame(a,
                        columns = [8, 16, 32, 64],
                        index = [4, 8, 16, 32])
    sns.heatmap(df, mask = mask, ax = ax[row][col], annot=False)
    ax[row][col].set_xlabel('The value of $\Theta$')
    ax[row][col].set_ylabel('The value of $\Omega$')
    ax[row][col].set_title(stencil)


def main():
    with open("heatmap_result", 'r') as f:
        for i in range(len(stencils)):
            draw_stencil_heat(stencils[i], f, i)
    plt.tight_layout()
    # plt.show()
    fig.savefig('swsten_heatmap.pdf')

if __name__ == "__main__":
    main()