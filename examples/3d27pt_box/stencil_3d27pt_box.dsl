stencil stencil_3d27pt_box(double input[258][258][258]) {
    iteration(20)
    operation (stencil_3d27pt_box_kernel)
    mpiTile(8, 4, 4)
    mpiHalo([1,1][1,1][1,1])

    kernel stencil_3d27pt_box_kernel {
        tile(4, 8, 64)
        swCacheAt(2)
        domain([1, 257][1, 257][1, 257])
        expr {
            # panel 0
            0.1*input[z-1][y-1][x-1] + 0.2*input[z-1][y-1][x] + 0.3*input[z-1][y-1][x+1]
            + 0.4*input[z-1][y][x-1] + 0.5*input[z-1][y][x] + 0.6*input[z-1][y][x+1]
            + 0.7*input[z-1][y+1][x-1] + 0.8*input[z-1][y+1][x] + 0.9*input[z-1][y+1][x+1]
            
            # panel 1
            + 0.1*input[z][y-1][x-1] + 0.2*input[z][y-1][x] + 0.3*input[z][y-1][x+1]
            + 0.4*input[z][y][x-1] + 0.5*input[z][y][x] + 0.6*input[z][y][x+1]
            + 0.7*input[z][y+1][x-1] + 0.8*input[z][y+1][x] + 0.9*input[z][y+1][x+1]

            # panel 2
            + 0.1*input[z+1][y-1][x-1] + 0.2*input[z+1][y-1][x] + 0.3*input[z+1][y-1][x+1]
            + 0.4*input[z+1][y][x-1] + 0.5*input[z+1][y][x] + 0.6*input[z+1][y][x+1]
            + 0.7*input[z+1][y+1][x-1] + 0.8*input[z+1][y+1][x] + 0.9*input[z+1][y+1][x+1]
        }
    }
}