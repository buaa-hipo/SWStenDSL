stencil stencil_3d7pt(double input[258][258][258]) {
    iteration(20)
    operation (stencil_3d7pt_kernel)
    mpiTile(8, 4, 4)
    mpiHalo([0,2][0,2][0,2])

    kernel stencil_3d7pt_kernel {
        tile(2, 8, 64)
        swCacheAt(2)
        domain([0, 256][0, 256][0, 256])
        expr {
            0.1*input[z+2][y][x] + 0.2*input[z+1][y][x] +
            0.3*input[z][y][x] + 0.4*input[z][y][x+1] + 0.5*input[z][y][x+2] +
            0.6*input[z][y+1][x] + 0.7*input[z][y+2][x]
        }
    }    
}