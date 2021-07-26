stencil stencil_3d13pt_star(double input[260][260][260]) {
    iteration(20)
    operation (stencil_3d13pt_star_kernel)
    mpiTile(8, 4, 4)
    mpiHalo([2,2][2,2][2,2])

    kernel stencil_3d13pt_star_kernel {
        tile(2, 8, 64)
        swCacheAt(2)
        domain([2, 258][2, 258][2, 258])
        expr {
            0.1*input[z-2][y][x] + 0.2*input[z-1][y][x]
            + 0.3*input[z+1][y][z] + 0.4*input[z+2][y][x]
            + 0.5*input[z][y-2][x] + 0.6*input[z][y-1][x]
            + 0.7*input[z][y+1][x] + 0.8*input[z][y+2][x]
            + 0.9*input[z][y][x-2] + 1.0*input[z][y][x-1]
            + 1.1*input[z][y][x+1] + 1.2*input[z][y][x+2]
            + 1.3*input[z][y][x]
        }
    }    
}