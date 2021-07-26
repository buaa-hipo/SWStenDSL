stencil stencil_2d9pt_star(double input[4100][4100]) {
    iteration(20)
    operation (stencil_2d9pt_star_kernel)
    mpiTile(16, 8)
    mpiHalo([2,2][2,2])

    kernel stencil_2d9pt_star_kernel {
        tile(32, 64)
        swCacheAt(1)
        domain([2, 4098][2, 4098])
        expr {
            0.1*input[y-2][x] + 0.2*input[y-1][x] + 0.3*input[y][x-2] 
            + 0.4*input[y][x-1] + 0.5*input[y][x] + 0.6*input[y][x+1] 
            + 0.7*input[y][x+2] + 0.8*input[y+1][x] + 0.9*input[y+2][x]
        }
    }    
}