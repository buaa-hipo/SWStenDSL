stencil stencil_2d121pt_box(double input[4106][4106]) {
    iteration(20)
    operation (stencil_2d121pt_box_kernel)
    mpiTile(16, 8)
    mpiHalo([5,5][5,5])

    kernel stencil_2d121pt_box_kernel {
        tile(32, 64)
        swCacheAt(1)
        domain([5, 4101][5, 4101])
        expr {
            0.01*input[y-5][x-5] + 0.02*input[y-5][x-4] + 0.03*input[y-5][x-3] + 0.04*input[y-5][x-2] + 0.05*input[y-5][x-1] + 0.06*input[y-5][x] 
            + 0.05*input[y-5][x+1] + 0.04*input[y-5][x+2] + 0.03*input[y-5][x+3] + 0.02*input[y-5][x+4] + 0.01*input[y-5][x+5] +

            0.01*input[y-4][x-5] + 0.02*input[y-4][x-4] + 0.03*input[y-4][x-3] + 0.04*input[y-4][x-2] + 0.05*input[y-4][x-1] + 0.06*input[y-4][x] 
            + 0.05*input[y-4][x+1] + 0.04*input[y-4][x+2] + 0.03*input[y-4][x+3] + 0.02*input[y-4][x+4] + 0.01*input[y-4][x+5] +

            0.01*input[y-3][x-5] + 0.02*input[y-3][x-4] + 0.03*input[y-3][x-3] + 0.04*input[y-3][x-2] + 0.05*input[y-3][x-1] + 0.06*input[y-3][x] 
            + 0.05*input[y-3][x+1] + 0.04*input[y-3][x+2] + 0.03*input[y-3][x+3] + 0.02*input[y-3][x+4] + 0.01*input[y-3][x+5] +

            0.01*input[y-2][x-5] + 0.02*input[y-2][x-4] + 0.03*input[y-2][x-3] + 0.04*input[y-2][x-2] + 0.05*input[y-2][x-1] + 0.06*input[y-2][x] 
            + 0.05*input[y-2][x+1] + 0.04*input[y-2][x+2] + 0.03*input[y-2][x+3] + 0.02*input[y-2][x+4] + 0.01*input[y-2][x+5] +
        
            0.01*input[y-1][x-5] + 0.02*input[y-1][x-4] + 0.03*input[y-1][x-3] + 0.04*input[y-1][x-2] + 0.05*input[y-1][x-1] + 0.06*input[y-1][x] 
            + 0.05*input[y-1][x+1] + 0.04*input[y-1][x+2] + 0.03*input[y-1][x+3] + 0.02*input[y-1][x+4] + 0.01*input[y-1][x+5] +

            0.01*input[y][x-5] + 0.02*input[y][x-4] + 0.03*input[y][x-3] + 0.04*input[y][x-2] + 0.05*input[y][x-1] + 0.06*input[y][x] 
            + 0.05*input[y][x+1] + 0.04*input[y][x+2] + 0.03*input[y][x+3] + 0.02*input[y][x+4] + 0.01*input[y][x+5] +

            0.01*input[y+1][x-5] + 0.02*input[y+1][x-4] + 0.03*input[y+1][x-3] + 0.04*input[y+1][x-2] + 0.05*input[y+1][x-1] + 0.06*input[y+1][x] 
            + 0.05*input[y+1][x+1] + 0.04*input[y+1][x+2] + 0.03*input[y+1][x+3] + 0.02*input[y+1][x+4] + 0.01*input[y+1][x+5] +

            0.01*input[y+2][x-5] + 0.02*input[y+2][x-4] + 0.03*input[y+2][x-3] + 0.04*input[y+2][x-2] + 0.05*input[y+2][x-1] + 0.06*input[y+2][x] 
            + 0.05*input[y+2][x+1] + 0.04*input[y+2][x+2] + 0.03*input[y+2][x+3] + 0.02*input[y+2][x+4] + 0.01*input[y+2][x+5] +

            0.01*input[y+3][x-5] + 0.02*input[y+3][x-4] + 0.03*input[y+3][x-3] + 0.04*input[y+3][x-2] + 0.05*input[y+3][x-1] + 0.06*input[y+3][x] 
            + 0.05*input[y+3][x+1] + 0.04*input[y+3][x+2] + 0.03*input[y+3][x+3] + 0.02*input[y+3][x+4] + 0.01*input[y+3][x+5] +

            0.01*input[y+4][x-5] + 0.02*input[y+4][x-4] + 0.03*input[y+4][x-3] + 0.04*input[y+4][x-2] + 0.05*input[y+4][x-1] + 0.06*input[y+4][x] 
            + 0.05*input[y+4][x+1] + 0.04*input[y+4][x+2] + 0.03*input[y+4][x+3] + 0.02*input[y+4][x+4] + 0.01*input[y+4][x+5] +

            0.01*input[y+5][x-5] + 0.02*input[y+5][x-4] + 0.03*input[y+5][x-3] + 0.04*input[y+5][x-2] + 0.05*input[y+5][x-1] + 0.06*input[y+5][x] 
            + 0.05*input[y+5][x+1] + 0.04*input[y+5][x+2] + 0.03*input[y+5][x+3] + 0.02*input[y+5][x+4] + 0.01*input[y+5][x+5]
        }
    }    
}