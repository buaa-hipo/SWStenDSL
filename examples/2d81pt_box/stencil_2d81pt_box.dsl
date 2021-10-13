stencil stencil_2d81pt_box(double input[4104][4104]) {
    iteration(20)
    operation (stencil_2d81pt_box_kernel)
    mpiTile(16, 8)
    mpiHalo([4,4][4,4])

    kernel stencil_2d81pt_box_kernel {
        tile(32, 64)
        swCacheAt(1)
        domain([4, 4100][4, 4100])
        expr {
            0.1*input[y-4][x-4] + 0.2*input[y-4][x-3] + 0.3*input[y-4][x-2] + 0.4*input[y-4][x-1] + 0.5*input[y-4][x] + 0.6*input[y-4][x+1] + 0.7*input[y-4][x+2] + 0.8*input[y-4][x+3] + 0.9*input[y-4][x+4] +
            0.1*input[y-3][x-4] + 0.2*input[y-3][x-3] + 0.3*input[y-3][x-2] + 0.4*input[y-3][x-1] + 0.5*input[y-3][x] + 0.6*input[y-3][x+1] + 0.7*input[y-3][x+2] + 0.8*input[y-3][x+3] + 0.9*input[y-3][x+4] +
            0.1*input[y-2][x-4] + 0.2*input[y-2][x-3] + 0.3*input[y-2][x-2] + 0.4*input[y-2][x-1] + 0.5*input[y-2][x] + 0.6*input[y-2][x+1] + 0.7*input[y-2][x+2] + 0.8*input[y-2][x+3] + 0.9*input[y-2][x+4] +
            0.1*input[y-1][x-4] + 0.2*input[y-1][x-3] + 0.3*input[y-1][x-2] + 0.4*input[y-1][x-1] + 0.5*input[y-1][x] + 0.6*input[y-1][x+1] + 0.7*input[y-1][x+2] + 0.8*input[y-1][x+3] + 0.9*input[y-1][x+4] +
            0.1*input[y][x-4] + 0.2*input[y][x-3] + 0.3*input[y][x-2] + 0.4*input[y][x-1] + 0.5*input[y][x] + 0.6*input[y][x+1] + 0.7*input[y][x+2] + 0.8*input[y][x+3] + 0.9*input[y][x+4] +
            0.1*input[y+1][x-4] + 0.2*input[y+1][x-3] + 0.3*input[y+1][x-2] + 0.4*input[y+1][x-1] + 0.5*input[y+1][x] + 0.6*input[y+1][x+1] + 0.7*input[y+1][x+2] + 0.8*input[y+1][x+3] + 0.9*input[y+1][x+4] +
            0.1*input[y+2][x-4] + 0.2*input[y+2][x-3] + 0.3*input[y+2][x-2] + 0.4*input[y+2][x-1] + 0.5*input[y+2][x] + 0.6*input[y+2][x+1] + 0.7*input[y+2][x+2] + 0.8*input[y+2][x+3] + 0.9*input[y+2][x+4] +
            0.1*input[y+3][x-4] + 0.2*input[y+3][x-3] + 0.3*input[y+3][x-2] + 0.4*input[y+3][x-1] + 0.5*input[y+3][x] + 0.6*input[y+3][x+1] + 0.7*input[y+3][x+2] + 0.8*input[y+3][x+3] + 0.9*input[y+3][x+4] +
            0.1*input[y+4][x-4] + 0.2*input[y+4][x-3] + 0.3*input[y+4][x-2] + 0.4*input[y+4][x-1] + 0.5*input[y+4][x] + 0.6*input[y+4][x+1] + 0.7*input[y+4][x+2] + 0.8*input[y+4][x+3] + 0.9*input[y+4][x+4]
        }
    }    
}