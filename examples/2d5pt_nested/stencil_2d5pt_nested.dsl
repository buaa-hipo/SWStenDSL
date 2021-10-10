stencil stencil_2d5pt_nested(double input[4098][4098]) {
    iteration(20)
    operation (stencil_2d9pt_x)
    mpiTile(16, 8)
    mpiHalo([16,16][16,16])

    kernel stencil_2d9pt_star {
        tile(32, 64)
        swCacheAt(1)
        domain([1, 4097][1, 4097])
        expr {
            0.2*input[y-1][x]
            + 0.4*input[y][x-1] - 0.5*input[y][x] + 0.4*input[y][x+1] 
            + 0.2*input[y+1][x]
        }
    }

    kernel stencil_2d9pt_x {
        tile(16, 32)
        swCacheAt(1)
        domain([17, 4081][17, 4081])
        expr {
            0.1*stencil_2d9pt_star[y-1][x-1] + 0.3*stencil_2d9pt_star[y-1][x+1] 
            - 0.5*stencil_2d9pt_star[y][x]
            + 0.3*stencil_2d9pt_star[y+1][x-1] + 0.1*stencil_2d9pt_star[y+1][x+1]
        }
    }
}