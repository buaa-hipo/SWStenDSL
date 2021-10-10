stencil stencil_3d7pt9pt_nested(double input[258][258][258]) {
    iteration(20)
    operation (stencil_3d9pt_x)
    mpiTile(8, 4, 4)
    mpiHalo([5,5][5,5][5,5])

    kernel stencil_3d7pt_star {
        tile(4, 8, 64)
        swCacheAt(2)
        domain([1, 257][1, 257][1, 257])
        expr {
            # panel 0
            0.5*input[z-1][y][x]
            
            # panel 1
            + 0.2*input[z][y-1][x]
            + 0.4*input[z][y][x-1] + 0.5*input[z][y][x] + 0.6*input[z][y][x+1]
            + 0.8*input[z][y+1][x]

            # panel 2
            + 0.5*input[z+1][y][x]
        }
    }

    kernel stencil_3d9pt_x {
        tile(4, 8, 8)
        swCacheAt(2)
        domain([5, 253][5, 253][5, 253])
        expr {
            # panel 0
            0.1*input[z-1][y-1][x-1] + 0.3*input[z-1][y-1][x+1]
            + 0.7*input[z-1][y+1][x-1] + 0.9*input[z-1][y+1][x+1]
            
            # panel 1
            + 0.5*input[z][y][x]

            # panel 2
            + 0.1*input[z+1][y-1][x-1] + 0.3*input[z+1][y-1][x+1]
            + 0.7*input[z+1][y+1][x-1] + 0.9*input[z+1][y+1][x+1]
        }
    }
}