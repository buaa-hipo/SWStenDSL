stencil stencil_3d125pt_box(double input[260][260][260]) {
    iteration(20)
    operation (stencil_3d125pt_box_kernel)
    mpiTile(8, 4, 4)
    mpiHalo([2,2][2,2][2,2])

    kernel stencil_3d125pt_box_kernel {
        tile(2, 8, 64)
        swCacheAt(2)
        domain([2, 258][2, 258][2, 258])
        expr {
            # panel 0
            0.1*input[z-2][y-2][x-2] + 0.2*input[z-2][y-2][x-1] + 0.3*input[z-2][y-2][x] + 0.4*input[z-2][y-2][x+1] + 0.5*input[z-2][y-2][x+2] + 
            0.1*input[z-2][y-1][x-2] + 0.2*input[z-2][y-1][x-1] + 0.3*input[z-2][y-1][x] + 0.4*input[z-2][y-1][x+1] + 0.5*input[z-2][y-1][x+2] +
            0.1*input[z-2][y][x-2] + 0.2*input[z-2][y][x-1] + 0.3*input[z-2][y][x] + 0.4*input[z-2][y][x+1] + 0.5*input[z-2][y][x+2] +
            0.1*input[z-2][y+1][x-2] + 0.2*input[z-2][y+1][x-1] + 0.3*input[z-2][y+1][x] + 0.4*input[z-2][y+1][x+1] + 0.5*input[z-2][y+1][x+2] +
            0.1*input[z-2][y+2][x-2] + 0.2*input[z-2][y+2][x-1] + 0.3*input[z-2][y+2][x] + 0.4*input[z-2][y+2][x+1] + 0.5*input[z-2][y+2][x+2] +

            # panel 1
            0.1*input[z-1][y-2][x-2] + 0.2*input[z-1][y-2][x-1] + 0.3*input[z-1][y-2][x] + 0.4*input[z-1][y-2][x+1] + 0.5*input[z-1][y-2][x+2] + 
            0.1*input[z-1][y-1][x-2] + 0.2*input[z-1][y-1][x-1] + 0.3*input[z-1][y-1][x] + 0.4*input[z-1][y-1][x+1] + 0.5*input[z-1][y-1][x+2] +
            0.1*input[z-1][y][x-2] + 0.2*input[z-1][y][x-1] + 0.3*input[z-1][y][x] + 0.4*input[z-1][y][x+1] + 0.5*input[z-1][y][x+2] +
            0.1*input[z-1][y+1][x-2] + 0.2*input[z-1][y+1][x-1] + 0.3*input[z-1][y+1][x] + 0.4*input[z-1][y+1][x+1] + 0.5*input[z-1][y+1][x+2] +
            0.1*input[z-1][y+2][x-2] + 0.2*input[z-1][y+2][x-1] + 0.3*input[z-1][y+2][x] + 0.4*input[z-1][y+2][x+1] + 0.5*input[z-1][y+2][x+2] +

            # panel 2
            0.1*input[z][y-2][x-2] + 0.2*input[z][y-2][x-1] + 0.3*input[z][y-2][x] + 0.4*input[z][y-2][x+1] + 0.5*input[z][y-2][x+2] + 
            0.1*input[z][y-1][x-2] + 0.2*input[z][y-1][x-1] + 0.3*input[z][y-1][x] + 0.4*input[z][y-1][x+1] + 0.5*input[z][y-1][x+2] +
            0.1*input[z][y][x-2] + 0.2*input[z][y][x-1] + 0.3*input[z][y][x] + 0.4*input[z][y][x+1] + 0.5*input[z][y][x+2] +
            0.1*input[z][y+1][x-2] + 0.2*input[z][y+1][x-1] + 0.3*input[z][y+1][x] + 0.4*input[z][y+1][x+1] + 0.5*input[z][y+1][x+2] +
            0.1*input[z][y+2][x-2] + 0.2*input[z][y+2][x-1] + 0.3*input[z][y+2][x] + 0.4*input[z][y+2][x+1] + 0.5*input[z][y+2][x+2] +

            # panel 3
            0.1*input[z+1][y-2][x-2] + 0.2*input[z+1][y-2][x-1] + 0.3*input[z+1][y-2][x] + 0.4*input[z+1][y-2][x+1] + 0.5*input[z+1][y-2][x+2] + 
            0.1*input[z+1][y-1][x-2] + 0.2*input[z+1][y-1][x-1] + 0.3*input[z+1][y-1][x] + 0.4*input[z+1][y-1][x+1] + 0.5*input[z+1][y-1][x+2] +
            0.1*input[z+1][y][x-2] + 0.2*input[z+1][y][x-1] + 0.3*input[z+1][y][x] + 0.4*input[z+1][y][x+1] + 0.5*input[z+1][y][x+2] +
            0.1*input[z+1][y+1][x-2] + 0.2*input[z+1][y+1][x-1] + 0.3*input[z+1][y+1][x] + 0.4*input[z+1][y+1][x+1] + 0.5*input[z+1][y+1][x+2] +
            0.1*input[z+1][y+2][x-2] + 0.2*input[z+1][y+2][x-1] + 0.3*input[z+1][y+2][x] + 0.4*input[z+1][y+2][x+1] + 0.5*input[z+1][y+2][x+2] +

            # panel 4
            0.1*input[z+2][y-2][x-2] + 0.2*input[z+2][y-2][x-1] + 0.3*input[z+2][y-2][x] + 0.4*input[z+2][y-2][x+1] + 0.5*input[z+2][y-2][x+2] + 
            0.1*input[z+2][y-1][x-2] + 0.2*input[z+2][y-1][x-1] + 0.3*input[z+2][y-1][x] + 0.4*input[z+2][y-1][x+1] + 0.5*input[z+2][y-1][x+2] +
            0.1*input[z+2][y][x-2] + 0.2*input[z+2][y][x-1] + 0.3*input[z+2][y][x] + 0.4*input[z+2][y][x+1] + 0.5*input[z+2][y][x+2] +
            0.1*input[z+2][y+1][x-2] + 0.2*input[z+2][y+1][x-1] + 0.3*input[z+2][y+1][x] + 0.4*input[z+2][y+1][x+1] + 0.5*input[z+2][y+1][x+2] +
            0.1*input[z+2][y+2][x-2] + 0.2*input[z+2][y+2][x-1] + 0.3*input[z+2][y+2][x] + 0.4*input[z+2][y+2][x+1] + 0.5*input[z+2][y+2][x+2]
        }
    }    
}