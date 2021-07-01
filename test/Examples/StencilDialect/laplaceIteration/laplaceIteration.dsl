stencil laplace(double input[72][18][16]) {
    iteration(10)
    operation(laplaceKernel)
    mpiTile(3, 3, 3)
    mpiHalo([1,1][1,1][0,0])

    kernel laplaceKernel {
        tile(5, 4, 4)
        swCacheAt(2)
        domain([1,71][1,17][0,16])
        expr {
            (input[x-1][y][z] + input[x+1][y][z] + input[x][y+1][z] + input[x][y-1][z]) - 4.0*input[x][y][z]
        }
    }
}
