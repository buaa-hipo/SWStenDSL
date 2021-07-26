stencil laplace(double input[18][18]) {
    iteration(10)
    operation(laplaceKernel)
    mpiTile(3, 3)
    mpiHalo([1,1][1,1])

    kernel laplaceKernel {
        tile(4, 4)
        swCacheAt(0)
        domain([1,17][1,17])
        expr {
            (input[x-1][y] + input[x+1][y] + input[x][y+1] + input[x][y-1]) - 4.0*input[x][y]
        }
    }
}
