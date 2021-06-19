stencil laplaceWithParameter(double input[72][18][16], param[5]) {
    iteration(10)
    operation(laplaceWithParamKernel)

    kernel laplaceWithParamKernel {
        tile(5, 4, 4)
        swCacheAt(0)
        domain([1,71][1,17][0,16])
        expr { 
            param[0]*input[x-1][y][z] + param[1]*input[x+1][y][z] + param[2]*input[x][y+1][z] + param[3]*input[x][y-1][z] + param[4]*input[x][y][z] 
        }
    }
}