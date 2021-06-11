stencil laplaceWithParamAddInput(double input[72][18][16], param[5]) {
    iteration(100)
    mpiTile(1, 1, 1)
    operation(AddInput)

    kernel laplaceWithParam {
        tile(5, 4, 4)
        swCacheAt(0)
        expr { 
            param[0]*input[x-1][y][z] + param[1]*input[x+1][y][z] + param[2]*input[x][y+1][z] + param[3]*input[x][y-1][z] + param[4]*input[x][y][z] 
        }
    }

    kernel AddInput{
        tile(5, 4, 4)
        swCacheAt(0)
        expr {
            laplaceWithParam[x][y][z] + input[x][y][z]
        }
    }
}