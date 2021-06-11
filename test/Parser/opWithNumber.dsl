stencil laplaceWithNumberAddInput(double input[72][18][16]) {
    iteration(100)
    mpiTile(1, 1, 1)
    operation(AddInput)

    kernel laplaceWithParam {
        tile(5, 4, 4)
        swCacheAt(0)
        expr { 
            -1*input[x-1][y][z] + 1*input[x+1][y][z] + 2*input[x][y+1][z] + 3*input[x][y-1][z] + 4*input[x][y][z] 
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