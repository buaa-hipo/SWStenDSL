stencil laplaceAddInput(double input[72][18][16]) {
    iteration(10)
    operation(addInput)

    kernel laplace {
        tile(5, 4, 4)
        swCacheAt(2)
        domain([1,71][1,17][0,16])
        expr {
            (input[x-1][y][z] + input[x+1][y][z] + input[x][y+1][z] + input[x][y-1][z]) - 4.0*input[x][y][z]
        }
    }

    kernel addInput {
        tile(5, 4, 4)
        swCacheAt(2)
        domain([1,71][1,17][0,16])
        expr {
            input[x][y][z]+laplace[x][y][z]
        }
    }
}