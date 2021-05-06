// laplace 算子
// A'(x, y, z) = (A(x-1, y, z) + A(x+1, y, z) + A(x, y+1, z) + A(x, y-1, z)) - 4.0 * A(x, y, z)
module {
    func @laplaceWithParameter(%in: !stencil.field<72x18x16xf64>, %out: !stencil.field<72x18x16xf64>, %parameterArray : !stencil.field<5xf64>) attributes {stencil.program} {
        %0 = stencil.apply(%arg0 = %in : !stencil.field<72x18x16xf64>, %parameter = %parameterArray : !stencil.field<5xf64>) -> !stencil.field<72x18x16xf64> {
            %1 = stencil.access %arg0 [-1, 0, 0] : (!stencil.field<72x18x16xf64>) -> f64
            %2 = stencil.access %arg0 [1, 0, 0] : (!stencil.field<72x18x16xf64>) -> f64
            %3 = stencil.access %arg0 [0, 1, 0] : (!stencil.field<72x18x16xf64>) -> f64
            %4 = stencil.access %arg0 [0, -1, 0] : (!stencil.field<72x18x16xf64>) -> f64
            %5 = stencil.access %arg0 [0, 0, 0] : (!stencil.field<72x18x16xf64>) -> f64

            %param0 = stencil.load %parameter [0] : (!stencil.field<5xf64>) -> f64
            %param1 = stencil.load %parameter [1] : (!stencil.field<5xf64>) -> f64
            %param2 = stencil.load %parameter [2] : (!stencil.field<5xf64>) -> f64
            %param3 = stencil.load %parameter [3] : (!stencil.field<5xf64>) -> f64
            %param4 = stencil.load %parameter [4] : (!stencil.field<5xf64>) -> f64
            
            %6 = mulf %1, %param0 : f64
            %7 = mulf %2, %param1 : f64
            %8 = mulf %3, %param2 : f64
            %9 = mulf %4, %param3 : f64
            %10 = mulf %5, %param4 : f64

            %11 = addf %6, %7 : f64
            %12 = addf %11, %8 : f64
            %13 = addf %12, %9 : f64
            %14 = addf %13, %10 : f64

            %15 = stencil.store %14 : (f64) -> !stencil.result<f64>
            stencil.return %15 : !stencil.result<f64>
        } in ([1, 1, 0] : [71, 17, 16]) tile([5, 4, 4]) cacheAt(0)

        stencil.copy %0 to %out ([1, 1, 0] : [71, 17, 16]) : !stencil.field<72x18x16xf64> to !stencil.field<72x18x16xf64>
        return
    }

    func @laplaceWithParameter_iteration(%in: !stencil.field<72x18x16xf64>, %out: !stencil.field<72x18x16xf64>, %parameterArray : !stencil.field<5xf64>) attributes { stencil.iteration }
    {
        stencil.iteration @laplaceWithParameter((%in: !stencil.field<72x18x16xf64>, %out: !stencil.field<72x18x16xf64>, %parameterArray : !stencil.field<5xf64>),
                                    (%out: !stencil.field<72x18x16xf64>, %in: !stencil.field<72x18x16xf64>, %parameterArray : !stencil.field<5xf64> ), 5)

        return
    }
}