// laplace 算子
// A'(x, y, z) = (A(x-1, y, z) + A(x+1, y, z) + A(x, y+1, z) + A(x, y-1, z)) - 4.0 * A(x, y, z)
module {
    func @laplace(%in: !stencil.field<72x18x16xf64>, %out: !stencil.field<72x18x16xf64>) attributes {stencil.program} {
        %0 = stencil.apply(%arg0 = %in : !stencil.field<72x18x16xf64>) -> !stencil.field<72x18x16xf64> {
            %1 = stencil.access %arg0 [-1, 0, 0] : (!stencil.field<72x18x16xf64>) -> f64
            %2 = stencil.access %arg0 [1, 0, 0] : (!stencil.field<72x18x16xf64>) -> f64
            %3 = stencil.access %arg0 [0, 1, 0] : (!stencil.field<72x18x16xf64>) -> f64
            %4 = stencil.access %arg0 [0, -1, 0] : (!stencil.field<72x18x16xf64>) -> f64
            %5 = stencil.access %arg0 [0, 0, 0] : (!stencil.field<72x18x16xf64>) -> f64
            
            %6 = addf %1, %2 : f64
            %7 = addf %3, %4 : f64
            %8 = addf %6, %7 : f64
            %cst = constant -4.0 : f64
            %9 = mulf %5, %cst : f64
            %10 = addf %9, %8 : f64

            %11 = stencil.store %10 : (f64) -> !stencil.result<f64>
            stencil.return %11 : !stencil.result<f64>
        } in ([1, 1, 0] : [71, 17, 16]) tile([5, 4, 4]) cacheAt(2)

        stencil.copy %0 to %out ([1, 1, 0] : [71, 17, 16]) : !stencil.field<72x18x16xf64> to !stencil.field<72x18x16xf64>
        return
    }

    func @laplace_iteration(%in: !stencil.field<72x18x16xf64>, %out: !stencil.field<72x18x16xf64>) attributes { stencil.iteration }
    {
        stencil.iteration @laplace((%in: !stencil.field<72x18x16xf64>, %out: !stencil.field<72x18x16xf64>),
                                    (%out: !stencil.field<72x18x16xf64>, %in: !stencil.field<72x18x16xf64>), 5)

        return
    }
}