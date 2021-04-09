// Stencil 方言测试

module {
    func @test(%in: !stencil.field<6x6xf64>, %out: !stencil.field<6x6xf64>, 
                %parameter: !stencil.field<4xf64>) 
        attributes { stencil.program } {
            %0 = stencil.apply(%arg0 = %in : !stencil.field<6x6xf64>, %arg1 = %parameter : !stencil.field<4xf64>)  -> !stencil.field<6x6xf64> {
                %1 = stencil.access %arg0 [0, 0] : (!stencil.field<6x6xf64>) ->f64
                %2 = stencil.access %arg0 [1, 0] : (!stencil.field<6x6xf64>) ->f64
                %3 = stencil.access %arg0 [-1, 0] : (!stencil.field<6x6xf64>) ->f64
                %4 = stencil.access %arg0 [0, 1] : (!stencil.field<6x6xf64>) ->f64
                %5 = stencil.access %arg0 [0, -1] : (!stencil.field<6x6xf64>) ->f64

                %6 = stencil.load %arg1 [0] : (!stencil.field<4xf64>) -> f64
                %7 = stencil.load %arg1 [1] : (!stencil.field<4xf64>) -> f64
                %8 = stencil.load %arg1 [2] : (!stencil.field<4xf64>) -> f64
                %9 = stencil.load %arg1 [3] : (!stencil.field<4xf64>) -> f64

                %10 = mulf %2, %6 : f64
                %11 = mulf %3, %7 : f64
                %12 = mulf %4, %8 : f64
                %13 = mulf %5, %9 : f64

                %14 = addf %10, %11 : f64
                %15 = addf %14, %12 : f64
                %16 = subf %15, %14 : f64
                %17 = divf %16, %15 : f64
                %18 = addf %15, %13 : f64

                %19 = stencil.store %18 : (f64) -> !stencil.result<f64>      
                stencil.return %19 : !stencil.result<f64>
            } in ([1, 1] : [5, 5]) tile([2, 2]) cacheAt(0)

            stencil.copy %0 to %out ([1, 1] : [5, 5]) : !stencil.field<6x6xf64> to !stencil.field<6x6xf64>

            return
        }
}