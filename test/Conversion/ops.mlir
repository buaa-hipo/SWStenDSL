module {
    func @test(%in: !stencil.field<6x6xf64>, %out: !stencil.field<6x6xf64>, 
                %parameter: !stencil.field<4xf64>) 
        attributes { stencil.program } {
            %0 = stencil.apply(%arg0 = %in : !stencil.field<6x6xf64>, %arg1 = %parameter : !stencil.field<4xf64>)  -> !stencil.field<6x6xf64> {
                %1 = stencil.access %arg0 [0, 0] : (!stencil.field<6x6xf64>) ->f64
                %6 = stencil.load %arg1 [0] : (!stencil.field<4xf64>) -> f64
                %16 = constant 1.0 : f64
                %17 = stencil.store %16 : (f64) -> !stencil.result<f64>      
                stencil.return %17 : !stencil.result<f64>
            } in ([1, 1] : [5, 5]) tile([2, 2]) cacheAt(0)

            stencil.copy %0 to %out ([1, 1] : [5, 5]) : !stencil.field<6x6xf64> to !stencil.field<6x6xf64>

        return
    }
}