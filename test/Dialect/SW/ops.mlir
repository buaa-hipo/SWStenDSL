// SW 方言测试

module {
        
    sw.module @symbol_name :
    cacheRead(%cacheRead1:!sw.memref<1x1x1xf64>, %cacheRead2:!sw.memref<2x2x2xf64>)
    cacheWrite(%cacheWrite1: !sw.memref<3x3x3xf64>, %cacheWrite2:!sw.memref<4x4x4xf64>)
    {
        sw.func @test(%arg0:!sw.memref<3x3x3xf64>, %arg1:!sw.memref<3x3x3xf64>) {

            %15 = sw.getID : i64
            %16 = sw.constant 3 : i64
            %17 = sw.constant 1 : i64
            sw.for %i = %15 to %16 step %17 : i64 {
                sw.for %j = %15 to %16 step %17 : i64 {
                    sw.for %k = %15 to %16 step %17 : i64 {
                        sw.memcpyToLDM %arg0, %cacheRead1 [%i, %j, %k] : z_dim(3) cnt(27) stride(0) bsize(0) : (!sw.memref<3x3x3xf64> , !sw.memref<1x1x1xf64>, i64)
                        %0 = sw.constant 2.0 : f64
                        // %1 = sw.constant 3.0 : f64
                        %1 = sw.load %cacheRead1 [%i, %j, %k] : (!sw.memref<1x1x1xf64>, i64) -> f64
                        %2 = sw.addf %0, %1 : f64
                        %3 = sw.subf %2, %0 : f64
                        %4 = sw.mulf %3, %2 : f64
                        %5 = sw.divf %4, %0 : f64

                        // 计算坐标
                        %6 = sw.addi %i, %15 : i64
                        %7 = sw.addi %j, %16 : i64
                        %8 = sw.addi %k, %17 : i64

                        %9 = sw.muli %6, %15 : i64
                        %10 = sw.muli %7, %16 : i64
                        %11 = sw.muli %8, %17 : i64

                        %12 = sw.subi %9, %15 : i64
                        %13 = sw.subi %10, %16 : i64
                        %14 = sw.subi %11, %17 : i64

                        sw.store %5, %cacheWrite1 [%12, %13, %14] : f64 to (!sw.memref<3x3x3xf64>, i64)
                        
                        sw.memcpyToMEM %cacheWrite1, %arg1 [%i, %j, %k] : z_dim(3) cnt(27) stride(0) bsize(0) : (!sw.memref<3x3x3xf64> , !sw.memref<3x3x3xf64>, i64)
                        
                        sw.yield
                    }
                    sw.yield
                }
                sw.yield
            }
            sw.return
        }
        sw.module_end
    }

    sw.main_func @main(%in: !sw.memref<1x1x1xf64>, %out: !sw.memref<2x2x2xf64>) {
        %18 = sw.alloc : !sw.memref<3x3x3xf64>
        sw.dealloc %18 : !sw.memref<3x3x3xf64>
        sw.launch_func @test(%in:!sw.memref<1x1x1xf64>, %out:!sw.memref<2x2x2xf64>)
        sw.launch (%arg0=%in:!sw.memref<1x1x1xf64>, %arg1=%out:!sw.memref<2x2x2xf64>) :
	        cacheRead(%cacheRead:!sw.memref<3x3x3xf64>)
	        cacheWrite(%cacheWrite:!sw.memref<4x4x4xf64>){
            
            sw.terminator
        }
        sw.main_return
    }
}