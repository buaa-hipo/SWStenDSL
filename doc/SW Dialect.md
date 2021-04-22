# SW Dialect

## 1. 变量类型

-   MemRef类型

    该类型对应相应的输入和输出数组, 支持1D, 2D, 3D的单双精度浮点数

-   Index类型(内部类型)

    多维64位整型数组, 用来表示坐标位置

## 2. 操作定义

-   module

    包围生成为从核函数的代码，每个从核函数都有一个相对应的module

    ```
    sw.module @symbol_name
    {
        ...
    }
    ```
    
-   module_end

    module的终结符

    ```
    sw.module_end
    ```
    
-   func

    outline之后从核函数， 位于某一个module中

    ```
    sw.func @foo(%arg0: index) 	cacheRead(%cacheRead:!sw.memref<3x3x3xf64>) cacheWrite(%cacheWrite:!sw.memref<3x3x3xf64>) {
    	...
    }
    ```

-   return 

    func操作的终结符

    ```
    sw.return
    ```

-   main_func

    主核函数

    ```
    sw.main_func @main(%in: !sw.memref<1x1x1xf64>, %out: !sw.memref<2x2x2xf64>) {
    	...
    }
    ```

-   main_return

    主核函数终结符

    ```
    sw.main_return
    ```

-   launch_func

    主核函数调用从核函数func

    ```
    sw.launch_func @kernels::@kernel_1(%arg0: f32, %arg1 : sw.memref<4xf32>)
    ```

-   launch

    未outline的从核函数，为IR转化提供便利

    ```
    sw.launch (%arg0=%0 : f32, %arg1=%1 : sw.memref<4xf32>) :
    	cacheRead(%cacheRead:!sw.memref<3x3x3xf64>)
    	cacheWrite(%cacheWrite:!sw.memref<3x3x3xf64>){
    	...
    }
    ```

-   terminator

    launch操作终结符

    ```
    sw.terminator
    ```
    
-   for

    for循环

    ```
    sw.for %i = %lb to %ub step %step : i64{
    	...
    }
    ```

-   yield

    for操作终结符

    ```
    sw.yield
    ```

-   load

    从指定位置加载数据

    ```
    %0 = sw.load %1 [0, 1, 3] : (!sw.memref<3x3x3xf64>, i32) -> f64
    ```

-   constant

    定义常量

    ```
    $1 = sw.constant 1.000000e+00 : f64
    ```

-   getID

    获取当前从核号

    ```
    $1 = sw.getID : I64
    ```

-   addf

    加法

    ```
    %2 = sw.addf %0, %1 : f64
    ```

-   subf

    减法

    ```
    %2 = sw.subf %0, %1 : f64
    ```

-   mulf

    乘法

    ```
    %2 = sw.mulf %0, %1 : f64
    ```

-   divf

    除法

    ```
    %2 = sw.divf %0, %1 : f64
    ```

-   store

    将结果写回到指定位置

    ```
    sw.store %1, %2 [0, 0, 1] : f64 to (!sw.memref<3x3x3xf64>, i32)
    ```
    
-   addi

    加法, 主要用于坐标运算

    ```
    %2 = sw.addi %0, %1 : i32
    ```
    
-   subi

    减法, 主要用于坐标运算

    ```
    %2 = sw.subi %0, %1 : i32
    ```
    
-   muli

    乘法, 主要用于坐标运算

    ```
    %2 = sw.muli %0, %1 : i32
    ```
    
-   memcpy

    LDM和主存之间内存传输，分为两个方向

    ```
    sw.memcpyToLDM %mem_addr, %ldm_addr [%i, %j, %k] : z_dim(3) cnt(4) stride(5) bsize(6) : (!sw.memref<6x6x6xf64>, !sw.memref<6x6x6xf64>, i64)
    sw.memcpyToMEM %ldm_addr, %mem_addr [%i, %j, %k] : z_dim(3) cnt(4) stride(5) bsize(6) : (!sw.memref<6x6x6xf64>, !sw.memref<6x6x6xf64>, i64)
    ```

-   allocOp

    为指定的类型申请内存空间

    ```
    %3 = sw.alloc : !sw.memref<6x6x6xf64>
    ```

-   deAllocOp

    释放指定变量的内存空间

    ```
    sw.dealloc %1
    ```

-	
    ------

    NOTE：未来可能需要支持

-   if

    if语句

-   else

    else语句