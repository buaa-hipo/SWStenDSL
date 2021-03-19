# Stencil Dialect

## 1. 变量类型

-   Field类型

    该类型对应输入和输出的结构化网格，支持1D，2D，3D的单双精度浮点数

-   Result类型

    该类型用来表示计算的结果，为标量浮点数

-   Index类型(内部类型)

    多维64位整型数组，可用来表示偏移量大小，域的边界，unroll的大小

## 2. 操作定义

-    apply

     在给定的结构化网格上（Field类型）执行具体的stencil计算，并返回结果（Field类型）

     Example：

     ```
                 %0 = stencil.apply(%arg0 = %in : !stencil.field<6x6xf64>, %arg1 = %parameter : !stencil.field<4xf64>)  -> !stencil.field<6x6xf64> {
                 	... ...
                 } in ([0, 0] : [4, 4])
     
                 
     ```

-   access

    位于apply操作的计算域中，stencil计算中访问相对于当前点偏移位置的点，返回对应点的值

    Example：

    ```
    %5 = stencil.access %arg0 [0, -1] : (!stencil.field<6x6xf64>) ->f64
    ```
-   load

    采用绝对坐标地址的方式访问数组，用于访问数组式参数

    Example:

    ```
    %6 = stencil.load %arg1 [0] : (!stencil.field<4xf64>) -> f64
    ```
-   store

    位于apply操作的计算域中，将计算结果写入到result类型中，有时可能回没有计算结果（unrolling中未能整除时）

    Example：

    ```
    %17 = stencil.store %16 : (f64) -> !stencil.result<f64>  
    %17 = stencil.store : () -> !stencil.result<f64>
    ```

-   return

    apply操作计算域的终结符，负责将result类型的计算结果返回，在unrolling时需要提供各个维度unrolling的factor

    Example：

    ```
    stencil.return %17 : !stencil.result<f64>
    stencil.return unroll [1, 3] %6#0, %6#1, %6#2 : !stencil.result<f64>, !stencil.result<f64>, !stencil.result<f64>
    ```

-   copy

    根据给定的上下界，将Field类型的计算结果写入到输出数组中（Field类型）

    Example：

    ```
    stencil.copy %0 to %out ([0, 0] : [4, 4]) : !stencil.field<6x6xf64> to !stencil.field<6x6xf64>
    ```

    





