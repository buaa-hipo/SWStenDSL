# SWSten DSL

## 1. keyword

 

| keyword      | instructions                                                 |
| ------------ | ------------------------------------------------------------ |
| stencil      | stencil definition begins，followed by name and inputs       |
| float/double | inputs Type                                                  |
| iteration    | iteration number                                             |
| operation    | Name for the kernel which will return the result to user     |
| mpiHalo      | Halo region in mpi communicate                               |
| mpiTile      | number of computing node in different dimension              |
| kernel       | kernel define begins, followed by name                       |
| tile         | data block size after tile                                   |
| domain       | The range of computing                                       |
| swCacheAt    | DMA position，0 is the outerest loop                         |
| expr         | kernel computing expression, stencil inputs and kernel name can be used in expression. |

## 2. Example

```
stencil laplaceAddInputWithParamter(double U[72][18][16], c[5])
{
	iteration(100)
	mpiTile(2, 2, 2)
	operation(kernelName2)
	mpiHalo([1,1][1,1][1,1])

	kernel kernelName1{
		tile(5, 4, 4)
		swCacheAt(0)
		domain([1,71][1,17][0,16])
		expr { c[0]*U[x-1,y,z] + c[1]*U[x+1,y,z] + c[2]*U[x,y+1,z] + c[3]*U[x,y-1,z] + c[4]*U[x,y,z] }
	}

	kernel kernelName2{
		tile(5, 4, 4)
		swcacheAt(0)
		domain([1,71][1,17][0,16])
		expr { U[x,y,z] + kernelName1[x, y, z] }
	}
}
```

