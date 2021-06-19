# SWSten DSL

## 1. Example

```
stencil laplaceAddInputWithParamter(double U[72][18][16], c[5])
{
	iteration(100)
	mpiTile(2, 2, 2)
	operation(kernelName2)

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
