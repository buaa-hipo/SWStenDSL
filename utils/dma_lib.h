/**
 * @file dma_lib.h
 * @author Bangduo Chen (chenbangduo@buaa.edu.cn)
 * @brief 本文件包含从核使用的dma相关函数
 * @version 0.1
 * @date 2021-07-04
 * 
 * @copyright Copyright (c) HiPO Beihang University 2021
 * 
 */

#ifndef _SWSTEN_DMA_LIB_H_
#define _SWSTEN_DMA_LIB_H_

// DMA_get
#define DMA_get(src, dst, z_dim_size, cnt, stride, bsize) {\
    volatile unsigned long get_reply = 0;\
    int z_iter;\
    for (z_iter = 0; z_iter < z_dim_size; z_iter++)\
        athread_get(PE_MODE, &src, &dst, cnt, &get_reply, 0, stride, bsize);\
    while(get_reply != z_dim_size);\
}

// DMA_put
#define DMA_put(src, dst, z_dim_size, cnt, stride, bsize) {\
    volatile unsigned long put_reply = 0;\
    int z_iter;\
    for (z_iter = 0; z_iter < z_dim_size; z_iter++)\
        athread_put(PE_MODE, &src, &dst, cnt, &put_reply, stride, bsize);\
    while(put_reply != z_dim_size);\
}

#endif /* end of _SWSTEN_DMA_LIB_H_ */
