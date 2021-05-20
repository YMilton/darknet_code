#include "im2col.h"
#include <stdio.h>

// 卷积层的每次卷积操作可以堪称权重矩阵，与输入特征图转化成同权重矩阵同等大小矩阵的乘积运算
float im2col_get_pixel(float* im, int height, int width, int channels,
    int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col)
{
    int c, h, w;
    // 输出图像的宽高
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    // 多个通道的卷积核权重个数，如ksize=3,输入channel=3,则展开的col元素为3*3*3=27
    int channels_col = channels * ksize * ksize; 
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize; // 核的cols
        int h_offset = (c / ksize) % ksize; // 核的rows
        int c_im = c / ksize / ksize; // 第几个核
        // 填充data_col
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                // h*stride,w*stride表示输入图像切去ksize*ksize像素左上角坐标点在输入的位置，加上偏置h_offset,w_offset,
                // 即输入图像按照ksize切去的像素块偏移量像素，与kernel中的某个权重对应
                int im_row = h_offset + h * stride;  // 同核权重对应像素在输入图像的坐标计算
                int im_col = w_offset + w * stride;
                // ??
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}


// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
// 判断矩阵的某元素位置是否在pad上
inline static int is_a_ge_zero_and_a_lt_b(int a, int b) {
    return (unsigned)(a) < (unsigned)(b); // unsigned强制转换负数，则为负数的补码
}

// https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cpp
// im2col_cpu_ext的data_col表示的为[channels*ksize*ksize,h_out*w_out]
void im2col_cpu_ext(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    float* data_col)
{
    // 计算输出卷积输出尺寸，dilation*(kernel-1)+1表示膨胀后的核的大小，
    // 其中dilation表示填充的空洞数量,k=3,填充的空洞数为2，则新的kernel_size=2*(3-1)+1
    const int output_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int channel_size = height * width;
    int channel, kernel_row, kernel_col, output_rows, output_col;
    // data_im表示某幅图像卷积时，某个组的开始位置
    for (channel = channels; channel--; data_im += channel_size) { // channels表示每个组的通道数
        // 第二与第三个for循环体现输出矩阵的行数
        for (kernel_row = 0; kernel_row < kernel_h; kernel_row++) { // kernel尺寸
            for (kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                // 找到卷积核的某一行在输入特征图中第一个元素的位置
                int input_row = -pad_h + kernel_row * dilation_h; // pad_h = l.pad*l.dilation,此处l.pad=1
                // 经过h_out*w_out个滑动窗口，第四个和第五个for循环体现了输出矩阵的列数
                for (output_rows = output_h; output_rows; output_rows--) { // 输出尺寸
                    // input_row<0或input_row>height的情况，height是输入图像的高
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) { 
                        for (output_col = output_w; output_col; output_col--) {
                            *(data_col++) = 0; // data_col输出
                        }
                    }
                    else {
                        int input_col = -pad_w + kernel_col * dilation_w;
                        for (output_col = output_w; output_col; output_col--) {
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                *(data_col++) = data_im[input_row * width + input_col];
                            }
                            else {
                                *(data_col++) = 0;
                            }
                            input_col += stride_w; // 增加一个列步长
                        }
                    }
                    input_row += stride_h; // 增加一个行步长
                }
            }
        }
    }
}
