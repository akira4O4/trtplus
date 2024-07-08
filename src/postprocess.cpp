//
// Created by main on 24-4-11.
//

#include "postprocess.h"

namespace postprocess
{
// easy impl
void argmax(float *x, int *y, int rows, int cols, int chan)
{
    for (int i = 0, wh = rows * cols; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int max     = -10000000;
            int max_ind = -1;
            for (int k = 0; k < chan; k++)
            {
                if (x[ k * wh + i * cols + j ] > max)
                {
                    max     = x[ k * wh + i * cols + j ];
                    max_ind = k;
                }
            }
            y[ i * cols + j ] = max_ind;
        }
    }
}

void classification(const float output[], int num_classes, std::vector<float> thr, std::vector<std::string> labels,
                    std::string save)
{

    auto max_score_id = argmax_impl(output, output + num_classes);
    INFO("Prediction-> label: %s score:%f", labels[ max_score_id ].c_str(), output[ max_score_id ]);
    //    for (int i = 0; i < num_classes; ++i)
    //    {
    //        std::cout << i << labels[ i ] << " : " << output[ i ] << std::endl;
    //        if (output[ i ] >= thr[ i ])
    //        {
    //            std::string label = labels[ i ];
    //            std::cout << label << " : " << output[ i ] << std::endl;
    //        }
    //    }
}
} // namespace postprocess