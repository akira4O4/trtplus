int main()
{
    cv::Mat img = cv::imread("image.jpg");

    if (img.empty())
    {
        std::cerr << "无法读取图像！" << std::endl;
        return -1;
    }

    int height   = img.rows;
    int width    = img.cols;
    int channels = img.channels();

    // 分配内存以存储 CHW 格式的图像
    float* chw_data = new float[ height * width * channels ];

    // 调用转换函数
    chw2hwc(img, chw_data);

    // 在这里使用 chw_data...

    // 使用完成后释放内存
    delete[] chw_data;

    return 0;
}
