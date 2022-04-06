#include "buffers.h"
#include "common.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <chrono>
#include "TRTModelBase.h"

const std::string gSampleName = "TensorRT.caffe_SSD";


struct SampleSSDParams : public ConfigParams
{
    int outputClsSize;              //!< The number of output classes
    int keepTopK;                   //!< The maximum number of detection post-NMS
    float visualThreshold;          //!< The minimum score threshold to consider a detection
};


class TRTModelSSD : public TRTModelBase
{
private:
    int outputClsSize;              //!< The number of output classes
    int keepTopK;                   //!< The maximum number of detection post-NMS
    float visualThreshold;          //!< The minimum score threshold to consider a detection
    std::vector<trtCommon::PPM<3, 300, 300>> mPPMs; //!< PPMs of test images

protected:
    bool processInput(const trtCommon::BufferManager& buffers) override;
    bool processOutput(const trtCommon::BufferManager& buffers) override;

public:
    TRTModelSSD(const SampleSSDParams params): TRTModelBase(params), outputClsSize(params.outputClsSize),
        keepTopK(params.keepTopK),visualThreshold(params.visualThreshold)
    {}

    bool teardown();
};


bool TRTModelSSD::processInput(const trtCommon::BufferManager& buffers)
{
    const int inputC = inputDims.d[0];
    const int inputH = inputDims.d[1];
    const int inputW = inputDims.d[2];
    const int batchSize = mParams.batch_size;

    // Available images
    std::vector<std::string> imageList = {"dog.ppm"};
    mPPMs.resize(batchSize);
    ASSERT(mPPMs.size() <= imageList.size());
    for (int i = 0; i < batchSize; ++i)
    {
        readPPMFile(locateFile(imageList[i], mParams.data_dirs), mPPMs[i]);
    }

    // Fill data buffer
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer("data"));
    float pixelMean[3]{104.0f, 117.0f, 123.0f}; // In BGR order
    // Host memory for input buffer
    for (int i = 0, volImg = inputC * inputH * inputW; i < mParams.batch_size; ++i)
    {
        for (int c = 0; c < inputC; ++c)
        {
            // The color image to input should be in BGR order
            for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
            {
                hostDataBuffer[i * volImg + c * volChl + j] = float(mPPMs[i].buffer[j * inputC + 2 - c]) - pixelMean[c];
            }
        }
    }

    return true;
}
bool TRTModelSSD::processOutput(const trtCommon::BufferManager& buffers)
{
    const int inputH = inputDims.d[1];
    const int inputW = inputDims.d[2];
    const int batchSize = mParams.batch_size;

    const float* detectionOut = static_cast<const float*>(buffers.getHostBuffer("detection_out"));
    const int* keepCount = static_cast<const int*>(buffers.getHostBuffer("keep_count"));

    const std::vector<std::string> classes{"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"}; // List of class labels

    bool pass = true;

    for (int p = 0; p < batchSize; ++p)
    {
        int numDetections = 0;
        // is there at least one correct detection?
        for (int i = 0; i < keepCount[p]; ++i)
        {
            const float* det = detectionOut + (p * keepTopK + i) * 7;
            if (det[2] < visualThreshold)
            {
                continue;
            }
            ASSERT((int) det[1] < outputClsSize);
            std::string storeName = classes[(int) det[1]] + "-" + std::to_string(det[2]) + ".ppm";

            numDetections++;
            std::cout << " Image name:" << mPPMs[p].fileName.c_str()
                             << ", Label: " << classes[(int) det[1]].c_str() << ","
                             << " confidence: " << det[2] * 100.f << " xmin: " << det[3] * inputW
                             << " ymin: " << det[4] * inputH << " xmax: " << det[5] * inputW
                             << " ymax: " << det[6] * inputH << std::endl;

            trtCommon::writePPMFileWithBBox(
                storeName, mPPMs[p], {det[3] * inputW, det[4] * inputH, det[5] * inputW, det[6] * inputH});
        }
        pass &= numDetections >= 1;
    }
    return pass;
}
bool TRTModelSSD::teardown()
{
    nvcaffeparser1::shutdownProtobufLibrary();
    return true;
}

SampleSSDParams initializeParams()
{
    SampleSSDParams params;

    params.data_dirs.push_back("data/SSD/");
    
    params.model_type = ModelType::Caffe;
    params.prototxtFileName = "ssd.prototxt";
    params.file_path = "VGG_VOC0712_SSD_300x300_iter_120000.caffemodel";
    params.inputTensorNames.push_back("data");
    params.batch_size = 1;
    params.outputTensorNames.push_back("detection_out");
    params.outputTensorNames.push_back("keep_count");
    params.engine_path = "../serialize_engine/serialize_engine_output.trt";
    params.load_engine = false;
    params.dla_core = -1;
    params.MaxWorkspaceSize = 960_MiB;
    params.outputClsSize = 21;
    params.keepTopK = 200; // Number of total bboxes to be kept per image after NMS step. It is same as
                           // detection_output_param.keep_top_k in prototxt file
    params.visualThreshold = 0.6f;

    return params;
}

int main()
{
    auto params = initializeParams();
    TRTModelSSD SSD{params};

    std::cout << "Building and running a GPU inference engine for SSD" << std::endl;


    if (!SSD.forward())
    {
        // return print::gLogger.reportFail(sampleTest);
    }
    if(!params.load_engine && !SSD.saveEngine())
    {
        // return print::gLogger.reportFail(sampleTest);
    }
    if (!SSD.teardown())
    {
        // return print::gLogger.reportFail(sampleTest);
    }
    return 0;
}