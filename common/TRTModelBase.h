#pragma once
#include "buffers.h"
#include "common.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <chrono>


enum class ModelType : int
{
    Onnx  = 0,
    Caffe = 1
};


struct ConfigParams
{
    ModelType model_type = ModelType::Onnx;
    std::string file_path = "";
    std::string prototxtFileName  = "";

    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    // Input Output Paths
    std::vector<std::string> data_dirs;
    std::string engine_path = ""; 

    int dla_core = -1;
    bool load_engine = false;
    int batch_size = 1;
    std::size_t MaxWorkspaceSize = 960_MiB;
};

class TRTLogger : public nvinfer1::ILogger 
{
 public:
  void log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept 
  {
    if (severity == Severity::kVERBOSE) 
    {
      return;
    }
    switch (severity) 
    {
      case Severity::kINTERNAL_ERROR:
        std::cerr << "INTERNAL_ERROR: ";
        break;
      case Severity::kERROR:
        std::cerr << "ERROR: ";
        break;
      case Severity::kWARNING:
        std::cerr << "WARNING: ";
        break;
      case Severity::kINFO:
        std::cerr << "INFO: ";
        break;
      case Severity::kVERBOSE:
        std::cerr << "VERBOSE: ";
        break;
      default:
        std::cerr << "UNKNOWN: ";
        break;
    }
    std::cerr << msg << std::endl;
  }
};

class TRTModelBase
{
    template <class T>
    using TRTUniquePtr = std::unique_ptr<T,trtCommon::InferDeleter>;

private:  
    int BATCH_SIZE = 1;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    std::shared_ptr<nvinfer1::ICudaEngine> build();
    std::shared_ptr<nvinfer1::ICudaEngine> buildFromSerializedEngine();
    std::string loadEngine(const std::string& file_path);
    bool engineInitlization();

protected:
    ConfigParams mParams;
    TRTLogger glogger_;
    nvinfer1::Dims inputDims;
    nvinfer1::Dims outputDims;

    virtual bool processInput(const trtCommon::BufferManager& buffers) = 0;

    virtual bool processOutput(const trtCommon::BufferManager& buffers) = 0;

public:
    TRTModelBase(const ConfigParams params)
         : mParams(params),BATCH_SIZE(params.batch_size),mEngine(nullptr)
    {
        if(!engineInitlization())
        {
            std::cout << "Fail to initialize engine "<< std::endl;
        }
    }
    virtual ~TRTModelBase() = default;

    virtual bool forward();
    bool saveEngine();
};



