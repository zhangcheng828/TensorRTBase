#include "TRTModelBase.h"

bool TRTModelBase::engineInitlization()
{
        std::cout << "Building engine . . . "<< std::endl;

        mEngine = mParams.load_engine ? buildFromSerializedEngine():build();

        inputDims = mEngine -> getBindingDimensions(0);
        outputDims = mEngine -> getBindingDimensions(2);
        return true;
}

std::shared_ptr<nvinfer1::ICudaEngine> TRTModelBase::buildFromSerializedEngine()
{
    initLibNvInferPlugins(&glogger_, "");
    std::cout <<"Create ICudaEngine from " << mParams.engine_path << "!" <<std::endl;
    auto trtStream = loadEngine(mParams.engine_path);
    std::cout <<"The size of serialized engine is " << trtStream.size() << "!" <<std::endl;
    TRTUniquePtr<IRuntime> runtime{createInferRuntime(glogger_)};
    std::shared_ptr<nvinfer1::ICudaEngine>  engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(trtStream.data(), trtStream.size()), 
        trtCommon::InferDeleter());
    if (!engine)
    {
        std::cout << "Failed to create engine \n";
        return nullptr;
    }
    return engine;
}

std::shared_ptr<nvinfer1::ICudaEngine> TRTModelBase::build()
{
    initLibNvInferPlugins(&glogger_, "");
    auto builder = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(glogger_));
    if (!builder)
    {
        std::cout << "Builder not created !" << std::endl;
        return nullptr;
    }
    TRTUniquePtr<nvinfer1::INetworkDefinition> network;
    if(mParams.model_type == ModelType::Onnx)
    {
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        network = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    }
    else
    {
        network = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    }
    if (!network)
    {
        std::cout << "Network not created !" << std::endl;
        return nullptr;
    }

    auto config = TRTUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        std::cout << "Config not created !" << std::endl;
        return nullptr;        
    }

    // auto parser = nvonnxparser::createParser(*network, glogger_);
    // auto parsed = parser->parseFromFile(locateFile(mParams.file_path, 
    //                      mParams.data_dirs).c_str(),
    //                 static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
    // if (!parsed)
    // {
    //     std::cout << "Onnx model cannot be parsed ! " << std::endl;
    //     return nullptr;
    // }
    auto parser = TRTUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
    const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor = parser->parse(locateFile(mParams.prototxtFileName, 
                        mParams.data_dirs).c_str(),
                        locateFile(mParams.file_path, mParams.data_dirs).c_str(), *network, DataType::kFLOAT);
    for (auto& s : mParams.outputTensorNames)
    {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }
    builder->setMaxBatchSize(mParams.batch_size);
    config->setMaxWorkspaceSize(mParams.MaxWorkspaceSize);
    trtCommon::enableDLA(builder.get(), config.get(), mParams.dla_core);
    
    
    // CUDA stream used for profiling by the builder.
    auto profileStream = trtCommon::makeCudaStream();
    if (!profileStream)
    {
        std::cout << "ProfileStream not created !" << std::endl;
        return nullptr;
    }
    config->setProfileStream(*profileStream);
    TRTUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        std::cout << "Plan not created !" << std::endl;
        return nullptr;
    }
    
    TRTUniquePtr<IRuntime> runtime{createInferRuntime(glogger_)};
    if (!runtime)
    {
        std::cout << "Runtime not created !" << std::endl;
        return nullptr;
    }
    
    auto engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), trtCommon::InferDeleter());
    if (!engine)
    {
        std::cout << "Engine not created !" << std::endl;
        return nullptr;
    }
    
    return engine;
}

bool TRTModelBase::forward()
{
    // Create RAII buffer manager object
    trtCommon::BufferManager buffers(mEngine);
    // buffers = new trtCommon::BufferManager(mEngine);
    auto context = TRTUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);

    cudaEvent_t start, stop;

    float time = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = mParams.model_type == ModelType::Onnx ? context->executeV2(buffers.getDeviceBindings().data()) :
                context->execute(mParams.batch_size, buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();
    // Verify results
    if (!processOutput(buffers))
    {
        return false;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Inference Time: " << time << " ms"<< std::endl;
    return true;
}


bool TRTModelBase::saveEngine()
{
    if(!mEngine)
    {
        std::cout << "Engine is null. . . "<< std::endl;
        return false;
    }
    std::shared_ptr<IHostMemory> seridata(mEngine->serialize());
    auto data = seridata->data();
    auto length = seridata->size();
    FILE* f = fopen(mParams.engine_path.c_str(), "wb");
    if (!f) return false;


    if (data && length > 0)
    {
        if (fwrite(data, 1, length, f) != length)
        {
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
}
std::string TRTModelBase::loadEngine(const std::string& file_path)
{
    std::ifstream in(file_path, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::string data;
    if (length > 0)
    {
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}