#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>

#include "logger.h"
#include "parserOnnxConfig.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "common.h"
using namespace nvinfer1;

static const int INPUT_H = 32;// 32;//Since the CIFAR-10 dataset is 32 X 32 X 3 image.....................
static const int INPUT_W = 32;// 32;
static const int OUTPUT_SIZE = 10;// 10;
Logger gLogger;
//gLogger.getTRTLogger()
static int gUseDLACore{-1};

void onnxToTRTModel(const std::string& modelFile, // name of the onnx model
                    unsigned int maxBatchSize,    // batch size - NB must be at least as large as the batch we want to run with
                    IHostMemory*& trtModelStream) // output buffer for the TensorRT model
{
    int verbosity = (int) nvinfer1::ILogger::Severity::kWARNING;

    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);//gLogger is the obj of Logger class...
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    auto parser = nvonnxparser::createParser(*network, gLogger);

   
    parser->reportParsingInfo();

    if (!parser->parseFromFile("densenet_new.onnx").c_str(), verbosity))
    {
        string msg("failed to parse onnx file");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);

    samplesCommon::enableDLA(builder, gUseDLACore);
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we can destroy the parser
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();
    engine->destroy();
    network->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex, outputIndex;
    for (int b = 0; b < engine.getNbBindings(); ++b)
    {
        if (engine.bindingIsInput(b))
            inputIndex = b;
        else
            outputIndex = b;
    }

    // create GPU buffers and a stream
     CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
     CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

     cudaStream_t stream;
     CHECK(cudaStreamCreate(&stream));

     // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
     CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
     context.enqueue(batchSize, buffers, stream, nullptr);
     CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
     cudaStreamSynchronize(stream);
     // release the stream and the buffers
     cudaStreamDestroy(stream);
     CHECK(cudaFree(buffers[inputIndex]));
     CHECK(cudaFree(buffers[outputIndex]));
}




int main(int argc, char** argv)
{   
    gUseDLACore = samplesCommon::parseDLA(argc, argv);
    // create a TensorRT model from the onnx model and serialize it to a stream
    IHostMemory* trtModelStream{nullptr};
    onnxToTRTModel("densenet_new.onnx", 1, trtModelStream);
    //cout<<"Done Converting to ONNX model to TRT Model da Dineshuuuuuuuuuuu";
    assert(trtModelStream != nullptr);


    srand(unsigned(time(nullptr)));
    uint8_t fileData[INPUT_H * INPUT_W];
	int num = 3;// rand() % 10;
    
    float data[INPUT_H * INPUT_W];
    for (int i = 0; i < INPUT_H * INPUT_W; i++)
        data[i] = 1.0 - float(fileData[i] / 255.0);

    // deserialize the engine
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    if (gUseDLACore >= 0)
    {
        runtime->setDLACore(gUseDLACore);
    }

    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);


    // run inference
    float prob[OUTPUT_SIZE];
    doInference(*context, data, prob, 1);


    //After running the inference we still don'y need the engine to be running..we can destroy it manually..
    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    float val{0.0f};
    int idx{0};

    //Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        prob[i] = exp(prob[i]);
        sum += prob[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        prob[i] /= sum;
        val = std::max(val, prob[i]);
        if (val == prob[i])
            idx = i;

        std::cout << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << prob[i] << " "
                  << "Class " << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << std::endl;
    }
    std::cout << std::endl;

    return (idx == num && val > 0.9f) ? EXIT_SUCCESS : EXIT_FAILURE;
}
