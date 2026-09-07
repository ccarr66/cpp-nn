#pragma once
#include "GLOBALS.h"

#if (DEBUG_NN_STATE_RECORDER)
#include "NNStateCommon.h"
#include "Matrix.h"
#include "Layer.h"
#include <mutex>
#include <fstream>
#include <thread>
#include <filesystem>
#include <map>

enum class nnOpInfo : uint32_t
{
      eNoOp
    , eSetRandInputs
    , eSetRandOutputs
    , eSetOutputCostGrad
    , eResetWBCostGrad
    , eAddWBCostGrad
    , eAvgWBCostGrad
    , eSetDesiredOutputs
    , eSetInputLayer
    , eSetRandWeights
    , eSetRandBiases
    , eZAct_WeightsXPrevAct
    , eZAct_AddBiases
    , eZAct_ActivationFunc
    , eWeightsAddCost
    , eBiasesAddCost
    , eBiasCostFromError
    , eWeightCostEqPrevActLinXBiasCosts
    , eErrorCalcPrevAct
    , eCalcActPrevAct
    , eNormAct
};

string getNNOpInfoStr(nnOpInfo type);

enum flushNeededBits
{
    noFlushNeeded   = 0,
    beforeOp        = BIT(0),    
    afterOp         = BIT(1),
    operandA        = BIT(2),    
    operandB        = BIT(3),
};

class NNStateRecorder
{
private:
    bool recording = false;
    std::filesystem::path fileName;
    nnOpInfo opInfo;
    nnStateType currentTarget; //This might otherwise be names which-part-of-the-NN-are-we-currently-recording
    size_t layerIdx;
    static std::timed_mutex threadRegisterLock;
    static std::map<std::thread::id,NNStateRecorder*> threadRegister; 
    static bool globalAnyRecording;
    std::timed_mutex recorderLock;

    unsigned char flushStatus;
    Matrix beforeOp;
    Matrix afterOp;
    Matrix operandA;
    Matrix operandB;
    bool suspended;
private:
    void flushIfNeeded();
    std::filesystem::path setupNNRecorderDirs() const;
    static NNStateRecorder* getRecorder();
    static void updateGlobalRecordingState();
protected:
    friend class MatrixAPI;
    static void recordEntireOp(const Matrix& beforeOp, const Matrix& afterOp, const Matrix& operandA, const Matrix& operandB);
    static void recordBeforeOp(const Matrix& beforeOp);
    static void recordAfterOp(const Matrix& afterOp); 
    static void recordOperandA(const Matrix& operandA);
    static void recordOperandB(const Matrix& operandB);
public:
    std::filesystem::path getRecordRootPath() const;
    void setNNetFile(const std::filesystem::path&);
    void setOpInfo(nnOpInfo);
    void setTarget(nnStateType);
    void setLayer(size_t);
    void enableRecording();
    void disableRecording();
    void startRecording();
    void suspendRecording();
    void resumeRecording();
    void endRecording();
    NNStateRecorder() = default;
    ~NNStateRecorder();
};

class MatrixAPI : NNStateRecorder
{
public:
    friend class Matrix;
	friend Matrix operator+(const Matrix&, const double&);
	friend Matrix operator+(const Matrix&, const Matrix&);

	friend Matrix operator-(const Matrix&, const double&);
	friend Matrix operator-(const Matrix&, const Matrix&);

	friend Matrix operator*(const Matrix&, const double&);
	friend Matrix operator*(const Matrix&, const Matrix&);

	friend Matrix operator/(const Matrix&, const double&);
	friend Matrix operator/(const Matrix&, const Matrix&);
    //static void recordEntireOp(const Matrix& beforeOp, const Matrix& afterOp, const Matrix& operandA, const Matrix& operandB);
    static void recordBeforeOp(const Matrix& beforeOp);
    static void recordAfterOp(const Matrix& afterOp); 
    static void recordOperandA(const Matrix& operandA);
    static void recordOperandB(const Matrix& operandB);
};

class NNStateRecord
{
public:
    nnOpInfo opInfo = nnOpInfo::eNoOp;
    nnStateType currentTarget = nnStateType::eInvalidTypeLow;
    size_t layerIdx = 0;

    unsigned char flushStatus = 0;
    Matrix beforeOp;
    Matrix afterOp;
    Matrix operandA;
    Matrix operandB;
	NNStateRecord() = default;
    NNStateRecord(FileObjHandle& fp);
};














#endif

