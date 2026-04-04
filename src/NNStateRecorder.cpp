#include "GLOBALS.h"

#if (DEBUG_NN_STATE_RECORDER)
#include "NNStateRecorder.h"
#include "ErrorHandler.h"
#include <filesystem>
#include <iostream>


string getNNOpInfoStr(nnOpInfo type)
{
	string result = string("UNKNOWN");
	switch (type)
	{
        case nnOpInfo::eSetRandInputs:
            result = string("SetRandInputs");
            break;
        case nnOpInfo::eSetRandOutputs:
            result = string("SetRandOutputs");
            break;
        case nnOpInfo::eSetOutputCostGrad:
            result = string("SetOutputCostGrad");
            break;
        case nnOpInfo::eResetWBCostGrad:
            result = string("ResetWBCostGrad");
            break;
        case nnOpInfo::eAddWBCostGrad:
            result = string("AddWBCostGrad");
            break;
        case nnOpInfo::eAvgWBCostGrad:
            result = string("AvgWBCostGrad");
            break;
        case nnOpInfo::eSetDesiredOutputs:
            result = string("SetDesiredOutputs");
            break;
        case nnOpInfo::eSetInputLayer:
            result = string("SetInputLayer");
            break;
        case nnOpInfo::eSetRandWeights:
            result = string("SetRandWeights");
            break;
        case nnOpInfo::eSetRandBiases:
            result = string("SetRandBiases");
            break;
        case nnOpInfo::eZAct_WeightsXPrevAct:
            result = string("ZAct_WeightsXPrevAct");
            break;
        case nnOpInfo::eZAct_AddBiases:
            result = string("ZAct_AddBiases");
            break;
        case nnOpInfo::eZAct_ActivationFunc:
            result = string("ZAct_ActivationFunc");
            break;
        case nnOpInfo::eWeightsAddCost:
            result = string("WeightsAddCost");
            break;
        case nnOpInfo::eBiasesAddCost:
            result = string("BiasesAddCost");
            break;
        case nnOpInfo::eBiasCostFromError:
            result = string("BiasCostFromError");
            break;
        case nnOpInfo::eWeightCostEqPrevActLinXBiasCosts:
            result = string("WeightCostEqPrevActLinXBiasCosts");
            break;
        default:
            break;
	}
    return result;
};

std::timed_mutex NNStateRecorder::threadRegisterLock = {};
std::map<std::thread::id,NNStateRecorder*> NNStateRecorder::threadRegister = {}; 
bool NNStateRecorder::globalAnyRecording = false;

void NNStateRecorder::setOpInfo(nnOpInfo info)
{
    //GP::out << "\nRecorder setup info: " + getNNOpInfoStr(info) << std::endl;
    this->opInfo = info;
}
void NNStateRecorder::setTarget(nnStateType target)
{
    //GP::out << "\nRecorder setup target: " + getNNTypeStr(target) << std::endl;
    this->currentTarget=target;
}

void NNStateRecorder::setLayer(size_t idx)
{
    //GP::out << "\nRecorder setup idx: " + std::to_string(idx) << std::endl;
    this->layerIdx=idx;
}

void NNStateRecorder::setNNetFile(string file)
{
    //GP::out << "\nRecorder setup file: " + file << std::endl;
    this->fileName=file;
}

void NNStateRecorder::flushIfNeeded()
{
    
	auto outputfile = FileObjOutputAppendHandle(HEADER_SETUP(defaultHeaderSetupCallback), this->setupNNRecorderDirs(), BIN_FILE_MODE );
    if (outputfile)
    {
        FileObj::write(outputfile, &(this->opInfo));
        FileObj::write(outputfile, &(this->currentTarget));
        FileObj::write(outputfile, &(this->layerIdx));
        FileObj::write(outputfile, &(NNStateRecorder::flushStatus));
        
        if (NNStateRecorder::flushStatus & flushNeededBits::beforeOp)
        {
            NNStateRecorder::beforeOp.WriteToFile(outputfile);
            CLEAR_BITS(NNStateRecorder::flushStatus, flushNeededBits::beforeOp);
        }

        if (NNStateRecorder::flushStatus & flushNeededBits::afterOp)
        {
            NNStateRecorder::afterOp.WriteToFile(outputfile);
            CLEAR_BITS(NNStateRecorder::flushStatus, flushNeededBits::afterOp);
        }

        if (NNStateRecorder::flushStatus & flushNeededBits::operandA)
        {
            NNStateRecorder::operandA.WriteToFile(outputfile);
            CLEAR_BITS(NNStateRecorder::flushStatus, flushNeededBits::operandA);
        }

        if (NNStateRecorder::flushStatus & flushNeededBits::operandB)
        {
            NNStateRecorder::operandB.WriteToFile(outputfile);
            CLEAR_BITS(NNStateRecorder::flushStatus, flushNeededBits::operandB);
        }
        
        if (NNStateRecorder::flushStatus)
            ErrorHandler::FatalError(string("Didnt flush completely!: " + std::to_string(flushStatus)));
    }
    else
    {
        if (!outputfile)
            ErrorHandler::FatalError(string("Output file not open! "));
        else
            ErrorHandler::FatalError(string("Calling flush on unexpected recorder! "));
    }
}

std::filesystem::path NNStateRecorder::getRecordRootPath() const
{
    std::filesystem::path newDirPath;
    std::filesystem::path filePath(this->fileName);
    std::filesystem::path parentDir = filePath.parent_path();

    // Construct the new path
    newDirPath = parentDir / std::filesystem::path(string("rec_") + filePath.filename().generic_string());
    return newDirPath;
}

std::filesystem::path NNStateRecorder::setupNNRecorderDirs() const
{
    std::filesystem::path newDirPath;
    try 
    {
        // Construct the new path
        newDirPath = this->getRecordRootPath() / getNNTypeStr(this->currentTarget);
        
        // Check if the directories exist, and create them if necessary
        if (!std::filesystem::exists(newDirPath)) 
        {
            std::filesystem::create_directories(newDirPath);
            GP::out << "\nDirectories created: " << newDirPath << std::endl;
        } 
        //else 
        //{
        //    GP::out << "\nDirectories already exist: " << newDirPath << std::endl;
        //}

        if ( isLayerType(this->currentTarget) )
            newDirPath = newDirPath / std::filesystem::path(string("out") + std::to_string(this->layerIdx) + string(".bin"));
        else
            newDirPath = newDirPath / std::filesystem::path("out.bin");

    } 
    catch (const std::filesystem::filesystem_error& e) 
    {
        ErrorHandler::FatalError(string("Filesystem error: ") + string(e.what()));
    } 
    catch (const std::exception& e) 
    {
        ErrorHandler::FatalError(string("Error: ") + string(e.what()));
    }
    return newDirPath;
}

void NNStateRecorder::recordEntireOp(const Matrix& beforeOp, const Matrix& afterOp, const Matrix& operandA, const Matrix& operandB)
{
    auto recorder = NNStateRecorder::getRecorder(); 
    if (recorder)
    {
        recorder->beforeOp = Matrix(beforeOp);  
        recorder->flushStatus |= flushNeededBits::beforeOp;
        recorder->afterOp = Matrix(afterOp);    
        recorder->flushStatus |= flushNeededBits::afterOp;
        recorder->operandA = Matrix(operandA);  
        recorder->flushStatus |= flushNeededBits::operandA;
        recorder->operandB = Matrix(operandB);  
        recorder->flushStatus |= flushNeededBits::operandB; 
    }     
}

void NNStateRecorder::recordBeforeOp(const Matrix& beforeOp)
{
    auto recorder = NNStateRecorder::getRecorder(); 
    if (recorder)
    {
        recorder->beforeOp = Matrix(beforeOp); 
        recorder->flushStatus |= flushNeededBits::beforeOp;
    }
}

void NNStateRecorder::recordAfterOp(const Matrix& afterOp)
{
    auto recorder = NNStateRecorder::getRecorder(); 
    if (recorder)
    {
        recorder->afterOp = Matrix(afterOp); 
        recorder->flushStatus |= flushNeededBits::afterOp;
    }
}

void NNStateRecorder::recordOperandA(const Matrix& operandA)
{
    auto recorder = NNStateRecorder::getRecorder(); 
    if (recorder)
    {
        recorder->operandA = Matrix(operandA); 
        recorder->flushStatus |= flushNeededBits::operandA;
    }
}

void NNStateRecorder::recordOperandB(const Matrix& operandB)
{
    auto recorder = NNStateRecorder::getRecorder(); 
    if (recorder)
    {
        recorder->operandB = Matrix(operandB); 
        recorder->flushStatus |= flushNeededBits::operandB;
    }
}

void NNStateRecorder::enableRecording()
{
    this->recording = true;    
}

void NNStateRecorder::disableRecording()
{
    this->recording = false;
}

void NNStateRecorder::startRecording()
{
    if (this->recorderLock.try_lock_for(STANDARD_TIMEOUT)) 
    {
        if (this->recording)
        {
            const auto thread = std::this_thread::get_id(); // Dynamically get the current thread ID

            if (NNStateRecorder::threadRegisterLock.try_lock_for(STANDARD_TIMEOUT))
            {
                if (!NNStateRecorder::globalAnyRecording)
                {       
                    //lets us skip everything bc we know we arent recording anything
                }
                else if (NNStateRecorder::threadRegister.find(thread) == NNStateRecorder::threadRegister.end()) 
                {
                    NNStateRecorder::threadRegister[thread]  = this;
                } 
                else 
                {
                    ErrorHandler::FatalError(string("Already Recording on this thread!"));
                }
                (void)NNStateRecorder::updateGlobalRecordingState();
                NNStateRecorder::threadRegisterLock.unlock();
            } 
            else 
            {
                ErrorHandler::FatalError(string("Couldn't acquire thread register!"));
            }

            NNStateRecorder::flushStatus = flushNeededBits::noFlushNeeded;
        } 
        this->recorderLock.unlock();
    } 
    else 
    {
        ErrorHandler::FatalError(string("Couldn't acquire recorder!"));
    }
   
}


void NNStateRecorder::endRecording()
{
    if (this->recorderLock.try_lock_for(STANDARD_TIMEOUT)) 
    {
        if (this->recording)
        {
            const auto thread = std::this_thread::get_id(); // Dynamically get the current thread ID

            if (NNStateRecorder::threadRegisterLock.try_lock_for(STANDARD_TIMEOUT))
            {
                if (!NNStateRecorder::globalAnyRecording)
                {       
                    //lets us skip everything bc we know we arent recording anything
                }
                else if (NNStateRecorder::threadRegister.find(thread) == NNStateRecorder::threadRegister.end()) 
                {
                    ErrorHandler::SoftError(string("Hey wasn't sure if this would happen, you managed to turn on recording on this neural network in the middle of a startRecording/endRecording pair"));
                    //no need to do anything, we will not have had valid calculations to flush
                }
                //at this point we know we have turned on NN State recording & know we have a threadRegister entry for the thread we are currently executing in
                //The threadRegistry is supposed to have ptrs to NNStateRecorder for each thread, so threadRegistry[current thread]
                //should point to exact same object on which endRecording is being called, e.g. 'this'. 
                //Now, iff 'this' is still in the same thread it was registered in, which must also be the same thread we are executing endRecording in,
                //we have no problem. Otherwise, fatal error, need to debug...
                else if (NNStateRecorder::threadRegister[thread] != this)
                {
                    ErrorHandler::FatalError(string("endRecording called on NNStateRecorder object from the wrong thread"));
                }
                else
                {
                    this->flushIfNeeded();
                    this->flushStatus = flushNeededBits::noFlushNeeded;
                    NNStateRecorder::threadRegister.erase(NNStateRecorder::threadRegister.find(thread));
                }

                (void)NNStateRecorder::updateGlobalRecordingState();
                NNStateRecorder::threadRegisterLock.unlock();
            } 
            else 
            {
                ErrorHandler::FatalError(string("Couldn't acquire thread register!"));
            }
            
            NNStateRecorder::flushStatus = flushNeededBits::noFlushNeeded;
        } 
        NNStateRecorder::recorderLock.unlock();
    } 
    else 
    {
        ErrorHandler::FatalError(string("Couldn't acquire recorder!"));
    }
}

//please only call in updateGlobalRecordingState locked context, this is just for readability
void NNStateRecorder::updateGlobalRecordingState()
{
    NNStateRecorder::globalAnyRecording = !NNStateRecorder::threadRegister.empty(); 
}

NNStateRecorder* NNStateRecorder::getRecorder()
{
    const auto thread = std::this_thread::get_id(); // Dynamically get the current thread ID
    NNStateRecorder* retRecorder=nullptr;

    if (NNStateRecorder::threadRegister.size() > 0 && NNStateRecorder::threadRegisterLock.try_lock_for(STANDARD_TIMEOUT))
    {
        if (NNStateRecorder::threadRegister.find(thread) == NNStateRecorder::threadRegister.end()) 
        {
            //recording not started on thread
        } 
        else 
        {
            retRecorder = NNStateRecorder::threadRegister[thread];
            
            if(retRecorder == nullptr)
            {
                ErrorHandler::FatalError(string("Thread Register left in invalid state!"));
            }
        }        
        
        NNStateRecorder::threadRegisterLock.unlock();
    } 
    else if (NNStateRecorder::threadRegister.size() > 0)
    {
        ErrorHandler::FatalError(string("Couldn't acquire thread register!"));
    }
    
    if (retRecorder && (!retRecorder->recording || retRecorder->suspended))
    {
        //if we arent recording we should abort any attempt to access the recorder from the MatrixAPI
        retRecorder = nullptr;
    }
    return retRecorder;
}

NNStateRecorder::~NNStateRecorder()
{
    NNStateRecorder::endRecording();
}

void NNStateRecorder::suspendRecording()
{
	return (void)(NNStateRecorder::suspended = true);
}

void NNStateRecorder::resumeRecording()
{
	return (void)(NNStateRecorder::suspended = false);
}

void MatrixAPI::recordBeforeOp(const Matrix& beforeOp)
{
    NNStateRecorder::recordBeforeOp(beforeOp);
}

void MatrixAPI::recordAfterOp(const Matrix& afterOp)
{
    NNStateRecorder::recordAfterOp(afterOp);
}

void MatrixAPI::recordOperandA(const Matrix& operandA)
{
    NNStateRecorder::recordOperandA(operandA);
}

void MatrixAPI::recordOperandB(const Matrix& operandB)
{
    NNStateRecorder::recordOperandB(operandB);
}


NNStateRecord::NNStateRecord(FileObjHandle& fp)
{
    FileObj::read(fp, &this->opInfo);
    FileObj::read(fp, &this->currentTarget);
    FileObj::read(fp, &this->layerIdx);
    FileObj::read(fp, &this->flushStatus);
    if (this->flushStatus & flushNeededBits::beforeOp)
    {
        this->beforeOp = Matrix(fp);
    }

    if (this->flushStatus & flushNeededBits::afterOp)
    {
        this->afterOp = Matrix(fp);
    }

    if (this->flushStatus & flushNeededBits::operandA)
    {
        this->operandA = Matrix(fp);
    }

    if (this->flushStatus & flushNeededBits::operandB)
    {
        this->operandB = Matrix(fp);
    }
}

#endif

