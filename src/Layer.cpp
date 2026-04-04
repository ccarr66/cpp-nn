#include <cmath>
#include "Layer.h"
#include "ErrorHandler.h"
#include "NeuralNet.h"

#if (DEBUG_NN_STATE_RECORDER)
#include "NNStateRecorder.h"
#define NN_LAYER_DEBUG_START_SUSPEND_RECORD(postStartExpression) \
{\
	if (this->parentNN)\
	{\
		this->parentNN->recorder.startRecording();\
		postStartExpression;\
		this->parentNN->recorder.suspendRecording();\
	}\
}

#define NN_LAYER_DEBUG_END_SUSPEND_RECORD(preEndExpression) \
{\
	if (this->parentNN)\
	{\
		this->parentNN->recorder.resumeRecording();\
		preEndExpression;\
		this->parentNN->recorder.endRecording();\
	}\
}

#define NN_LAYER_DEBUG_RECORD(expression) \
{\
	if (this->parentNN)\
	{\
		this->parentNN->recorder.startRecording();	\
		(expression);	\
		this->parentNN->recorder.endRecording();	\
	}\
}
#define NN_LAYER_DEBUG_SET_RECORD_TARGET(target, info) \
{\
	if (this->parentNN)\
	{\
		this->parentNN->recorder.setTarget(target);\
		this->parentNN->recorder.setLayer(this->layerIdx);\
		this->parentNN->recorder.setOpInfo(info);\
	}\
}
#else
#define NN_LAYER_DEBUG_START_SUSPEND_RECORD(postStartExpression)  
#define NN_LAYER_DEBUG_END_SUSPEND_RECORD(preEndExpression)  
#define NN_LAYER_DEBUG_RECORD(expression) (expression)
#define NN_LAYER_DEBUG_SET_RECORD_TARGET(target, info) 
#endif

const Matrix& Layer::getPrevLayerActivations(size_t layerIdx) const
{
	return ((layerIdx == 0) ? (this->parentNN->InputLayer) : (this->parentNN->Layers[layerIdx-1].Activations));
}

const Matrix& Layer::getLayerComponent(string component) const
{
	#define LCT Layer::LayerComponentTypes
	const Matrix* selectedComponent = nullptr;
	for (size_t layerComp = LCT::eMinLayerComponentTypes; layerComp < LCT::eMaxLayerComponentTypes; layerComp++) {
		if(component == string(layerComponents[layerComp])) {
			switch (layerComp) {
				case LCT::ePrevLayerActivations:
					selectedComponent = &(this->getPrevLayerActivations(this->layerIdx));
					break;
				case LCT::eZ_Activations:
					selectedComponent = &(this->Activations);
					break;
				case LCT::eActivations:
					selectedComponent = &(this->Z_Activations);
					break;
				case LCT::eWeights:
					selectedComponent = &(this->Weights);
					break;
				case LCT::eBiases:
					selectedComponent = &(this->Biases);
					break;
				case LCT::eBiasCosts:
					selectedComponent = &(this->BiasCosts);
					break;
				case LCT::eWeightCosts:
					selectedComponent = &(this->WeightCosts);
					break;
				default:
					ErrorHandler::FatalError(string("Invalid layer component!: ") + std::to_string(layerComp));
					break;
			}
		}
	}
	if (selectedComponent == nullptr)
	{
		ErrorHandler::FatalError(string("Couldn't determine layer component somehow!"));
	}
	return *selectedComponent;
	#undef LCT
}

const Matrix& Layer::getLayerComponent(int component) const
{
	#define LCT Layer::LayerComponentTypes
	const Matrix* selectedComponent = nullptr;
	switch (component) {
		case LCT::ePrevLayerActivations:
			selectedComponent =  &(this->getPrevLayerActivations(this->layerIdx));
			break;
		case LCT::eZ_Activations:
			selectedComponent = &(this->Z_Activations);
			break;
		case LCT::eActivations:
			selectedComponent = &(this->Activations);
			break;
		case LCT::eWeights:
			selectedComponent = &(this->Weights);
			break;
		case LCT::eBiases:
			selectedComponent = &(this->Biases);
			break;
		case LCT::eBiasCosts:
			selectedComponent = &(this->BiasCosts);
			break;
		case LCT::eWeightCosts:
			selectedComponent = &(this->WeightCosts);
			break;
		default:
			ErrorHandler::FatalError(string("Invalid layer component!: ") + std::to_string(component));
			break;
	}

	if (selectedComponent == nullptr)
	{
		ErrorHandler::FatalError(string("Couldn't determine layer component somehow!"));
	}
	return *selectedComponent;
	#undef LCT
}

Layer::Layer()
{
	this->parentNN=nullptr;
	this->Activations = Matrix();
	this->Weights = Matrix();
	this->Biases = Matrix();
}

Matrix* Layer::InitLayer(NeuralNet* nnet, const size_t layerIdx, FileObjHandle& fp)
{
	if (nnet == nullptr)
        ErrorHandler::FatalError(string("NNet is NULL"));

	this->parentNN=nnet;
	this->layerIdx=layerIdx;

	this->Activations = Matrix(fp);
	this->Z_Activations = Matrix(fp);
	this->Weights = Matrix(fp);
	this->Biases = Matrix(fp);
	this->BiasCosts = Matrix(fp);
	this->WeightCosts = Matrix(fp);

	return &(this->Activations);
}

Matrix* Layer::InitLayer(NeuralNet* nnet, const size_t layerIdx, const size_t& layerSize)
{
	size_t prevLayerRows = 0;
	if (nnet == nullptr)
        ErrorHandler::FatalError(string("NNet is NULL"));
		
	prevLayerRows = (layerIdx==0) ? (nnet->InputLayer.getNumElements()) : (nnet->Layers[layerIdx-1].Activations.getNumElements());
	if (prevLayerRows == 0)
        ErrorHandler::FatalError(string("Prev layer len is 0"));

	this->parentNN=nnet;
	this->layerIdx=layerIdx;

	this->Activations = Matrix(layerSize, 1);
	this->Z_Activations = Matrix(layerSize, 1);
	this->Weights = Matrix(layerSize, prevLayerRows);
	this->Biases = Matrix(layerSize, 1);
	this->BiasCosts = Matrix(layerSize, 1);
	this->WeightCosts = Matrix(layerSize, prevLayerRows);

	return &(this->Activations);
}

void Layer::WriteToFile(FileObjHandle& fp) const
{
	this->Activations.WriteToFile(fp);
	this->Z_Activations.WriteToFile(fp);
	this->Weights.WriteToFile(fp);
	this->Biases.WriteToFile(fp);
	this->BiasCosts.WriteToFile(fp);
	this->WeightCosts.WriteToFile(fp);
}

void Layer::setRandDWB(std::uniform_real_distribution<double>* distribution)
{
	NN_LAYER_DEBUG_SET_RECORD_TARGET(nnStateType::eLayer_Weights, nnOpInfo::eSetRandWeights);
	NN_LAYER_DEBUG_RECORD(this->Weights.setRandDMatrix(distribution));
	NN_LAYER_DEBUG_SET_RECORD_TARGET(nnStateType::eLayer_Biases, nnOpInfo::eSetRandBiases);
	NN_LAYER_DEBUG_RECORD(this->Biases.setRandDMatrix(distribution));
}
static std::mutex actNormLock;
static double actNormMax;
static double actNormMin;
double actNorm(const double& val)
{
	return (val - actNormMin)/(actNormMax-actNormMin);
}
void Layer::calculateActivations(double(*activationFunc)(const double&), bool layerNorm)
{
	NN_LAYER_DEBUG_SET_RECORD_TARGET(nnStateType::eLayer_prevLayerActivations, nnOpInfo::eCalcActPrevAct);
	NN_LAYER_DEBUG_START_SUSPEND_RECORD(MatrixAPI::recordAfterOp(this->getPrevLayerActivations(this->layerIdx)));
	NN_LAYER_DEBUG_END_SUSPEND_RECORD(;)
	NN_LAYER_DEBUG_SET_RECORD_TARGET(nnStateType::eLayer_Z_Activations, nnOpInfo::eZAct_WeightsXPrevAct);
	NN_LAYER_DEBUG_RECORD(Matrix::Mult(this->Z_Activations, this->Weights, this->getPrevLayerActivations(this->layerIdx)));
	NN_LAYER_DEBUG_SET_RECORD_TARGET(nnStateType::eLayer_Z_Activations, nnOpInfo::eZAct_AddBiases);
	NN_LAYER_DEBUG_RECORD(this->Z_Activations += this->Biases);
	NN_LAYER_DEBUG_SET_RECORD_TARGET(nnStateType::eLayer_Activations, nnOpInfo::eZAct_ActivationFunc);
	NN_LAYER_DEBUG_RECORD(Matrix::ApplyElementWiseFunc(this->Activations, this->Z_Activations, activationFunc));

	if (layerNorm)
	{
		double min = std::numeric_limits<double>::max();
		double max = std::numeric_limits<double>::lowest();	
		for (auto& elem : this->Activations)
		{
			if (elem < min)
			{
				min = elem;
			}
			if (elem > max)
			{
				max = elem;
			}
		}
		if ((max - min) > 0.000000000001){
			std::lock_guard<std::mutex> lock(actNormLock);
			actNormMax = max;
			actNormMin = min;
			NN_LAYER_DEBUG_SET_RECORD_TARGET(nnStateType::eLayer_Activations, nnOpInfo::eNormAct);
			NN_LAYER_DEBUG_RECORD(Matrix::ApplyElementWiseFunc(this->Activations, this->Activations, actNorm));
			
		}
	}
}

const Matrix& Layer::getActivations() const
{
	return this->Activations;
}

const Matrix& Layer::getZActivations() const
{
	return this->Z_Activations;
}

const Matrix& Layer::getWeights() const
{
	return this->Weights;
}

const Matrix& Layer::getBiases() const
{
	return this->Biases;
}

const Matrix& Layer::getError() const
{
	return this->BiasCosts;
}

const Matrix& Layer::getBiasCosts() const
{
	return this->BiasCosts;
}

const Matrix& Layer::getWeightCosts() const
{
	return this->WeightCosts;
}

size_t Layer::SetWeightsFromCostGradSection(const Matrix& costGrad, size_t offset)
{
	NN_LAYER_DEBUG_SET_RECORD_TARGET(nnStateType::eLayer_Weights, nnOpInfo::eWeightsAddCost);
	NN_LAYER_DEBUG_START_SUSPEND_RECORD(MatrixAPI::recordBeforeOp(this->Weights); MatrixAPI::recordOperandA(costGrad))
	for (size_t layerWInd = 0; layerWInd < this->Weights.getNumElements(); layerWInd++)
	{
		try
		{
			this->Weights.addLinElement(layerWInd, costGrad.getLinElement(offset++));
		}
		catch (...) { ErrorHandler::FatalError("Error accessing WBCostGrad (Weights)"); }
	}
	NN_LAYER_DEBUG_END_SUSPEND_RECORD(MatrixAPI::recordAfterOp(this->Weights))
	return offset;
}

size_t Layer::SetBiasesFromCostGradSection(const Matrix& costGrad, size_t offset)
{
	NN_LAYER_DEBUG_SET_RECORD_TARGET(nnStateType::eLayer_Biases, nnOpInfo::eBiasesAddCost);
	NN_LAYER_DEBUG_START_SUSPEND_RECORD(MatrixAPI::recordBeforeOp(this->Biases); MatrixAPI::recordOperandA(costGrad));
	for (size_t layerBInd = 0; layerBInd < this->Biases.getNumElements(); layerBInd++)
	{
		try
		{
			this->Biases.addLinElement(layerBInd, costGrad.getLinElement(offset++));
		}
		catch (...) { ErrorHandler::FatalError("Error accessing WBCostGrad (Biases)"); }
	}
	NN_LAYER_DEBUG_END_SUSPEND_RECORD(MatrixAPI::recordAfterOp(this->Biases))
	return offset;
}

void Layer::setError(Matrix&& error)
{
	NN_LAYER_DEBUG_SET_RECORD_TARGET(nnStateType::eLayer_BiasCosts, nnOpInfo::eBiasCostFromError);
	NN_LAYER_DEBUG_START_SUSPEND_RECORD(MatrixAPI::recordBeforeOp(this->BiasCosts));
	this->BiasCosts = std::move(error);
	NN_LAYER_DEBUG_END_SUSPEND_RECORD(MatrixAPI::recordAfterOp(this->BiasCosts))

	NN_LAYER_DEBUG_SET_RECORD_TARGET(nnStateType::eLayer_prevLayerActivations, nnOpInfo::eErrorCalcPrevAct);
	NN_LAYER_DEBUG_START_SUSPEND_RECORD(MatrixAPI::recordAfterOp(this->getPrevLayerActivations(this->layerIdx)));
	NN_LAYER_DEBUG_END_SUSPEND_RECORD(;)

	NN_LAYER_DEBUG_SET_RECORD_TARGET(nnStateType::eLayer_WeightCosts, nnOpInfo::eWeightCostEqPrevActLinXBiasCosts);
	NN_LAYER_DEBUG_START_SUSPEND_RECORD(MatrixAPI::recordBeforeOp(this->WeightCosts); MatrixAPI::recordOperandA(this->getPrevLayerActivations(this->layerIdx)); MatrixAPI::recordOperandB(this->BiasCosts));
	for (size_t row = 0; row < this->WeightCosts.getNumRows(); row++)
		for (size_t col = 0; col < this->WeightCosts.getNumCols(); col++)
			this->WeightCosts.setElement(row, col, (this->getPrevLayerActivations(this->layerIdx)).getLinElement(col) * this->BiasCosts.getLinElement(row));
	NN_LAYER_DEBUG_END_SUSPEND_RECORD(MatrixAPI::recordAfterOp(this->WeightCosts))
}
