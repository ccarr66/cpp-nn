#include "NeuralNet.h"
#include <cmath>
#include <iostream>
#include <numeric>
#include "ErrorHandler.h"
#include "GlobalRand.h"

#if (DEBUG_NN_STATE_RECORDER)
#define NN_DEBUG_START_SUSPEND_RECORD(postStartExpression) this->recorder.startRecording(); postStartExpression; this->recorder.suspendRecording();
#define NN_DEBUG_END_SUSPEND_RECORD(preEndExpression) this->recorder.resumeRecording(); preEndExpression; this->recorder.endRecording();
#define NN_DEBUG_RECORD(expression)\
{                           	\
    this->recorder.startRecording();  	\
	(expression); 			 	\
    this->recorder.endRecording();	 	\
}
#define NN_DEBUG_RECORD_SETUP(target, info) (this->recorder.setTarget(target), this->recorder.setOpInfo(info))
#else
#define NN_DEBUG_START_SUSPEND_RECORD(postStartExpression)  
#define NN_DEBUG_END_SUSPEND_RECORD(preEndExpression)  
#define NN_DEBUG_RECORD(expression) (expression)
#define NN_DEBUG_RECORD_SETUP(target, info) 
#endif

const std::vector<const char *> ActivationFunctionNames = { "Sigmoid", "ReLU" };
const std::vector<const char *> CostFunctionNames = { "Quadratic" };


//NeuralNetLabState_t//////////////////////////////////////


#if (DEBUG_NN_STATE_RECORDER)	
size_t	NeuralLabViewer_t::getRecordIdx()
{
	if (this->recordStore.size() > 0)
	{
		this->recordIdx %= this->recordStore.size();
	}
	else
	{
		this->recordIdx = 0;
		ErrorHandler::FatalError(string("This shouldnt happen! recordIdx = 0"));
	}

	return  (this->recordIdx);
}
#endif

///////////////////////////////////////////////////////////

//NeuralNetProcessLifetimeManager//////////////////////////
NeuralNet::NeuralNetProcessLifetimeManager::NeuralNetProcessLifetimeManager(NeuralNet* parent)
{
	this->parent=parent;
}

NeuralNet::NeuralNetProcessLifetimeManager::~NeuralNetProcessLifetimeManager()
{
	this->parent->stopProcess(NeuralNet::ProcessingTypes::Any);
}

///////////////////////////////////////////////////////////

const Matrix& NeuralNet::getNNComponent(string component) const
{
	std::shared_lock lock(this->accessMutex);

	#define NCT NeuralNet::NNComponentTypes
	const Matrix* selectedComponent = nullptr;
	for (size_t comp = NCT::eMinNNComponentTypes; comp < NCT::eMaxNNComponentTypes; comp++) {
		if(component == string(this->NNComponents[comp])) {
			switch (comp) {
				case NCT::eDesiredOutputs:
					selectedComponent = &(this->DesiredOutputs);
					break;
				case NCT::eInputLayer:
					selectedComponent = &(this->InputLayer);
					break;
				case NCT::eOutputCostGrad:
					selectedComponent = &(this->OutputCostGrad);
					break;
				case NCT::eWBCostGrad:
					selectedComponent = &(this->WBCostGrad);
					break;
				default:
					ErrorHandler::FatalError(string("Invalid NN component!: ") + std::to_string(comp));
					break;
			}
		}
	}
	if (selectedComponent == nullptr)
	{
		ErrorHandler::FatalError(string("Couldn't determine NN component somehow!"));
	}
	return *selectedComponent;
	#undef NCT
}

const Matrix& NeuralNet::getNNComponent(int component) const
{
	std::shared_lock lock(this->accessMutex);
	#define NCT NeuralNet::NNComponentTypes
	const Matrix* selectedComponent = nullptr;
	switch (component) {
		case NCT::eDesiredOutputs:
			selectedComponent = &(this->DesiredOutputs);
			break;
		case NCT::eInputLayer:
			selectedComponent = &(this->InputLayer);
			break;
		case NCT::eOutputCostGrad:
			selectedComponent = &(this->OutputCostGrad);
			break;
		case NCT::eWBCostGrad:
			selectedComponent = &(this->WBCostGrad);
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
	#undef NCT
}

void NeuralNet::verifyFiles()
{
	std::unique_lock lock(this->accessMutex);

	string localTrainLblDataFP = this->TrainingData.labelFile.string();
	correctFilePathForTarget(localTrainLblDataFP);
	auto lcltrainLblDataFP = FileObjInputHandle(HEADER_NOT_NEEDED, localTrainLblDataFP, BIN_FILE_MODE);

	string localTrainImgDataFP = this->TrainingData.imageFile.string();
	correctFilePathForTarget(localTrainImgDataFP);
	auto lcltrainImgDataFP = FileObjInputHandle(HEADER_NOT_NEEDED, localTrainImgDataFP, BIN_FILE_MODE);

	string localTestLblDataFP = this->TestData.labelFile.string();
	correctFilePathForTarget(localTestLblDataFP);
	auto lcltestLblDataFP = FileObjInputHandle(HEADER_NOT_NEEDED, localTestLblDataFP, BIN_FILE_MODE);

	string localTestImgDataFP = this->TestData.imageFile.string();
	correctFilePathForTarget(localTestImgDataFP);
	auto lcltestImgDataFP = FileObjInputHandle(HEADER_NOT_NEEDED, localTestImgDataFP, BIN_FILE_MODE);

	uint32_t trainLbl_MagicNumber, trainImg_MagicNumber;
	uint32_t testLbl_MagicNumber, testImg_MagicNumber;

	FileObj::read(lcltrainLblDataFP, &trainLbl_MagicNumber);
	FileObj::read(lcltrainImgDataFP, &trainImg_MagicNumber);
	FileObj::read(lcltestLblDataFP, &testLbl_MagicNumber);
	FileObj::read(lcltestImgDataFP, &testImg_MagicNumber);

	if(trainLbl_MagicNumber != testLbl_MagicNumber || trainImg_MagicNumber != testImg_MagicNumber)
		ErrorHandler::FatalError("\nTraining and Test files incompatible: MAGICNUM");

	uint8_t lblDataType = 0x0F & (trainLbl_MagicNumber >> 16);
	uint8_t imgDataType = 0x0F & (trainImg_MagicNumber >> 16);

	if (lblDataType != imgDataType)
		ErrorHandler::FatalError("\nTraining and Test files incompatible: DATATYPE");

	uint8_t lblDimNum = 0x0F & (trainLbl_MagicNumber >> 24);
	uint8_t imgDimNum = 0x0F & (trainImg_MagicNumber >> 24);

	if (lblDimNum != 1 || imgDimNum != 3)
		ErrorHandler::FatalError("\nTraining or Test files incompatible: DIMNUM");

	//implement SFINAE later or find alt solution
	switch (lblDataType)
	{
	case 0x08:
	case 0x09:
		this->inputTypeSize = 1;
		break;
	case 0x0B:
		this->inputTypeSize = 2;
		break;
	case 0x0C:
	case 0x0D:
		this->inputTypeSize = 4;
		break;
	case 0x0E:
		this->inputTypeSize = 8;
		break;
	default:
		ErrorHandler::FatalError("\nInvalid datatype");
	}

	lcltrainImgDataFP.validSeekCur(4);
	lcltestImgDataFP.validSeekCur(4);

	uint32_t trainRows, testRows;
	uint32_t trainCols, testCols;

	lcltrainImgDataFP.enableEndSwap(); FileObj::read(lcltrainImgDataFP, &trainRows); lcltrainImgDataFP.disableEndSwap();
	lcltrainImgDataFP.enableEndSwap(); FileObj::read(lcltrainImgDataFP, &trainCols); lcltrainImgDataFP.disableEndSwap();
	lcltestImgDataFP.enableEndSwap(); FileObj::read(lcltestImgDataFP, &testRows); lcltestImgDataFP.disableEndSwap();
	lcltestImgDataFP.enableEndSwap(); FileObj::read(lcltestImgDataFP, &testCols); lcltestImgDataFP.disableEndSwap();

	if (trainRows != testRows || trainCols != testCols)
		ErrorHandler::FatalError("\nTraining or Test files incompatible: IMG SIZE");

	this->img_rows = trainRows;
	this->img_cols = trainCols;

	lcltrainLblDataFP.close();
	lcltrainImgDataFP.close();
	lcltestLblDataFP.close();
	lcltestImgDataFP.close();
}


std::filesystem::path NeuralNet::getFileName() const
{
	std::shared_lock lock(this->accessMutex);
	return (this->FileName);
}


void NeuralNet::setFileName(const std::filesystem::path& newName)
{
	std::unique_lock lock(this->accessMutex);
	this->FileName = newName;
}

const Matrix& NeuralNet::getInputLayer() const
{
	std::shared_lock lock(this->accessMutex);
	return (this->InputLayer);
}

void NeuralNet::getInputLayerDims(int& width, int& height) const
{
	std::shared_lock lock(this->accessMutex);
	width = static_cast<int>(this->img_cols);
	height = static_cast<int>(this->img_rows);
	return;
}

const std::vector<Layer>& NeuralNet::getLayers() const
{
	std::shared_lock lock(this->accessMutex);
	return this->Layers;
} 

const Matrix& NeuralNet::getOutputActivations(int& width, int& height) const
{
	std::shared_lock lock(this->accessMutex);
	width=static_cast<int>(this->Layers.back().getActivations().getNumCols());
	height=static_cast<int>(this->Layers.back().getActivations().getNumRows());
	return (this->Layers.back().getActivations());
}

const Matrix& NeuralNet::getOutputCostGrad(int& width, int& height) const
{
	
	std::shared_lock lock(this->accessMutex);
	width=static_cast<int>(this->OutputCostGrad.getNumCols());
	height=static_cast<int>(this->OutputCostGrad.getNumRows());
	return (this->OutputCostGrad);
}

const Matrix& NeuralNet::getDesiredOutputs(int& width, int& height) const
{
	std::shared_lock lock(this->accessMutex);
	width=static_cast<int>(this->DesiredOutputs.getNumCols());
	height=static_cast<int>(this->DesiredOutputs.getNumRows());
	return (this->DesiredOutputs);
}

void NeuralNet::enableLayerNorm()
{
	std::unique_lock lock(this->accessMutex);
	this->layerNorm = true;
}

void NeuralNet::disableLayerNorm()
{
	std::unique_lock lock(this->accessMutex);
	this->layerNorm = false;
}


void NeuralNet::setActFunction(ActivationFunction af)
{
	std::unique_lock lock(this->accessMutex);
	this->af = af;
	switch (af)
	{
	case ActivationFunction::ReLU:
		this->ActFunc = NeuralNet::ReLU;
		this->ActFuncDeriv = NeuralNet::ReLU_deriv;
		break;
	default: //sigmoid is default
		this->ActFunc = NeuralNet::sigmoid;
		this->ActFuncDeriv = NeuralNet::sigmoid_deriv;
		break;
	}
}

void NeuralNet::setCostFunction(CostFunction cf)
{
	std::unique_lock lock(this->accessMutex);
	this->cf = cf;
	//switch (cf)
	{
	//default:
		this->CostFunc = NeuralNet::quadratic;
		this->CostFuncPartialDeriv = NeuralNet::quadratic_partialderiv;
	}
}

double NeuralNet::calcActFunc(const double& input)
{
	return this->ActFunc(input);
}

double NeuralNet::calcActFuncDeriv(const double& input)
{	
	return this->ActFuncDeriv(input);
}

//double NeuralNet::calcCostFunc(const double& input)
//{
//	return this->CostFunc(input);
//}


void NeuralNet::setRandDInputs()
{
	std::unique_lock lock(this->accessMutex);
	NN_DEBUG_RECORD_SETUP(nnStateType::eNonLayer_InputLayer, nnOpInfo::eSetRandInputs);
	NN_DEBUG_RECORD(this->InputLayer.setRandDMatrix(&(this->lclRandDistrbution)));
}

void NeuralNet::setRandWBAcrossNetwork()
{
	std::unique_lock lock(this->accessMutex);
	for (auto& layer : this->Layers)
		layer.setRandDWB(&(this->lclRandDistrbution));

	GP::out << "\nWB Randomization just performed!" << std::endl;
}

void NeuralNet::setRandDOutputs()
{
	std::unique_lock lock(this->accessMutex);
	NN_DEBUG_RECORD_SETUP(nnStateType::eNonLayer_DesiredOutputs, nnOpInfo::eSetRandOutputs);
	NN_DEBUG_RECORD(this->DesiredOutputs.setRandDMatrix(&(this->lclRandDistrbution)));
}

void NeuralNet::setTrainingParams(const size_t& batchsz, const int& numbatches, const double& eta)
{
	std::unique_lock lock(this->accessMutex);
	if (isValidBatchSize(batchsz))
		this->BatchSize = batchsz;
	else
		ErrorHandler::SoftError("setTrainingParams invalid batchsz"); //note these should probably be handled differently if this ever happens

	if (isValidBatchCount(numbatches))
		this->NumBatches = numbatches;
	else
		ErrorHandler::SoftError("setTrainingParams invalid numbatches"); //note these should probably be handled differently if this ever happens

	if (isValidEta(eta))
		this->Eta = eta;
	else
		ErrorHandler::SoftError("setTrainingParams invalid eta"); //note these should probably be handled differently if this ever happens
}

void NeuralNet::updateNetwork()
{
	std::unique_lock lock(this->accessMutex);
	for (auto& layer : this->Layers)
		layer.calculateActivations(this->ActFunc, this->layerNorm);
}

void NeuralNet::computeBackpropErrors()
{
	std::unique_lock lock(this->accessMutex);
	NN_DEBUG_RECORD_SETUP(nnStateType::eNonLayer_OutputCostGrad, nnOpInfo::eSetOutputCostGrad);
	NN_DEBUG_START_SUSPEND_RECORD(MatrixAPI::recordBeforeOp(this->OutputCostGrad));
	this->OutputCostGrad = this->CostFuncPartialDeriv(this->DesiredOutputs, this->Layers.back().getActivations());
	NN_DEBUG_END_SUSPEND_RECORD(MatrixAPI::recordAfterOp(this->OutputCostGrad));
	

	auto nextlayer = this->Layers.rbegin();
	for (auto layer = this->Layers.rbegin(); layer != this->Layers.rend(); ++layer)
	{
		if (layer == this->Layers.rbegin())
			(*layer).setError(this->OutputCostGrad * Matrix::ApplyElementWiseFunc((*layer).getZActivations(), this->ActFuncDeriv));
		else
			(*layer).setError(Matrix::Mult((*nextlayer).getWeights(), (*nextlayer).getError(), MatCond::Transpose) * Matrix::ApplyElementWiseFunc((*layer).getZActivations(), this->ActFuncDeriv));

		nextlayer = layer;
	}
}

void NeuralNet::addCost()
{
	std::unique_lock lock(this->accessMutex);
#if (FP_EXCEPTIONS_ENABLED)
    try 
#endif
	{
		this->Cost += double{ this->CostFunc(this->DesiredOutputs, this->Layers.back().getActivations()) };
	} 
#if (FP_EXCEPTIONS_ENABLED)
    catch (...) {
        ErrorHandler::SoftError(string("Floating-point exception occurred in addCost!"));
    }
#endif
}

double NeuralNet::getCost() const
{
	std::shared_lock lock(this->accessMutex);

	return double{ this->CostFunc(this->DesiredOutputs, this->Layers.back().getActivations()) };
}

size_t NeuralNet::getCompletedBatches() const
{
	std::shared_lock lock(this->accessMutex);
	return this->BatchesCompleted;
}

size_t NeuralNet::getBatchImgSize() const
{
	std::shared_lock lock(this->accessMutex);
	return this->InputLayer.getNumRows();
} 

size_t NeuralNet::getBatchLblSize() const
{
	std::shared_lock lock(this->accessMutex);
	return 1;
}

std::filesystem::path NeuralNet::testingRecordFilename()
{
	return std::filesystem::path("testingResults.bin");
}

std::filesystem::path NeuralNet::trainingRecordFilename()
{
	return std::filesystem::path("trainingResults.bin");
}


void NeuralNet::avgCost(const size_t& numTrainings)
{
	std::unique_lock lock(this->accessMutex);
#if (FP_EXCEPTIONS_ENABLED)
    try 
#endif
	{
		this->Cost /= static_cast<double>(numTrainings);
	} 
#if (FP_EXCEPTIONS_ENABLED)
    catch (...) {
        ErrorHandler::SoftError(string("Floating-point exception occurred in avgCost!"));
    }
#endif
}

void NeuralNet::resetCost()
{
	std::unique_lock lock(this->accessMutex);
	this->Cost = 0;
}

void NeuralNet::resetWBCostGrad()
{
	std::unique_lock lock(this->accessMutex);
	NN_DEBUG_RECORD_SETUP(nnStateType::eNonLayer_WBCostGrad, nnOpInfo::eResetWBCostGrad);
	NN_DEBUG_RECORD(this->WBCostGrad.setElementsToZero());
}

void NeuralNet::addWBCosts()
{
	std::unique_lock lock(this->accessMutex);
	NN_DEBUG_RECORD_SETUP(nnStateType::eNonLayer_WBCostGrad, nnOpInfo::eAddWBCostGrad);
	NN_DEBUG_START_SUSPEND_RECORD(MatrixAPI::recordBeforeOp(this->WBCostGrad));
	auto wbc_ind = size_t{ 0 };
	for (const auto& layer : this->Layers)
	{
		for (const auto& val : layer.getWeightCosts())
			this->WBCostGrad.addLinElement(wbc_ind++, val);
		for (const auto& val : layer.getBiasCosts())
			this->WBCostGrad.addLinElement(wbc_ind++, val);
	}
	NN_DEBUG_END_SUSPEND_RECORD(MatrixAPI::recordAfterOp(this->WBCostGrad));
}

void NeuralNet::avgWBCosts(const size_t& numTrainings, const double overrideEta)
{
	std::unique_lock lock(this->accessMutex);
	double nT = static_cast<double>(numTrainings);
	double learning = 0.0;
	double eta = (isValidEta(overrideEta) ? overrideEta : this->Eta);
	
#if (FP_EXCEPTIONS_ENABLED)
    try 
#endif
	{
		learning = (-1 * eta / nT);
	} 
#if (FP_EXCEPTIONS_ENABLED)
    catch (...) {
        ErrorHandler::SoftError(string("Floating-point exception occurred in addCost!"));
    }
#endif

	NN_DEBUG_RECORD_SETUP(nnStateType::eNonLayer_WBCostGrad, nnOpInfo::eAvgWBCostGrad);
	NN_DEBUG_RECORD(this->WBCostGrad *= learning);
}

void NeuralNet::applyGrad()
{
	std::unique_lock lock(this->accessMutex);
	size_t wbInd = { 0 };
	for (auto& layer : this->Layers)
	{
		wbInd = layer.SetWeightsFromCostGradSection(this->WBCostGrad, wbInd);
		wbInd = layer.SetBiasesFromCostGradSection(this->WBCostGrad, wbInd);
	}
}

void NeuralNet::printInputs() const
{
	std::shared_lock lock(this->accessMutex);
	GP::out << "\n";//Inputs:\n" << this->InputLayer.toString('[', ']', ' ', ';');
	size_t ind = 0;
	for (size_t r = 0; r < this->img_rows; r++)
	{
		for (size_t c = 0; c < this->img_cols; c++)
		{
			if (this->InputLayer.getLinElement(ind++) > 0)
				GP::out << '+';
			else
				GP::out << ' ';
		}
		GP::out << '\n';
	}
	GP::out << std::endl;
}

void NeuralNet::printActivations() const
{
	std::shared_lock lock(this->accessMutex);
	auto layerID = size_t{ 0 };
	for (const auto& layer : this->Layers)
		GP::out << "\nActivations: " << layerID++ << "\n" << layer.getActivations().toString('[', ']', ' ', ';')  << std::endl;
}

void NeuralNet::printZActivations() const
{
	std::shared_lock lock(this->accessMutex);
	auto layerID = size_t{ 0 };
	for (const auto& layer : this->Layers)
		GP::out << "\nZActivations: " << layerID++ << "\n" << layer.getZActivations().toString('[', ']', ' ', ';')  << std::endl;
}

void NeuralNet::printLayers() const
{
	std::shared_lock lock(this->accessMutex);
	auto layerID = size_t{ 0 };
	for (const auto& layer : this->Layers)
	{
		GP::out << "\nLayer: " << layerID++ << "\nWeights:";
		GP::out << layer.getWeights().toString('[', ']', ' ', ';');
		GP::out << "\nBiases: ";
		GP::out << layer.getBiases().toString('[', ']', ' ', ';');
	}
	GP::out << std::endl;
}

void NeuralNet::printOutputs() const
{
	std::shared_lock lock(this->accessMutex);
	GP::out << "\nOutputs:\n" << this->DesiredOutputs.toString('[', ']', ' ', ';') << std::endl;
}

void NeuralNet::printCost() const
{
	std::shared_lock lock(this->accessMutex);
	GP::out << "\nCost: " << this->Cost << std::endl;
}

void NeuralNet::printOutputCostGrad() const
{
	std::shared_lock lock(this->accessMutex);
	GP::out << "\nOutputs Cost Grad:\n" << this->OutputCostGrad.toString('[', ']', ' ', ';') << std::endl;
}

void NeuralNet::printLyrCostGrad() const
{
	std::shared_lock lock(this->accessMutex);
	auto layerID = size_t{ 0 };
	for (const auto& layer : this->Layers)
	{
		GP::out << "\nLayer: " << layerID++ << "\nWeight Costs:";
		GP::out << layer.getWeightCosts().toString('[', ']', ' ', ';');
		GP::out << "\nBias Costs: ";
		GP::out << layer.getBiasCosts().toString('[', ']', ' ', ';');
	}
	
	GP::out << std::endl;
}

void NeuralNet::printWBCostGrad() const
{
	std::shared_lock lock(this->accessMutex);
	GP::out << "\nWBCostGrad:\n" << this->WBCostGrad.toString('[', ']', ' ', ';') << std::endl;
}

#define NN_FILE_VERSION 1

void NeuralNet::headerSetupReadCallback(HeaderStatus& status, HeaderBuffer_t& header)
{
	this->tryToReadLayerNorm = false;
    switch (status) {
        case HeaderStatus::DEFAULT_HEADER_FIRST_WRITTEN:
        case HeaderStatus::DEFAULT_HEADER_FILE_UPDATE:
        case HeaderStatus::NON_OVERWRITE_HEADER_UPDATE:
			
			auto localVersion =  NeuralNet::fileVersioner(status, header);
			if (localVersion == NN_FILE_VERSION)
			{
				this->tryToReadLayerNorm = true;
			}
            break;
    }
}

uint32_t NeuralNet::fileVersioner(HeaderStatus& status, HeaderBuffer_t& header)
{
	auto localVersion = reinterpret_cast<const FileObjHeader*>(header.data())->version;
	if (localVersion == INVALID_HEADER_VERSION)
	{
		reinterpret_cast<FileObjHeader*>(header.data())->version = NN_FILE_VERSION;

		status = HeaderStatus::POST_CALLBACK_HEADER_REWRITE;
	}
	return localVersion;
}

void NeuralNet::headerSetupWriteCallback(HeaderStatus& status, HeaderBuffer_t& header) const
#ifdef UNIT_TEST_CONTEXT
{
	NeuralNet::g_headerSetupWriteCallback(status, header);
}

void NeuralNet::g_headerSetupWriteCallback(HeaderStatus& status, HeaderBuffer_t& header)
#endif
{
    switch (status) {
        case HeaderStatus::DEFAULT_HEADER_FIRST_WRITTEN:
        case HeaderStatus::DEFAULT_HEADER_FILE_UPDATE:
        case HeaderStatus::NON_OVERWRITE_HEADER_UPDATE:
			(void)NeuralNet::fileVersioner(status, header);
            break;
    }
}

void NeuralNet::saveFile() const
{
	std::shared_lock lock(this->accessMutex);
	auto FP = FileObjOutputOverwriteHandle(HEADER_CLASS_SETUP(&NeuralNet::headerSetupWriteCallback), this->FileName, BIN_FILE_MODE );

	FP << this->TrainingData.labelFile.string() << '\n';
	FP << this->TrainingData.imageFile.string() << '\n';
	FP << this->TestData.labelFile.string() << '\n';
	FP << this->TestData.imageFile.string() << '\n';

	this->DesiredOutputs.WriteToFile(FP);
	this->InputLayer.WriteToFile(FP);

	auto numLyrs = this->Layers.size();
	FileObj::write(FP, &numLyrs);
	for (const auto& layer : this->Layers)
		layer.WriteToFile(FP);

	FileObj::write(FP, &this->Cost);
	this->OutputCostGrad.WriteToFile(FP);
	this->WBCostGrad.WriteToFile(FP);

	FileObj::write(FP, &this->BatchSize);
	FileObj::write(FP, &this->NumBatches);
	FileObj::write(FP, &this->BatchesCompleted);

	FileObj::write(FP, &this->Eta);

	FileObj::write(FP, &this->af);
	FileObj::write(FP, &this->cf);
	FileObj::write(FP, &this->layerNorm);
}

void NeuralNet::readFile()
{

	string localFilePath = this->FileName.string();
	correctFilePathForTarget(localFilePath);

	auto FP = FileObjInputHandle(HEADER_CLASS_SETUP(&NeuralNet::headerSetupReadCallback), localFilePath, BIN_FILE_MODE);

	{
		std::unique_lock lock(this->accessMutex);

		string tmp;
		std::getline(FP, tmp); this->TrainingData.labelFile = tmp;
		std::getline(FP, tmp); this->TrainingData.imageFile = tmp;
		std::getline(FP, tmp); this->TestData.labelFile = tmp;
		std::getline(FP, tmp); this->TestData.imageFile = tmp;

		// GP::out << "localFilePath: " << localFilePath << "\n";
		std::filesystem::path baseDir = this->FileName.parent_path();
		// GP::out << "Base directory determined: " << baseDir << "\n";

		// State before adjustments
		// GP::out << "Original TrainingData.labelFile: " << this->TrainingData.labelFile << "\n";
		// GP::out << "Original TrainingData.imageFile: " << this->TrainingData.imageFile << "\n";
		// GP::out << "Original TestData.labelFile: " << this->TestData.labelFile << "\n";
		// GP::out << "Original TestData.imageFile: " << this->TestData.imageFile << "\n";
		// GP::out << std::endl
		this->TrainingData.labelFile = baseDir / this->TrainingData.labelFile;
		this->TrainingData.imageFile = baseDir / this->TrainingData.imageFile;
		this->TestData.labelFile = baseDir / this->TestData.labelFile;
		this->TestData.imageFile = baseDir / this->TestData.imageFile;


		// State before adjustments
		// td::cout << "After TrainingData.labelFile: " << this->TrainingData.labelFile << "\n";
		// td::cout << "After TrainingData.imageFile: " << this->TrainingData.imageFile << "\n";
		// td::cout << "After TestData.labelFile: " << this->TestData.labelFile << "\n";
		// td::cout << "After TestData.imageFile: " << this->TestData.imageFile << "\n";

		this->DesiredOutputs = Matrix(FP);
		this->InputLayer = Matrix(FP);

		auto numLyrs = size_t{ 0 };
		auto layerCount = size_t{ 0 };
		FileObj::read(FP, &numLyrs);
		this->Layers = std::vector<Layer>(numLyrs);
		auto* prevLayer = &(this->InputLayer);
		for (auto& layer : this->Layers)
			prevLayer = layer.InitLayer(this, layerCount++, FP);

		FileObj::read(FP, &this->Cost);
		this->OutputCostGrad = Matrix(FP);
		this->WBCostGrad = Matrix(FP);

		FileObj::read(FP, &this->BatchSize);
		FileObj::read(FP, &this->NumBatches);
		FileObj::read(FP, &this->BatchesCompleted);

		FileObj::read(FP, &this->Eta);
	}

	ActivationFunction af;
	FileObj::read(FP, &af);
	this->setActFunction(af);

	CostFunction cf;
	FileObj::read(FP, &cf);
	this->setCostFunction(cf);

	{
		std::unique_lock lock(this->accessMutex);
			
		if (this->tryToReadLayerNorm)
		{
			FileObj::read(FP, &this->layerNorm);
		}
	}
}

#if (DEBUG_NN_STATE_RECORDER)	
std::filesystem::path NeuralNet::getRecordRootPath() const
{
	std::shared_lock lock(this->accessMutex);
	return (this->recorder.getRecordRootPath());
}

void NeuralNet::setRecordingState(bool record)
{
	std::unique_lock lock(this->accessMutex);
	if (record)
	{
		this->recorder.enableRecording();
	}
	else
	{
		this->recorder.disableRecording();
	}
}
#endif

void NeuralNet::trainingSetup(bool setRandomWB)
{
	size_t trainDataSize = (this->InputLayer.getNumElements() + 1);
	try
	{
		this->batchImgContainer = std::vector<uint8_t>(this->BatchSize * this->getBatchImgSize(), 0);
		this->batchLblContainer = std::vector<uint8_t>(this->BatchSize * this->getBatchLblSize(), 0);
	}
	catch (const std::bad_alloc&) {
		ErrorHandler::FatalError("Could not allocate enough memory to contain training batch...");
	}

	if (this->trainLblDataFP && this->trainLblDataFP->is_open())
		this->trainLblDataFP->close();	
	if (this->trainImgDataFP && this->trainImgDataFP->is_open())
		this->trainImgDataFP->close();

	string localTrainLblDataFP = this->TrainingData.labelFile.string();
	correctFilePathForTarget(localTrainLblDataFP);
	this->trainLblDataFP = std::make_unique<FileObjHandle>(HEADER_NOT_NEEDED, localTrainLblDataFP, INPUT_BIN_FILE_MODE );

	string localTrainImgDataFP = this->TrainingData.imageFile.string();
	correctFilePathForTarget(localTrainImgDataFP);
	this->trainImgDataFP = std::make_unique<FileObjHandle>(HEADER_NOT_NEEDED, localTrainImgDataFP, INPUT_BIN_FILE_MODE );

	this->trainLblDataFP->validSeek(8);
	this->trainImgDataFP->validSeek(16);

	this->BatchesCompleted = 0;

	if (setRandomWB)
	{	
		GP::out << "\n[TRAINING] WB Randomization!" << std::endl;
		this->setRandWBAcrossNetwork();
	}
}

bool NeuralNet::setupNextTrainingBatch(bool shuffle)
{
	size_t trainNo = 0;
	bool eof = false;

	if (!this->trainLblDataFP || !(this->trainLblDataFP->is_open()) || !this->trainImgDataFP || !(this->trainImgDataFP->is_open()))
		ErrorHandler::FatalError("\nBatch input files not in good state");

	//load training data for batch
	while (!this->trainLblDataFP->eof() && !this->trainImgDataFP->eof() && trainNo < this->BatchSize 
			&& ((trainNo + 1) * this->getBatchImgSize()) <= this->batchImgContainer.size()
			&& ((trainNo + 1) * this->getBatchLblSize()) <= this->batchLblContainer.size())
	{
		bool success = true;
		try
		{
			FileObj::read(*(this->trainLblDataFP), &(this->batchLblContainer[trainNo * this->getBatchLblSize()]));
		} 
		catch ([[maybe_unused]] const EndOfFileException& e) 
		{
			success = false; 
			eof = true;
		}
		catch (...)
		{
			ErrorHandler::FatalError("\nBATCH READ ERROR");
		}

		for (size_t i = 0; i < this->InputLayer.getNumRows(); i++)
		{
			try
			{
				FileObj::read(*(this->trainImgDataFP), &(this->batchImgContainer[trainNo * this->getBatchImgSize() + i]));
			} 
			catch ([[maybe_unused]] const EndOfFileException& e)
			{
				success = false; 
				eof = true;
			}
			catch (...)
			{
				ErrorHandler::FatalError("\nBATCH READ ERROR");
			}
		}
		if (success)
		{
			trainNo++;
		}
	}
	//GP::out << "\nTraining data loaded!"  << std::endl;

	this->shuffledProcessingBatches = std::vector<size_t>(trainNo);
	std::iota(shuffledProcessingBatches.begin(), shuffledProcessingBatches.end(), 0);
	if (shuffle)
	{
		std::shuffle(shuffledProcessingBatches.begin(), shuffledProcessingBatches.end(), GlobalRand::rng_engine);
	}

	return (eof || trainLblDataFP->eof() || trainImgDataFP->eof());
}

void NeuralNet::performNextTrainingIteration(const size_t& lbl, const std::vector<double>& img)
{
	NN_DEBUG_RECORD_SETUP(nnStateType::eNonLayer_DesiredOutputs, nnOpInfo::eSetDesiredOutputs);
	NN_DEBUG_START_SUSPEND_RECORD(MatrixAPI::recordBeforeOp(this->DesiredOutputs));
	this->DesiredOutputs.setElementsToZero();
	this->DesiredOutputs.setLinElement(lbl, 1.0);	//will throw exceptin if output layer is too small
	NN_DEBUG_END_SUSPEND_RECORD(MatrixAPI::recordAfterOp(this->DesiredOutputs));

	NN_DEBUG_RECORD_SETUP(nnStateType::eNonLayer_InputLayer, nnOpInfo::eSetInputLayer);
	NN_DEBUG_START_SUSPEND_RECORD(MatrixAPI::recordBeforeOp(this->InputLayer));
	for (size_t i = 0; i < this->InputLayer.getNumRows(); i++)
		this->InputLayer.setLinElement(i, img[i]);
	NN_DEBUG_END_SUSPEND_RECORD(MatrixAPI::recordAfterOp(this->InputLayer));

	this->updateNetwork();
	this->addCost();
	this->computeBackpropErrors();
	this->addWBCosts();
}

bool NeuralNet::isProcessingHappening() {
	std::lock_guard<std::mutex> lock(this->process.processingMutex);
	return (this->process.isProcessing);
}

bool NeuralNet::hasProcessingStopBeenRequested() {
	std::lock_guard<std::mutex> lock(this->process.processingMutex);
	return (this->process.stopProcessing);
}

void NeuralNet::trainingWorker(NeuralNet* nnet, NeuralNetLabState_t& nnls) 
{
	SCOPE_ENABLE_SERIOUS_FP_EXCEPTIONS();
	try {
		nnet->userInteractiveTraining(nnls);
	}
	catch (const std::exception & e) {
		ErrorHandler::FatalError(string(string("Training error: ") + string(e.what())));
	}

	{
		// Ensure `isTraining` is reset when training is complete
		std::lock_guard<std::mutex> lock(nnet->process.processingMutex);
		nnet->process.isProcessing = false; // Mark training as completed
	}
	nnet->process.processStoppedCV.notify_all(); // Notify waiting stopProcess
};

// Start training on a separate thread
void NeuralNet::startTraining(NeuralNetLabState_t& nnls) {
	
	this->stopProcess(NeuralNet::ProcessingTypes::Any);

	{
		std::lock_guard<std::mutex> lock(this->process.processingMutex);
		if (this->process.isProcessing) {
			GP::out << "\n[ERROR] processing is already running!" << std::endl;
			return;
		} else {
			this->stopProcess(NeuralNet::ProcessingTypes::Training);
		}
		this->process.isProcessing = true;
		this->process.currentProcess = NeuralNet::ProcessingTypes::Training;
	}


	{
		

#if (USE_THREADPOOL)
    	SubmitToTaskingSystem("startTraining", std::bind(trainingWorker, this, nnls));
#else
    	std::thread workerThread(std::bind(trainingWorker, this, nnls));
#endif
#if (!USE_THREADPOOL)
    	// Detach so it runs independently
    	workerThread.detach();
#endif
		
	}
}

void NeuralNet::testingWorker(NeuralNet* nnet, NeuralNetLabState_t& nnls)
{
	SCOPE_ENABLE_SERIOUS_FP_EXCEPTIONS();
	try {
		nnet->userInteractiveTesting(nnls);
	}
	catch (const std::exception & e) {
		ErrorHandler::FatalError(string(string("Testing error: ") + string(e.what())));
	}

	{
		// Ensure `isProcessing` is reset when processing is complete
		std::lock_guard<std::mutex> lock(nnet->process.processingMutex);
		nnet->process.isProcessing = false; // Mark processing as completed
	}
	nnet->process.processStoppedCV.notify_all(); // Notify waiting stopProcess
};

// Start testing on a separate thread
void NeuralNet::startTesting(NeuralNetLabState_t& nnls) {
	
	this->stopProcess(NeuralNet::ProcessingTypes::Any);

	{
		std::lock_guard<std::mutex> lock(this->process.processingMutex);
		if (this->process.isProcessing) {
			GP::out << "\n[ERROR] processing is already running!" << std::endl;			return;
		} else {
			this->stopProcess(NeuralNet::ProcessingTypes::Testing);
		}
		this->process.isProcessing = true;
		this->process.currentProcess = NeuralNet::ProcessingTypes::Testing;
	}


	{
#if (USE_THREADPOOL)
    	SubmitToTaskingSystem("startTesting", std::bind(testingWorker, this, nnls));
#else
    	std::thread workerThread(std::bind(testingWorker, this, nnls));
#endif
#if (!USE_THREADPOOL)
    	// Detach so it runs independently
    	workerThread.detach();
#endif
	}
}

// Request training to stop
void NeuralNet::stopProcess(NeuralNet::ProcessingTypes expectedProcessing) {


	if ((expectedProcessing != NeuralNet::ProcessingTypes::Any) && (this->process.currentProcess != expectedProcessing) || (this->process.currentProcess == NeuralNet::ProcessingTypes::None))
		return;
	try {
		std::lock_guard<std::mutex> lock(this->process.processingMutex);
		this->process.stopProcessing = true;
	}
	catch (const std::exception& e) {
		ErrorHandler::FatalError(string("Couldn't lock! :") + string(e.what())); 
	}

	
	#if (!USE_THREADPOOL)
	//We do not manage thread lifetime when we use threadpool
	if (this->process.processingThread.joinable())
	#endif
	{
		std::unique_lock<std::mutex> lock(this->process.processingMutex);
		bool stopped = this->process.processStoppedCV.wait_for(
			lock, std::chrono::seconds(5), // Timeout of 5 seconds
			[this]() { return !this->process.isProcessing; });

		if (!stopped) {
			ErrorHandler::FatalError(string("Warning: Processing did not stop within the timeout period."));
		}

		#if (!USE_THREADPOOL)
		this->process.processingThread.join(); // Ensure the thread is joined even if it times out
		this->process.processingThread = std::thread(); // Reset to a default, non-running state
		#endif
	} 

	{	
		std::lock_guard<std::mutex> lock(this->process.processingMutex);
		this->process.stopProcessing = false;
		this->process.currentProcess = NeuralNet::ProcessingTypes::None;
	}
}


ProcessingRecord_t& ProcessingRecord_t::operator=(const ProcessingRecord_t& from)
{
	this->procIdx		 	= from.procIdx;
	this->procLabel		 	= from.procLabel;
	this->inputImage	 	= std::vector<double>(from.inputImage);
	this->guess			 	= from.guess;
	this->percentCorrect 	= from.percentCorrect;
	this->totalCount	 	= from.totalCount;
	this->outputActivations = Matrix(from.outputActivations);
	return (*this);
}

ProcessingRecord_t::ProcessingRecord_t(FileObjHandle& fp)
{
	size_t imageSize 			= 0;
	FileObj::read(fp, &this->procIdx);
	FileObj::read(fp, &this->procLabel);
	FileObj::read(fp, &imageSize);
	for (auto i= 0; i < imageSize; i++)
	{
		double elem = 0.0;
		FileObj::read(fp, &elem);
		this->inputImage.push_back(elem);
	}
	FileObj::read(fp, &this->guess);
	FileObj::read(fp, &this->percentCorrect);
	FileObj::read(fp, &this->totalCount);
	this->outputActivations = Matrix(fp);
}

void ProcessingRecord_t::WriteToFile(FileObjHandle& fp) const
{
	size_t imageSize 			= this->inputImage.size();

	FileObj::write(fp, &this->procIdx);
	FileObj::write(fp, &this->procLabel);
	FileObj::write(fp, &imageSize);
	for (const auto& elem : this->inputImage)
		FileObj::write(fp, &elem);
	FileObj::write(fp, &this->guess);
	FileObj::write(fp, &this->percentCorrect);
	FileObj::write(fp, &this->totalCount);
	this->outputActivations.WriteToFile(fp);
}

void NeuralNet::userInteractiveTraining(NeuralNetLabState_t& nnls)
{
	static size_t percentCorrect = 0;
	static size_t totalCount = 0;	
	static bool batchEOF=false;

	size_t epochCount = nnls.repeatCount;
	nnls.epochsRemaining = epochCount;
	while (epochCount)
	{
		size_t currentBatchesRemaining=nnls.batchStep;
		size_t currentTrainingsRemaining=nnls.processingStep;

		if (this->hasProcessingStopBeenRequested())
		{
			
			std::lock_guard<std::mutex> lock(this->process.processingMutex);
			this->process.stopProcessing=false;
			return ;
		}

		if (nnls.lastProcessType != NeuralNet::ProcessingTypes::Training)
		{
			nnls.processingInitialized = false;
		}

		if (!nnls.processingInitialized || nnls.processingRunEntire)
		{
			GP::out << "\nReinit training!" << std::endl;
			nnls.currentPercentCorrect = double(0.0);
			nnls.currentCost = double(0.0);
			bool randomize = nnls.randomizeWeightsBiases;
			if (nnls.wbHaveBeenRandomized && (nnls.wbRandomizationSelection == ECBR_EcTy(NeuralNetLabState_t::eWBRandomizationChoices, eOnFirstEpoch)))
			{
				randomize = false;
			}
			this->trainingSetup(randomize);
			nnls.wbHaveBeenRandomized = randomize || nnls.wbHaveBeenRandomized;
			nnls.processingInitialized = true;
			nnls.lastProcessType = NeuralNet::ProcessingTypes::Training;
			if (this->setupNextTrainingBatch(nnls.shuffleSets))
			{
				ErrorHandler::FatalError("Shouldnt be EOF right after loading...whats the batch size?");
			}
			else
			{
				batchEOF=false;
			}

			this->resetWBCostGrad();
			this->resetCost();
			percentCorrect = 0;
			totalCount = 0;
		}

		if (nnls.processingRunEntire)
		{
			nnls.processingStep = this->BatchSize;
			currentBatchesRemaining=this->NumBatches;
		}

		if (this->hasProcessingStopBeenRequested())
		{
			
			std::lock_guard<std::mutex> lock(this->process.processingMutex);
			this->process.stopProcessing=false;
			return ;
		}	

		while (/*(this->NumBatches < 0 || this->BatchesCompleted < this->NumBatches) &&*/ ((currentBatchesRemaining > 0) || (currentTrainingsRemaining > 0)))
		{

			size_t trainNo = this->shuffledProcessingBatches.size();
			if (nnls.processingStep > 0)
				trainNo = std::min(trainNo, nnls.processingStep);

			if (this->hasProcessingStopBeenRequested())
			{
				
				std::lock_guard<std::mutex> lock(this->process.processingMutex);
				this->process.stopProcessing=false;
				return ;
			}	

			size_t tr = 0;
			bool updateNN = true;
			for (; tr < trainNo; tr++)
			{
				if (tr >= shuffledProcessingBatches.size())	//This shouldn't happen, but lets abort if it does
				{
					currentBatchesRemaining = 0;
					currentTrainingsRemaining = 0;
					break;
				}
				const size_t lbl = static_cast<size_t>(this->batchLblContainer[this->shuffledProcessingBatches[tr]*this->getBatchLblSize()]);
				auto img = std::vector<double>(); 
				for (auto i = 0; i < this->getBatchImgSize(); i++)
				{
					img.push_back(static_cast<double>(this->batchImgContainer[this->shuffledProcessingBatches[tr]*this->getBatchImgSize() + i])/255.0);
				}

				this->performNextTrainingIteration(lbl, img);

				size_t guess = this->Layers.back().getActivations().getMaxValuePos().first;
				totalCount++;
				if (guess == lbl)
					percentCorrect++;
					nnls.currentPercentCorrect = static_cast<double>(percentCorrect) / static_cast<double>(totalCount) * 100.0;

				if (nnls.trainingResultFlush && directoryExists(nnls.getAndCreateProcessingResultPath()))
				{                            
					const auto results = nnls.getProcessingResultPath() / NeuralNet::trainingRecordFilename(); 
					auto outputfile = FileObjOutputAppendHandle(HEADER_SETUP(defaultHeaderSetupCallback), results, BIN_FILE_MODE );
					if (outputfile)
					{
						ProcessingRecord_t record = {};
						record.procIdx 					= tr;
						record.procLabel 				= lbl;
						record.inputImage 				= img;
						record.guess 		  			= guess;
						record.percentCorrect 			= percentCorrect;
						record.totalCount 				= totalCount;
						record.outputActivations		= Matrix(this->Layers.back().getActivations());
						record.WriteToFile(outputfile);
						outputfile.close();
					}
				}

				//GP::out << "\nBatch #: " + std::to_string(this->BatchesCompleted) + "\tTraining #:" + std::to_string(tr) << std::endl;
				//GP::out << "\nCurrent Batch #: " + std::to_string(currentBatchesRemaining) + "\tcurrent Training #:" + std::to_string(currentTrainingsRemaining) << std::endl;
				
				if (currentBatchesRemaining == 0)
					if (currentTrainingsRemaining > 0)
						currentTrainingsRemaining--;
				
				
				if (this->hasProcessingStopBeenRequested())
				{
					
					std::lock_guard<std::mutex> lock(this->process.processingMutex);
					this->process.stopProcessing=false;
					return ;
				}	
			}

			if (batchEOF)	//Training done, lets leave
			{
				currentBatchesRemaining = 0;
				currentTrainingsRemaining = 0;
				break;
			}

			if (currentBatchesRemaining > 0)
				currentBatchesRemaining--;
		
			if ((currentBatchesRemaining > 0 || currentTrainingsRemaining > 0) || nnls.completeLastBatch)
			{
				this->avgCost(tr);
				nnls.currentCost = this->Cost;
				this->avgWBCosts(tr, nnls.currentEta);
				this->applyGrad();
				this->BatchesCompleted++;
			}
				
			if (currentBatchesRemaining == 0 && currentTrainingsRemaining == 0)
			{
				break;
			}
			else
			{
				batchEOF = this->setupNextTrainingBatch();
				this->resetWBCostGrad();
				this->resetCost();
			}
		}

		epochCount--;
		nnls.epochsRemaining = epochCount; 
	}
	return;
}

void NeuralNet::training()
{
	this->trainingSetup();

	while ((!trainLblDataFP->eof() && !trainImgDataFP->eof()) && (this->NumBatches < 0 || this->BatchesCompleted < this->NumBatches))
	{
		this->setupNextTrainingBatch();
		
		size_t trainNo = this->shuffledProcessingBatches.size();
		for (size_t tr = 0; tr < trainNo; tr++)
		{
			const size_t lbl = this->batchLblContainer[this->shuffledProcessingBatches[tr]*this->getBatchLblSize()];
			auto img = std::vector<double>(); 
			for (auto i = 0; i < this->getBatchImgSize(); i++)
			{
				img.push_back(static_cast<double>(this->batchImgContainer[this->shuffledProcessingBatches[tr]*this->getBatchImgSize() + i])/255.0);
			}
			this->performNextTrainingIteration(lbl, img);
			
			//this->printActivations();
			//this->printLayers();
			//this->printOutputs();
			//this->printOutputCostGrad();
			//this->printLyrCostGrad();
			//this->printWBCostGrad();*/
		}

		this->avgCost(trainNo);
		this->printCost();
		this->avgWBCosts(trainNo);
		this->applyGrad();
		this->BatchesCompleted++;
	}

}

void NeuralNet::testingSetup()
{
	try
	{
		this->batchImgContainer = std::vector<uint8_t>(this->BatchSize * this->getBatchImgSize(), 0);
		this->batchLblContainer = std::vector<uint8_t>(this->BatchSize * this->getBatchLblSize(), 0);
	}
	catch (const std::bad_alloc&) {
		ErrorHandler::FatalError("Could not allocate enough memory to contain training batch...");
	}


	string localTestLblDataFP = this->TestData.labelFile.string();
	correctFilePathForTarget(localTestLblDataFP);
	this->testLblDataFP = std::make_unique<FileObjHandle>(HEADER_NOT_NEEDED, localTestLblDataFP, INPUT_BIN_FILE_MODE );

	string localTestImgDataFP = this->TestData.imageFile.string();
	correctFilePathForTarget(localTestImgDataFP);
	this->testImgDataFP = std::make_unique<FileObjHandle>(HEADER_NOT_NEEDED, localTestImgDataFP, INPUT_BIN_FILE_MODE );

	this->testLblDataFP->validSeek(8);
	this->testImgDataFP->validSeek(16);
	
   	this->BatchesCompleted = 0;
}

bool NeuralNet::setupNextTestingBatch(bool shuffle)
{
	size_t testNo = 0;
	bool eof = false;
	while (!testLblDataFP->eof() && !testImgDataFP->eof() && testNo < this->BatchSize 
			&& ((testNo + 1) * this->getBatchImgSize()) <= this->batchImgContainer.size()
			&& ((testNo + 1) * this->getBatchLblSize()) <= this->batchLblContainer.size())
	{
		bool success = true;
		try
		{
			FileObj::read(*(this->testLblDataFP), &(this->batchLblContainer[testNo]));
		}
		catch ([[maybe_unused]] const EndOfFileException& e) 
		{
			success = false; 
			eof = true;
		}
		catch (...)
		{
			ErrorHandler::FatalError("\nBATCH READ ERROR");
		}

		for (size_t i = 0; i < this->InputLayer.getNumRows(); i++)
		{
			try
			{
				FileObj::read(*(this->testImgDataFP), &(this->batchImgContainer[testNo * this->getBatchImgSize() + i]));
			}
			catch ([[maybe_unused]] const EndOfFileException& e)
			{
				success = false; 
				eof = true;
			}
			catch (...)
			{
				ErrorHandler::FatalError("\nBATCH READ ERROR");
			}
		}
		if (success)
		{
			testNo++;
		}
	}
	//GP::out << "\nTest data loaded!" << std::endl;

	this->shuffledProcessingBatches = std::vector<size_t>(testNo);
	std::iota(shuffledProcessingBatches.begin(), shuffledProcessingBatches.end(), 0);
	if (shuffle)
	{
		std::shuffle(shuffledProcessingBatches.begin(), shuffledProcessingBatches.end(), GlobalRand::rng_engine);
	}
	return (eof || testLblDataFP->eof() && testImgDataFP->eof());
}

void NeuralNet::performNextTestingIteration(const size_t& lbl, const std::vector<double>& img)
{
	NN_DEBUG_RECORD_SETUP(nnStateType::eNonLayer_DesiredOutputs, nnOpInfo::eSetDesiredOutputs);
	NN_DEBUG_START_SUSPEND_RECORD(MatrixAPI::recordBeforeOp(this->DesiredOutputs));
	this->DesiredOutputs.setElementsToZero();
	this->DesiredOutputs.setLinElement(lbl, 1.0);	//will throw exception if output layer is too small
	NN_DEBUG_END_SUSPEND_RECORD(MatrixAPI::recordAfterOp(this->DesiredOutputs));

	NN_DEBUG_RECORD_SETUP(nnStateType::eNonLayer_InputLayer, nnOpInfo::eSetInputLayer);
	NN_DEBUG_START_SUSPEND_RECORD(MatrixAPI::recordBeforeOp(this->InputLayer));
	for (size_t i = 0; i < this->InputLayer.getNumRows(); i++)
		this->InputLayer.setLinElement(i, img[i]);
	NN_DEBUG_END_SUSPEND_RECORD(MatrixAPI::recordAfterOp(this->InputLayer));

	this->updateNetwork();
	this->addCost();
}

void NeuralNet::userInteractiveTesting(NeuralNetLabState_t& nnls)
{
	static double percentCorrect = 0;
	static size_t totalTests = 0;	
	static bool batchEOF = false;
	size_t currentBatchesRemaining=nnls.batchStep;
	size_t currentTestsRemaining=nnls.processingStep;

	if (this->hasProcessingStopBeenRequested())
	{
		
		std::lock_guard<std::mutex> lock(this->process.processingMutex);
		this->process.stopProcessing=false;
		return;
	}

	if (nnls.lastProcessType != NeuralNet::ProcessingTypes::Testing)
	{
		nnls.processingInitialized = false;
		nnls.lastProcessType = NeuralNet::ProcessingTypes::None;
	}

	if (!nnls.processingInitialized || nnls.processingRunEntire)
	{
		GP::out << "\nReinit testing!" << std::endl;
		totalTests=0;
		percentCorrect=0;
		nnls.currentPercentCorrect = double(0.0);
		nnls.currentCost = double(0.0);
		this->testingSetup();
		nnls.processingInitialized = true;
		nnls.lastProcessType = NeuralNet::ProcessingTypes::Testing;
		if (this->setupNextTestingBatch(nnls.shuffleSets))
		{
			ErrorHandler::FatalError("Shouldnt have EOF so soon!");
		}
		else
		{
			batchEOF = false;
		}

		this->resetCost();
	}

	if (nnls.processingRunEntire)
	{
		nnls.processingStep = this->BatchSize;
		currentBatchesRemaining=this->NumBatches;
	}

	if (this->hasProcessingStopBeenRequested())
	{
		
		std::lock_guard<std::mutex> lock(this->process.processingMutex);
		this->process.stopProcessing=false;
		return;
	}	


	while (/*(this->NumBatches < 0 || this->BatchesCompleted < this->NumBatches) &&*/ ((currentBatchesRemaining > 0) || (currentTestsRemaining > 0)))
	{
		size_t testNo = this->shuffledProcessingBatches.size();
		if (nnls.processingStep > 0)
			testNo = std::min(testNo, nnls.processingStep);

		if (this->hasProcessingStopBeenRequested())
		{
			
			std::lock_guard<std::mutex> lock(this->process.processingMutex);
			this->process.stopProcessing=false;
			return;
		}	

		size_t tr = 0;
		bool updateNN = true;
		size_t testsRun=0;
		for (; tr < testNo; tr++)
		{
			if (tr >= shuffledProcessingBatches.size())	//This sholdn't happen, but lets abort if it does
			{
				currentBatchesRemaining = 0;
				currentTestsRemaining = 0;
				break;
			}

			const size_t lbl = static_cast<size_t>(this->batchLblContainer[this->shuffledProcessingBatches[tr]*this->getBatchLblSize()]);
			auto img = std::vector<double>(); 
			for (auto i = 0; i < this->getBatchImgSize(); i++)
			{
				img.push_back(static_cast<double>(this->batchImgContainer[this->shuffledProcessingBatches[tr]*this->getBatchImgSize() + i])/255.0);
			}

			this->performNextTestingIteration(lbl, img);
			
			size_t guess = this->Layers.back().getActivations().getMaxValuePos().first;
			totalTests++;
			if (guess == lbl)
				percentCorrect++;
				nnls.currentPercentCorrect = static_cast<double>(percentCorrect) / static_cast<double>(totalTests) * 100.0;
			testsRun++;


			if (nnls.testingResultFlush && directoryExists(nnls.getAndCreateProcessingResultPath()))
			{                            
				const auto results = nnls.getProcessingResultPath() / NeuralNet::testingRecordFilename(); 
				auto outputfile = FileObjOutputAppendHandle(HEADER_SETUP(defaultHeaderSetupCallback), results, BIN_FILE_MODE );
				if (outputfile)
				{
					ProcessingRecord_t record = {};
					record.procIdx 					= tr;
					record.procLabel 				= lbl;
					record.inputImage 				= img;
					record.guess 		  			= guess;
					record.percentCorrect 			= static_cast<size_t>(percentCorrect);
					record.totalCount 				= totalTests;
					record.outputActivations		= Matrix(this->Layers.back().getActivations());
					record.WriteToFile(outputfile);
					outputfile.close();
				}
			}

			//GP::out << "\nBatch #: " + std::to_string(this->BatchesCompleted) + "\tTesting #:" + std::to_string(tr) << std::endl;
			//GP::out << "\nCurrent Batch #: " + std::to_string(currentBatchesRemaining) + "\tcurrent Testing #:" + std::to_string(currentTestsRemaining) << std::endl;
			
			if (currentBatchesRemaining == 0)
				if (currentTestsRemaining > 0)
					currentTestsRemaining--;
			

			if (this->hasProcessingStopBeenRequested())
			{
				
				std::lock_guard<std::mutex> lock(this->process.processingMutex);
				this->process.stopProcessing=false;
				return;
			}	
		}

		if (batchEOF)	//exhausted trainings, stop the training, make user restart
		{
			currentBatchesRemaining = 0;
			currentTestsRemaining = 0;
		}

		if (currentBatchesRemaining > 0)
			currentBatchesRemaining--;
	
		this->BatchesCompleted++;

		if ((currentBatchesRemaining > 0 || currentTestsRemaining > 0) || nnls.completeLastBatch)
		{
			this->avgCost(testsRun); //[crc] investigate if this is right once training is working
			
			nnls.currentCost = this->Cost;
		}
			
		if (currentBatchesRemaining == 0 && currentTestsRemaining == 0)
		{
			break;
		}
		else
		{
			batchEOF = this->setupNextTestingBatch(nnls.shuffleSets);
			this->resetCost();
		}
	}
	return;
}

double NeuralNet::testing(int32_t numBatches)
{
	this->testingSetup();

	double percentCorrect = 0;
	size_t totalTests = 0;
	int32_t batchNo = 0;
	while ((!testLblDataFP->eof() && !testImgDataFP->eof()) && (numBatches < 0 || batchNo < this->NumBatches))
	{
		this->resetCost();

		this->setupNextTestingBatch();
		
		size_t testNo = this->shuffledProcessingBatches.size();

		for (size_t tr = 0; tr < testNo; tr++)
		{

			const size_t lbl = static_cast<size_t>(this->batchLblContainer[this->shuffledProcessingBatches[tr]*this->getBatchLblSize()]);
			auto img = std::vector<double>(); 
			for (auto i = 0; i < this->getBatchImgSize(); i++)
			{
				img.push_back(static_cast<double>(this->batchImgContainer[this->shuffledProcessingBatches[tr]*this->getBatchImgSize() + i])/255.0);
			}

			this->performNextTestingIteration(lbl, img);

			size_t guess = this->Layers.back().getActivations().getMaxValuePos().first;
			if (guess == lbl)
				percentCorrect++;
			totalTests++;
		}

		this->avgCost(testNo); //[crc] investigate if this is right once training is working
		batchNo++;
	}

	return (static_cast<double>(percentCorrect) / static_cast<double>(totalTests) * 100.0);
	//GP::out << "\nPercent Correct: " << (percentCorrect / totalTests) * 100 << "%" << std::endl;
}

double NeuralNet::sigmoid(const double& x)
{
	return 1 / (1 + std::exp(-x));
}

double NeuralNet::ReLU(const double& x)
{
	return (std::max(x, (0.1 * x)));
}
//	return (x > 0) ? x : 0;
//}

double NeuralNet::sigmoid_deriv(const double& x)
{
	return exp(-x) / (pow(exp(-x) + 1, 2));
	//return (NeuralNet::sigmoid(x) * (1 - NeuralNet::sigmoid(x)));
}

double NeuralNet::ReLU_deriv(const double& x)
{
	return (x > 0) ? 1 : -0.1;
}

double NeuralNet::quadratic(const Matrix& Y, const Matrix& A)
{
	auto diff = Y - A;

	try
	{		
		double squareSum = 0;
		for (size_t i = 0; i < diff.getNumElements(); i++)
			squareSum += (diff.getLinElement(i)*diff.getLinElement(i));
		return (squareSum/static_cast<double>(diff.getNumElements()));
		//const auto prod = Matrix::Mult(diff, diff, MatCond::Transpose);
		//const auto intRes = prod.getLinElement(0);
		//return (intRes / 2);
	}
	catch(...) { ErrorHandler::FatalError(); }

	return 0.0f;
}

Matrix NeuralNet::quadratic_partialderiv(const Matrix& Y, const Matrix& A)
{
	return A - Y;
}

NeuralNet::NeuralNet(const NeuralNetParams_t& nnp) : process(this)
{
	const std::filesystem::path& filename 	= nnp.filename;
	std::vector<size_t> 		layerSzs 	= nnp.layerSizes;
	const size_t& 				batchsz 	= nnp.batchSize;
	const int& 					numbatches 	= nnp.numBatches;
	const double& 				eta 		= nnp.eta;
	const mnistFileInfo& 		train 		= nnp.trainingData;
	const mnistFileInfo& 		test 		= nnp.testData;
	const ActivationFunction& 	af 			= nnp.activationFunction;
	const CostFunction& 		cf 			= nnp.costFunction;

	if (
		isValidNumLayer(layerSzs.size()) && 
		isValidBatchSize(batchsz) && 
		isValidEta(eta)
		)
	{
		
		{
			std::unique_lock lock(this->accessMutex);
			this->FileName = filename;
			#if(DEBUG_NN_STATE_RECORDER)
			this->recorder.setNNetFile(this->FileName);
			#endif
			this->TrainingData = train;
			this->TestData = test;
		}

		this->verifyFiles();

		{
			std::unique_lock lock(this->accessMutex);
			this->Cost = 0;

			this->BatchSize = batchsz;
			this->NumBatches = numbatches;
			this->BatchesCompleted = 0;

			this->Eta = eta;

			//Initializes Neuron Layers
			this->InputLayer = Matrix(this->img_rows * this->img_cols, 1);
			this->Layers = std::vector<Layer>(layerSzs.size());

			auto* prevLayer = &(this->InputLayer);
			auto prevLayerSize = this->InputLayer.getNumRows();
			auto index = size_t{ 0 };
			auto costGradLength = size_t{ 0 };
			for (const auto& size : layerSzs)
			{
				if (!isValidLayerSize(size))
				{
					ErrorHandler::FatalError(string(string("INVALID LAYER SIZE") + std::to_string(size)));
					break;
				}

				try
				{
					prevLayer = this->Layers[index].InitLayer(this, index, size);
					costGradLength += size * (prevLayerSize + 1);
					prevLayerSize = size;
				}
				catch (...) { ErrorHandler::FatalError("LAYER INITIALIZATION FAILED"); }

				index++;
			}

			this->WBCostGrad = Matrix(costGradLength, 1);
			this->DesiredOutputs = Matrix(layerSzs.back(), 1);
			this->OutputCostGrad = Matrix();
			this->lclRandDistrbution=GlobalRand::getLocalRealDistribution(0.0, std::sqrt((double)(this->InputLayer.getNumElements())/((double)(layerSzs.back()))));

		}

		this->setActFunction(af);
		this->setCostFunction(cf);


		if (!nnp.disableInitialWBRandomization)
			this->setRandWBAcrossNetwork();		
	}
	else
	{

		{
			std::unique_lock lock(this->accessMutex);
			this->Cost = 0;
			this->BatchSize = 0;
			this->Eta = 0;

			this->InputLayer = Matrix();
			this->Layers = std::vector<Layer>();

			this->WBCostGrad = Matrix();
			this->OutputCostGrad = Matrix();
			this->DesiredOutputs = Matrix();
		}

		setActFunction((ActivationFunction)0);
		setCostFunction((CostFunction)0);

		ErrorHandler::FatalError("INVALID NN PARAMS");
	}
}

NeuralNet::NeuralNet(const std::filesystem::path& filename, NeuralNetParams_t& nnp) : process(this)
{
	{
		std::unique_lock lock(this->accessMutex);
		this->FileName = filename;
#if(DEBUG_NN_STATE_RECORDER)
		this->recorder.setNNetFile(this->FileName);
#endif
	}
 	this->readFile();
	this->verifyFiles();

	{
		std::shared_lock lock(this->accessMutex);

		std::vector<size_t> layerSizes = {};
		for (size_t layer = 0; layer < this->Layers.size(); layer++)
		{
			size_t numElements = this->Layers[layer].getActivations().getNumElements();
			layerSizes.push_back(size_t(numElements));
		}

		nnp.filename = this->FileName;
		nnp.layerSizes = layerSizes;
		nnp.batchSize = this->BatchSize;
		nnp.numBatches = this->NumBatches;
		nnp.eta = this->Eta;
		nnp.trainingData = this->TrainingData;
		nnp.testData = this->TestData;
		nnp.activationFunction = this->af;
		nnp.costFunction = this->cf;

		//ctrl behavior not saved to NN file
		nnp.disableInitialWBRandomization = false;
	}
	GP::out << "\nNeural Net successfully loaded from file!" << std::endl;
}

bool NeuralNet::isValidLayerSize(const size_t& layerSize)
{
	return ((layerSize >= AP_MinLayerSize) && (layerSize <= AP_MaxLayerSize));
}

bool NeuralNet::isValidNumLayer(const size_t& numLayer)
{
	return ((numLayer >= AP_MinNumLayer) && (numLayer <= AP_MaxNumLayer));
}

bool NeuralNet::isValidBatchSize(const size_t& batchSize)
{
	return ((batchSize >= AP_MinBatchCount) && (batchSize <= AP_MaxBatchCount));
}

bool NeuralNet::isValidStepSize(const size_t& stepSize)
{
	return ((stepSize >= AP_MinStepSize) && (stepSize <= AP_MaxStepSize));
}

bool NeuralNet::isValidEpochCount(const size_t& epochCount)
{
	return ((epochCount >= AP_MinEpochCount) && (epochCount <= AP_MaxEpochCount));
}

bool NeuralNet::isValidBatchCount(const int& batchNum)
{
	return ((batchNum == AP_AllBatchesSentinel) || ((batchNum >= AP_MinBatchCount) && (batchNum <= AP_MaxBatchCount)));
}

bool NeuralNet::isValidEta(const double& eta)
{
	return (std::isfinite(eta) && ((eta >= AP_MinEta) && (eta <= AP_MaxEta)));
}

void NeuralNetLabState_t::setCurrentTrainingResults(size_t idx, const ProcessingRecord_t& rec)
{
	this->trainingResults[idx] = rec;
} 

void NeuralNetLabState_t::setCurrentTestingResults(size_t idx, const ProcessingRecord_t& rec)
{
	this->testingResults[idx] = rec;
}

const ProcessingRecord_t& 	NeuralNetLabState_t::getCurrentTrainingResults() const
{
	return this->trainingResults[this->trainingResultsIdx % maxNumResults];
} 

const ProcessingRecord_t& 	NeuralNetLabState_t::getCurrentTestingResults() const
{
	return this->testingResults[this->testingResultsIdx % maxNumResults];
} 


const size_t& 	NeuralNetLabState_t::getCurrentTrainingResultsIdx() const
{
	return this->trainingResultsIdx;
}

const size_t& 	NeuralNetLabState_t::getCurrentTestingResultsIdx() const
{
	return this->testingResultsIdx;
}

void NeuralNetLabState_t::updateTrainingResultIdx(size_t idx)
{
	const auto targetChunkIdx = static_cast<size_t>((idx)/(maxNumResults));
	if (targetChunkIdx != this->trainingChunkIdx)
	{
		this->trainingChunkReloadNeeded = true;
		this->trainingChunkReloadIdx = targetChunkIdx;
	}
	
	this->trainingResultsIdx = idx;
}

void NeuralNetLabState_t::updateTestingResultIdx(size_t idx)
{
	const auto targetChunkIdx = static_cast<size_t>((idx)/(/*FIX*/maxNumResults));
	if (targetChunkIdx != this->testingChunkIdx)
	{
		this->testingChunkReloadNeeded = true;
		this->testingChunkReloadIdx = targetChunkIdx;
	}

	this->testingResultsIdx = idx;
}

void NeuralNetLabState_t::setProcessingResultPath(const std::filesystem::path& path)
{
	this->processingResultPath = path;
}

const std::filesystem::path& NeuralNetLabState_t::getProcessingResultPath() const
{
	return (this->processingResultPath);
}

const std::filesystem::path& NeuralNetLabState_t::getAndCreateProcessingResultPath()
{

	// Check if the directories exist, and create them if necessary
	if (!std::filesystem::exists(this->processingResultPath)) 
	{
		std::filesystem::create_directories(this->processingResultPath);
		GP::out << "\nDirectories created: " << this->processingResultPath << std::endl;
	} 

	if (!std::filesystem::exists(this->processingResultPath)) 
	{
		ErrorHandler::FatalError("Could not create directory!");
	}
	return (this->processingResultPath);
}

void	NeuralNetLabState_t::incrTrainingResultIdx()
{
	this->updateTrainingResultIdx(this->trainingResultsIdx + 1);
}	

void	NeuralNetLabState_t::decrTrainingResultIdx()
{
	this->updateTrainingResultIdx((this->trainingResultsIdx > 1) ? (this->trainingResultsIdx - 1) : (0));
}	

void	NeuralNetLabState_t::resetTrainingResultsIdx()
{

	this->updateTrainingResultIdx(0);
}	

void	NeuralNetLabState_t::setTrainingResultsIdx(size_t newIdx)
{

	this->updateTrainingResultIdx(newIdx);
}	

void	NeuralNetLabState_t::resetTrainingResults()
{
    std::fill(this->trainingResults.begin(), this->trainingResults.end(), ProcessingRecord_t{});
}	

void	NeuralNetLabState_t::incrTestingResultIdx()
{
	this->updateTestingResultIdx(this->testingResultsIdx + 1);
}	

void	NeuralNetLabState_t::decrTestingResultIdx()
{
	this->updateTestingResultIdx((this->testingResultsIdx > 1) ? (this->testingResultsIdx - 1) : (0));
}	

void	NeuralNetLabState_t::resetTestingResultsIdx()
{

	this->updateTestingResultIdx(0);
}	

void	NeuralNetLabState_t::setTestingResultsIdx(size_t newIdx)
{

	this->updateTestingResultIdx(newIdx);
}	

void	NeuralNetLabState_t::resetTestingResults()
{
    std::fill(this->testingResults.begin(), this->testingResults.end(), ProcessingRecord_t{});
}	